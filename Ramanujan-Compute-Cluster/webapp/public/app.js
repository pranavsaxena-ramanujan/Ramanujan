'use strict';
const $ = id => document.getElementById(id);
let selectedRoom;
let chats = new Map();
let pending = false;
let managementKey = localStorage.getItem('ramanujan-management-key');
async function api(url, body, method) {
  const response = await fetch(url, { method: method || (body ? 'POST' : 'GET'), credentials: 'same-origin',
    headers: { ...(body ? { 'Content-Type': 'application/json' } : {}), ...(managementKey ? { Authorization: `Bearer ${managementKey}` } : {}) },
    body: body ? JSON.stringify(body) : undefined });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || 'Request failed');
  return result;
}
function notice(error) { $('notice').textContent = error.message; $('notice').hidden = false; }
function clearNotice() { $('notice').hidden = true; }
function safe(fn) { return async event => { event?.preventDefault(); clearNotice(); try { await fn(event); } catch (error) { notice(error); } }; }
function roomPath(suffix) { return `/api/clusters/${encodeURIComponent(selectedRoom.room_id)}/${suffix}`; }
function message(role, content) {
  const entry = document.createElement('div'); entry.className = `message ${role}`;
  const text = document.createElement('div'); text.textContent = content;
  if (role === 'note') { entry.append(text); return entry; }
  const label = document.createElement('div'); label.className = 'message-label'; label.textContent = role === 'user' ? 'You' : 'Ramanujan';
  entry.append(label, text); return entry;
}
function renderChat() {
  $('messages').replaceChildren(...(chats.get(selectedRoom?.room_id) || []).map(item => message(item.role, item.text)));
  $('messages').scrollTop = $('messages').scrollHeight;
}
async function devices() {
  if (!selectedRoom) return;
  const list = await api(roomPath('devices'));
  $('device-list').replaceChildren();
  if (!list.length) { $('device-list').textContent = 'No devices joined yet. Download the client and enter this room ID.'; return; }
  for (const device of list) {
    const item = document.createElement('div'); item.className = 'device';
    const info = document.createElement('div'); info.textContent = device.name;
    const status = document.createElement('small');
    const date = device.last_seen ? new Date(String(device.last_seen).includes('T') ? device.last_seen : device.last_seen + 'Z') : null;
    const online = !device.revoked && date && Date.now() - date.getTime() < 60000;
    status.textContent = `${device.platform} - ${device.revoked ? 'Revoked' : online ? 'Online' : 'Offline'}`;
    if (online) status.className = 'online'; info.append(status); item.append(info);
    if (!device.revoked) {
      const revoke = document.createElement('button'); revoke.textContent = 'Revoke';
      revoke.addEventListener('click', safe(async () => { await api(roomPath(`devices/${device.id}`), null, 'DELETE'); await devices(); }));
      item.append(revoke);
    }
    $('device-list').append(item);
  }
}
async function selectRoom(room) {
  if (pending) throw new Error('Wait for the current answer before changing rooms.');
  selectedRoom = room; $('selected-room').textContent = room.name; $('room-id-label').textContent = `Room ID: ${room.room_id}`;
  document.querySelectorAll('.room-button').forEach(button => button.classList.toggle('active', button.dataset.id === room.id));
  renderChat(); ide.setRoom(room.room_id); await devices();
}
async function loadWorkspace() {
  const rooms = await api('/api/clusters');
  $('workspace').hidden = false;
  $('room-list').replaceChildren();
  for (const room of rooms) {
    const button = document.createElement('button'); button.className = 'room-button'; button.textContent = room.name; button.dataset.id = room.id;
    button.addEventListener('click', safe(() => selectRoom(room))); $('room-list').append(button);
  }
  await loadModels();
  const room = rooms.find(item => item.id === selectedRoom?.id) || rooms[0];
  if (room) await selectRoom(room);
}
let modelPoll;
const formatSize = bytes => bytes >= 1024 ** 3 ? `${(bytes / 1024 ** 3).toFixed(1)} GB` : `${Math.round(bytes / 1024 ** 2)} MB`;
const modelStates = { UPLOADING: 'Uploading', DOWNLOADING: 'Downloading', QUEUED: 'Waiting to convert', CONVERTING: 'Converting', READY: 'Ready', FAILED: 'Failed' };
async function loadModels() {
  const models = await api('/api/models');
  const select = $('model-select');
  const current = select.value || localStorage.getItem('ramanujan-model');
  const ready = models.filter(model => model.status === 'READY');
  select.replaceChildren();
  if (!ready.length) { const option = document.createElement('option'); option.textContent = 'No model configured'; option.value = ''; select.append(option); }
  for (const model of ready) { const option = document.createElement('option'); option.value = model.id; option.textContent = model.owned ? `${model.name} (yours)` : model.name; select.append(option); }
  if (ready.some(model => model.id === current)) select.value = current;
  const owned = models.filter(model => model.owned);
  $('model-list').replaceChildren();
  if (!owned.length) { const empty = document.createElement('small'); empty.textContent = 'You have not added any models yet.'; $('model-list').append(empty); }
  for (const model of owned) {
    const row = document.createElement('div'); row.className = 'model-row';
    const info = document.createElement('div'); info.className = 'model-info';
    const name = document.createElement('div'); name.textContent = model.name;
    const status = document.createElement('small'); status.className = `model-status ${model.status.toLowerCase()}`;
    const details = [modelStates[model.status] || model.status];
    if (['UPLOADING', 'DOWNLOADING', 'CONVERTING'].includes(model.status) && model.progress) details[0] += ` ${Math.round(model.progress * 100)}%`;
    if (model.sizeBytes) details.push(formatSize(model.sizeBytes));
    if (model.architecture) details.push(model.architecture);
    if (model.detail) details.push(model.detail);
    status.textContent = details.join(' \u00b7 ');
    info.append(name, status); row.append(info);
    if (['UPLOADING', 'DOWNLOADING', 'CONVERTING'].includes(model.status)) {
      const bar = document.createElement('progress'); bar.max = 1; bar.value = model.progress || 0; row.append(bar);
    }
    if (model.status !== 'UPLOADING') {
      const remove = document.createElement('button'); remove.type = 'button'; remove.textContent = 'Delete';
      remove.addEventListener('click', safe(async () => {
        if (!confirm(`Delete ${model.name}? Its converted files are removed from the server.`)) return;
        await api(`/api/models/${encodeURIComponent(model.id)}`, null, 'DELETE'); await loadModels();
      }));
      row.append(remove);
    }
    $('model-list').append(row);
  }
  clearTimeout(modelPoll);
  if (owned.some(model => ['DOWNLOADING', 'QUEUED', 'CONVERTING'].includes(model.status))) modelPoll = setTimeout(() => loadModels().catch(notice), 2000);
}
$('model-select').addEventListener('change', () => localStorage.setItem('ramanujan-model', $('model-select').value));
$('manage-models').addEventListener('click', () => {
  const open = $('models-panel').hidden;
  $('models-panel').hidden = !open; $('manage-models').setAttribute('aria-expanded', String(open));
});
function uploadModel(file, name, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `/api/models/upload?name=${encodeURIComponent(name)}`);
    xhr.setRequestHeader('Content-Type', 'application/octet-stream');
    if (managementKey) xhr.setRequestHeader('Authorization', `Bearer ${managementKey}`);
    xhr.upload.onprogress = event => { if (event.lengthComputable) onProgress(event.loaded / event.total); };
    xhr.onload = () => {
      let result = {};
      try { result = JSON.parse(xhr.responseText); } catch { /* non-JSON error page */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(result); else reject(new Error(result.error || `Upload failed (HTTP ${xhr.status})`));
    };
    xhr.onerror = () => reject(new Error('Upload failed; check your connection'));
    xhr.send(file);
  });
}
$('model-upload-form').addEventListener('submit', safe(async () => {
  const file = $('model-file').files[0];
  if (!file) throw new Error('Choose a .gguf file first.');
  const name = $('model-upload-name').value.trim() || file.name.replace(/\.gguf$/i, '');
  const bar = $('model-upload-progress');
  bar.hidden = false; bar.value = 0; $('model-upload').disabled = true; $('model-upload-status').textContent = `Uploading ${formatSize(file.size)}...`;
  try {
    await uploadModel(file, name.slice(0, 120), fraction => { bar.value = fraction; $('model-upload-status').textContent = `Uploading ${Math.round(fraction * 100)}% of ${formatSize(file.size)}`; });
    $('model-upload-status').textContent = 'Uploaded. Conversion progress is shown below.';
    $('model-upload-form').reset();
  } catch (error) { $('model-upload-status').textContent = ''; throw error; }
  finally { bar.hidden = true; $('model-upload').disabled = false; await loadModels().catch(notice); }
}));
$('model-url-form').addEventListener('submit', safe(async () => {
  await api('/api/models', { url: $('model-url').value.trim(), name: $('model-url-name').value.trim() || undefined });
  $('model-url-form').reset(); await loadModels();
}));
$('access-key').addEventListener('click', () => { $('key-panel').hidden = !$('key-panel').hidden; $('management-key').value = managementKey || ''; });
$('show-key').addEventListener('click', () => { $('management-key').type = $('management-key').type === 'password' ? 'text' : 'password'; });
$('restore-form').addEventListener('submit', safe(async () => {
  const previous = managementKey;
  managementKey = $('restore-key').value.trim();
  try {
    await api('/api/clusters');
    localStorage.setItem('ramanujan-management-key', managementKey);
    $('restore-key').value = ''; $('management-key').value = managementKey;
    selectedRoom = undefined; chats.clear(); await loadWorkspace();
  } catch (error) { managementKey = previous; throw error; }
}));
$('room-form').addEventListener('submit', safe(async () => {
  const room = await api('/api/clusters', { name: $('room-name').value }); $('room-name').value = '';
  $('new-room-id').value = room.roomId; $('new-room-secret').value = room.joinSecret; $('join-details').hidden = false;
  await loadWorkspace(); const rooms = await api('/api/clusters'); await selectRoom(rooms.find(item => item.id === room.clusterId));
}));
$('show-secret').addEventListener('click', () => { $('new-room-secret').type = $('new-room-secret').type === 'password' ? 'text' : 'password'; });
$('refresh-devices').addEventListener('click', safe(devices));
/** The conversation sent to the model: completed user/assistant exchanges plus the new question. */
function transcript(history) {
  const turns = [];
  for (const item of history) {
    if (item.role === 'user') { if (turns.at(-1)?.role === 'user') turns.pop(); turns.push({ role: 'user', content: item.text }); }
    else if (item.role === 'assistant' && turns.at(-1)?.role === 'user') turns.push({ role: 'assistant', content: item.text });
    else if (item.role === 'error' && turns.at(-1)?.role === 'user') turns.pop();
  }
  return turns;
}
$('chat-form').addEventListener('submit', safe(async () => {
  if (!selectedRoom) throw new Error('Create or select a room first.');
  if (pending) return;
  const roomId = selectedRoom.room_id;
  const question = $('question').value.trim();
  if (!question) return;
  const history = chats.get(roomId) || []; chats.set(roomId, history);
  history.push({ role: 'user', text: question }); $('question').value = ''; renderChat();
  pending = true; $('send').disabled = true; $('new-chat').disabled = true; $('send').textContent = 'Inferring...';
  try {
    const result = await api(roomPath('chat'), { messages: transcript(history), model: $('model-select').value });
    if (result.droppedTurns) history.push({ role: 'note', text: `The ${result.droppedTurns} oldest messages did not fit in the model's context window and were left out.` });
    history.push({ role: 'assistant', text: result.answer || '(The model returned no text.)' });
  } catch (error) { history.push({ role: 'error', text: error.message }); }
  finally { pending = false; $('send').disabled = false; $('new-chat').disabled = false; $('send').textContent = 'Send \u2192'; renderChat(); }
}));
$('new-chat').addEventListener('click', () => {
  if (pending || !selectedRoom) return;
  chats.delete(selectedRoom.room_id); renderChat(); $('question').focus();
});
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
async function runCode(project) {
  if (!selectedRoom) throw new Error('Select a room first.');
  const endpoint = roomPath('jobs');
  const submitted = await api(endpoint, project);
  const deadline = Date.now() + 60 * 60 * 1000;
  for (let failures = 0; Date.now() < deadline;) {
    await sleep(2000);
    let job;
    try { job = (await api(endpoint)).find(item => item.id === submitted.id); failures = 0; }
    catch (error) { if (++failures >= 5) throw error; continue; }
    if (!job) throw new Error('This run no longer exists.');
    if (!['SUCCESS', 'FAILED'].includes(job.status)) continue;
    const detail = JSON.parse(job.result_json || '{}');
    return { status: job.status, result: detail.latest?.data?.result,
      error: detail.error || (job.status === 'FAILED' ? 'Your devices reported a failure while running this program.' : undefined) };
  }
  throw new Error('Stopped waiting after an hour. The run may still finish on your devices.');
}
const ide = createCodeWorkspace({ run: runCode, notice });
function showSection(name) {
  for (const [tab, panel] of [['tab-chat', 'chat-panel'], ['tab-code', 'code-panel']]) {
    const active = tab === `tab-${name}`;
    $(tab).setAttribute('aria-selected', String(active));
    $(panel).hidden = !active;
  }
  localStorage.setItem('ramanujan-section', name);
}
$('tab-chat').addEventListener('click', () => showSection('chat'));
$('tab-code').addEventListener('click', () => showSection('code'));
showSection(localStorage.getItem('ramanujan-section') === 'code' ? 'code' : 'chat');
async function loadDownloads() {
  const labels = { macos: 'macOS', windows: 'Windows', linux: 'Linux', android: 'Android' };
  const releases = await api('/api/downloads');
  $('downloads').replaceChildren();
  for (const release of releases) {
    const card = document.createElement('div'); card.className = 'download';
    const title = document.createElement('h3'); title.textContent = labels[release.platform];
    const desc = document.createElement('p'); desc.textContent = release.platform === 'android' ? 'Join from your phone or tablet.' : 'Share this computer\'s idle compute.';
    card.append(title, desc);
    if (release.available) {
      const link = document.createElement('a'); link.className = 'download-link'; link.href = release.url; link.textContent = 'Download installer';
      const checksum = document.createElement('small'); checksum.textContent = `SHA-256: ${release.sha256}`; card.append(link, checksum);
    } else { const unavailable = document.createElement('span'); unavailable.className = 'unavailable'; unavailable.textContent = 'Coming soon'; card.append(unavailable); }
    $('downloads').append(card);
  }
}
$('refresh-downloads').addEventListener('click', safe(loadDownloads));
loadDownloads().catch(notice);
async function initialize() {
  if (!managementKey) {
    const session = await api('/api/session', {});
    managementKey = session.managementKey;
    localStorage.setItem('ramanujan-management-key', managementKey);
  }
  await loadWorkspace();
}
initialize().catch(notice);
setInterval(() => { if (selectedRoom && !document.hidden) devices().catch(notice); }, 15000);
setInterval(() => { if (!document.hidden) loadDownloads().catch(notice); }, 60000);
