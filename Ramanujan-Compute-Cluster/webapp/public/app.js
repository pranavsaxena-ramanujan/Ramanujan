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
  const label = document.createElement('div'); label.className = 'message-label'; label.textContent = role === 'user' ? 'You' : 'Ramanujan';
  const text = document.createElement('div'); text.textContent = content;
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
  renderChat(); await devices();
}
async function loadWorkspace() {
  const rooms = await api('/api/clusters');
  $('workspace').hidden = false;
  $('room-list').replaceChildren();
  for (const room of rooms) {
    const button = document.createElement('button'); button.className = 'room-button'; button.textContent = room.name; button.dataset.id = room.id;
    button.addEventListener('click', safe(() => selectRoom(room))); $('room-list').append(button);
  }
  const models = await api('/api/models');
  $('model-select').replaceChildren();
  if (!models.length) { const option = document.createElement('option'); option.textContent = 'No model configured'; option.value = ''; $('model-select').append(option); }
  for (const model of models) { const option = document.createElement('option'); option.value = model.id; option.textContent = model.name; $('model-select').append(option); }
  const room = rooms.find(item => item.id === selectedRoom?.id) || rooms[0];
  if (room) await selectRoom(room);
}
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
$('chat-form').addEventListener('submit', safe(async () => {
  if (!selectedRoom) throw new Error('Create or select a room first.');
  if (pending) return;
  const roomId = selectedRoom.room_id;
  const question = $('question').value.trim();
  const history = chats.get(roomId) || []; chats.set(roomId, history);
  history.push({ role: 'user', text: question }); $('question').value = ''; renderChat();
  pending = true; $('send').disabled = true; $('send').textContent = 'Inferring...';
  try {
    const result = await api(roomPath('chat'), { question, model: $('model-select').value });
    history.push({ role: 'assistant', text: result.answer || '(The model returned no text.)' });
  } catch (error) { history.push({ role: 'error', text: error.message }); }
  finally { pending = false; $('send').disabled = false; $('send').textContent = 'Send \u2192'; renderChat(); }
}));
$('job-form').addEventListener('submit', safe(async () => {
  if (!selectedRoom) throw new Error('Select a room first.');
  $('job-result').textContent = 'Running on your cluster...';
  try {
    const endpoint = roomPath('jobs');
    const result = await api(endpoint, { code: $('job-code').value });
    $('job-result').textContent = JSON.stringify(result, null, 2);
    for (let attempt = 0; result.status === 'QUEUED' && attempt < 300; attempt++) {
      await new Promise(resolve => setTimeout(resolve, 2000));
      const job = (await api(endpoint)).find(item => item.id === result.id);
      if (!job) throw new Error('Submitted job no longer exists.');
      $('job-result').textContent = JSON.stringify({ status: job.status, result: JSON.parse(job.result_json) }, null, 2);
      if (['SUCCESS', 'FAILED'].includes(job.status)) return;
    }
  }
  catch (error) { $('job-result').textContent = error.message; throw error; }
}));
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
