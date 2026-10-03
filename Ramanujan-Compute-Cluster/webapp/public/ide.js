'use strict';
/* exported createCodeWorkspace */
// Code execution workspace: a small VS Code-style editor for multi-file Python projects.
// Files live in this browser (localStorage, per room) until they are run on the room's devices.
function createCodeWorkspace({ run, notice }) {
  const $ = id => document.getElementById(id);
  const STARTER = {
    'main.py': 'from helpers import square\n\n# Sum the squares of 1..n on your devices.\ndef sum_of_squares(n):\n    total = 0\n    i = 1\n    while i <= n:\n        sq = square(i)\n        total = total + sq\n        i = i + 1\n    return total\n\nresult = sum_of_squares(10)\n',
    'helpers.py': 'def square(x):\n    y = x * x\n    return y\n'
  };
  const MAIN_CANDIDATES = ['main.py', 'app.py', 'run.py', '__main__.py'];
  const PATH = /^(?!.*(?:^|\/)\.\.?(?:\/|$))[A-Za-z0-9_][A-Za-z0-9_.-]*(?:\/[A-Za-z0-9_][A-Za-z0-9_.-]*)*\.py$/;
  const MAX_FILES = 100;
  const MAX_BYTES = 2 * 1024 * 1024;
  const KEYWORDS = new Set(('False None True and as assert break class continue def del elif else except finally for from ' +
    'global if import in is lambda nonlocal not or pass raise return try while with yield').split(' '));
  const BUILTINS = new Set('abs bool float int len max min pow print range round sum'.split(' '));
  const editor = $('code-editor');
  const highlight = $('code-highlight');
  const gutter = $('code-gutter');
  let roomId = null;
  let state = fresh();
  let saveTimer = null;
  const runs = new Map();

  const sortedPaths = () => Object.keys(state.files).sort((a, b) => a.localeCompare(b));
  const bytes = files => Object.values(files).reduce((total, source) => total + new Blob([source]).size, 0);
  const escape = text => text.replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));

  function highlightPython(source) {
    const token = /(#[^\n]*)|("""[\s\S]*?(?:"""|$)|'''[\s\S]*?(?:'''|$)|"(?:\\.|[^"\\\n])*"?|'(?:\\.|[^'\\\n])*'?)|(\b\d+(?:\.\d*)?(?:[eE][+-]?\d+)?\b)|([A-Za-z_]\w*)/g;
    let html = '';
    let last = 0;
    let previous = '';
    for (let match; (match = token.exec(source));) {
      html += escape(source.slice(last, match.index));
      last = token.lastIndex;
      const word = match[4];
      const kind = match[1] ? 'comment' : match[2] ? 'string' : match[3] ? 'number' : KEYWORDS.has(word) ? 'keyword'
        : previous === 'def' || previous === 'class' ? 'function' : BUILTINS.has(word) ? 'builtin' : '';
      previous = word || '';
      html += kind ? `<span class="tok-${kind}">${escape(match[0])}</span>` : escape(match[0]);
    }
    return html + escape(source.slice(last)) + '\n';
  }

  function defaultEntry(files) {
    const paths = Object.keys(files);
    return MAIN_CANDIDATES.find(path => paths.includes(path)) || paths.sort()[0] || null;
  }
  function fresh() {
    return { files: { ...STARTER }, entryPoint: 'main.py', open: ['main.py', 'helpers.py'], active: 'main.py' };
  }
  function normalize(saved) {
    if (!saved || typeof saved.files !== 'object' || Array.isArray(saved.files)) return fresh();
    const files = {};
    for (const [path, source] of Object.entries(saved.files)) if (PATH.test(path) && typeof source === 'string') files[path] = source;
    const open = (Array.isArray(saved.open) ? saved.open : []).filter(path => path in files);
    const active = saved.active in files ? saved.active : open[0] || null;
    return { files, entryPoint: saved.entryPoint in files ? saved.entryPoint : defaultEntry(files), open, active };
  }
  function storageKey() { return `ramanujan-code:${roomId}`; }
  function save() {
    clearTimeout(saveTimer);
    if (!roomId) return;
    try { localStorage.setItem(storageKey(), JSON.stringify(state)); }
    catch { notice(new Error('This browser has no room left to keep these files. Remove some files and try again.')); }
  }
  function scheduleSave() { clearTimeout(saveTimer); saveTimer = setTimeout(save, 400); }

  function setRoom(id) {
    save();
    roomId = id;
    let saved = null;
    try { saved = JSON.parse(localStorage.getItem(storageKey())); } catch { saved = null; }
    state = saved ? normalize(saved) : fresh();
    render();
  }

  function openFile(path) {
    if (!(path in state.files)) return;
    if (!state.open.includes(path)) state.open.push(path);
    state.active = path;
    scheduleSave();
    render();
    editor.focus();
  }
  function closeFile(path) {
    const index = state.open.indexOf(path);
    if (index < 0) return;
    state.open.splice(index, 1);
    if (state.active === path) state.active = state.open[Math.min(index, state.open.length - 1)] || null;
    scheduleSave();
    render();
  }
  function setMain(path) {
    state.entryPoint = path;
    scheduleSave();
    render();
  }
  function askPath(message, initial) {
    const value = prompt(message, initial);
    if (value === null) return null;
    const path = value.trim().replace(/^\.\//, '');
    if (!PATH.test(path)) throw new Error('Use a relative Python file path such as utils.py or pkg/module.py.');
    return path;
  }
  function newFile() {
    const path = askPath('New file path', state.active?.includes('/') ? state.active.replace(/[^/]+$/, 'new_file.py') : 'new_file.py');
    if (!path) return;
    if (path in state.files) throw new Error(`${path} already exists.`);
    if (Object.keys(state.files).length >= MAX_FILES) throw new Error(`A project can have at most ${MAX_FILES} files.`);
    state.files[path] = '';
    if (!state.entryPoint) state.entryPoint = path;
    openFile(path);
  }
  function renameFile(path) {
    const next = askPath('Rename file', path);
    if (!next || next === path) return;
    if (next in state.files) throw new Error(`${next} already exists.`);
    state.files[next] = state.files[path];
    delete state.files[path];
    state.open = state.open.map(item => item === path ? next : item);
    if (state.active === path) state.active = next;
    if (state.entryPoint === path) state.entryPoint = next;
    scheduleSave();
    render();
  }
  function deleteFile(path) {
    if (!confirm(`Delete ${path}?`)) return;
    delete state.files[path];
    closeFile(path);
    if (state.entryPoint === path) state.entryPoint = defaultEntry(state.files);
    scheduleSave();
    render();
  }

  async function addUploads(fileList, fromFolder) {
    const incoming = {};
    let skipped = 0;
    for (const file of fileList) {
      let path = fromFolder && file.webkitRelativePath ? file.webkitRelativePath.split('/').slice(1).join('/') : file.name;
      path = path.replace(/\\/g, '/');
      if (!PATH.test(path) || path.split('/').includes('__pycache__')) { skipped++; continue; }
      incoming[path] = await file.text();
    }
    const count = Object.keys(incoming).length;
    if (!count) throw new Error('No Python (.py) files were found in that selection.');
    const replace = fromFolder && Object.keys(state.files).length > 0 &&
      confirm('Replace the current files with this folder? Choose Cancel to add the folder to the current files.');
    const files = replace ? incoming : { ...state.files, ...incoming };
    if (Object.keys(files).length > MAX_FILES) throw new Error(`A project can have at most ${MAX_FILES} files.`);
    if (bytes(files) > MAX_BYTES) throw new Error('A project can be at most 2 MB of source code.');
    state.files = files;
    state.open = state.open.filter(path => path in files);
    const first = defaultEntry(incoming);
    if (!(state.entryPoint in files)) state.entryPoint = defaultEntry(files);
    if (!state.open.includes(first)) state.open.push(first);
    state.active = first;
    save();
    render();
    if (skipped) notice(new Error(`Added ${count} Python file${count === 1 ? '' : 's'}; skipped ${skipped} other file${skipped === 1 ? '' : 's'}.`));
  }

  function action(label, title, handler) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'icon-button';
    button.textContent = label;
    button.title = title;
    button.setAttribute('aria-label', title);
    button.addEventListener('click', event => { event.stopPropagation(); guard(handler)(); });
    return button;
  }
  function guard(handler) {
    return async () => { try { await handler(); } catch (error) { notice(error); } };
  }
  function renderTree() {
    const tree = $('file-tree');
    tree.replaceChildren();
    const shownFolders = new Set();
    const paths = sortedPaths();
    if (!paths.length) {
      const empty = document.createElement('li');
      empty.className = 'tree-empty';
      empty.textContent = 'No files yet. Upload files or a folder, or create a new file.';
      tree.append(empty);
    }
    for (const path of paths) {
      const parts = path.split('/');
      for (let depth = 0; depth < parts.length - 1; depth++) {
        const folder = parts.slice(0, depth + 1).join('/');
        if (shownFolders.has(folder)) continue;
        shownFolders.add(folder);
        const row = document.createElement('li');
        row.className = 'tree-row folder';
        row.style.paddingLeft = `${10 + depth * 14}px`;
        row.textContent = `\u25BE ${parts[depth]}`;
        tree.append(row);
      }
      const row = document.createElement('li');
      row.className = 'tree-row file';
      row.classList.toggle('active', path === state.active);
      row.style.paddingLeft = `${10 + (parts.length - 1) * 14}px`;
      row.title = path;
      row.tabIndex = 0;
      const name = document.createElement('span');
      name.className = 'file-name';
      name.textContent = parts[parts.length - 1];
      row.append(name);
      if (path === state.entryPoint) {
        const badge = document.createElement('span');
        badge.className = 'main-badge';
        badge.textContent = 'main';
        row.append(badge);
      }
      const actions = document.createElement('span');
      actions.className = 'row-actions';
      if (path !== state.entryPoint) actions.append(action('\u2605', `Set ${path} as main`, () => setMain(path)));
      actions.append(action('\u270E', `Rename ${path}`, () => renameFile(path)), action('\u2715', `Delete ${path}`, () => deleteFile(path)));
      row.append(actions);
      row.addEventListener('click', () => openFile(path));
      row.addEventListener('keydown', event => { if (event.key === 'Enter') openFile(path); });
      tree.append(row);
    }
  }
  function renderTabs() {
    const tabs = $('editor-tabs');
    tabs.replaceChildren();
    for (const path of state.open) {
      const tab = document.createElement('div');
      tab.className = 'editor-tab';
      tab.classList.toggle('active', path === state.active);
      tab.title = path;
      const name = document.createElement('span');
      name.textContent = path.split('/').pop();
      tab.append(name);
      if (path === state.entryPoint) {
        const badge = document.createElement('span');
        badge.className = 'main-dot';
        badge.title = 'Main file';
        badge.textContent = '\u25CF';
        tab.append(badge);
      }
      tab.append(action('\u2715', `Close ${path}`, () => closeFile(path)));
      tab.addEventListener('click', () => openFile(path));
      tabs.append(tab);
    }
  }
  function renderEntrySelect() {
    const select = $('entry-select');
    select.replaceChildren();
    for (const path of sortedPaths()) {
      const option = document.createElement('option');
      option.value = path;
      option.textContent = path;
      select.append(option);
    }
    select.value = state.entryPoint || '';
    select.disabled = !state.entryPoint;
  }
  function renderEditor() {
    const active = state.active;
    $('editor-empty').hidden = Boolean(active);
    $('code-surface').hidden = !active;
    if (!active) return;
    if (editor.dataset.path !== active || editor.value !== state.files[active]) {
      editor.value = state.files[active];
      editor.dataset.path = active;
      editor.scrollTop = 0;
      editor.scrollLeft = 0;
    }
    paint();
  }
  function paint() {
    const source = editor.value;
    highlight.innerHTML = highlightPython(source);
    const lines = source.split('\n').length;
    gutter.textContent = Array.from({ length: lines }, (_, index) => index + 1).join('\n') + '\n';
    syncScroll();
    renderStatus();
  }
  function syncScroll() {
    highlight.scrollTop = editor.scrollTop;
    highlight.scrollLeft = editor.scrollLeft;
    gutter.scrollTop = editor.scrollTop;
  }
  function renderStatus() {
    const before = editor.value.slice(0, editor.selectionStart);
    const line = before.split('\n').length;
    const column = before.length - before.lastIndexOf('\n');
    $('status-position').textContent = state.active ? `Ln ${line}, Col ${column}` : '';
    $('status-files').textContent = `${Object.keys(state.files).length} file${Object.keys(state.files).length === 1 ? '' : 's'}`;
    $('status-main').textContent = state.entryPoint ? `Main: ${state.entryPoint}` : 'No main file';
  }
  function render() {
    renderTree();
    renderTabs();
    renderEntrySelect();
    renderEditor();
    renderOutput();
  }

  function formatValue(value) {
    const number = typeof value === 'number' ? value : typeof value === 'string' && value.trim() !== '' ? Number(value) : NaN;
    if (Number.isFinite(number)) return Number.isInteger(number) ? String(number) : String(Number(number.toPrecision(10)));
    return value === null || value === undefined ? 'None' : String(value);
  }
  function resultTables(result) {
    const fragment = document.createDocumentFragment();
    const seen = new Set();
    const variables = (result?.variables || [])
      .filter(item => item.variableName && !/^(arg_|return_)/.test(item.variableId || '') && !String(item.variableId).includes('func'))
      .map(item => [item.variableName, formatValue(item.object)])
      .filter(([name, value]) => !seen.has(`${name}=${value}`) && seen.add(`${name}=${value}`))
      .sort((a, b) => a[0].localeCompare(b[0]));
    const arrays = new Map();
    for (const cell of result?.arrays || []) {
      const name = String(cell.arrayId || '').split('_name_').pop() || 'array';
      if (!arrays.has(name)) arrays.set(name, []);
      arrays.get(name).push([String(cell.indexStr), formatValue(cell.object)]);
    }
    if (variables.length) {
      const heading = document.createElement('div');
      heading.className = 'output-heading';
      heading.textContent = 'Variables';
      const table = document.createElement('table');
      for (const [name, value] of variables) {
        const row = table.insertRow();
        row.insertCell().textContent = name;
        row.insertCell().textContent = value;
      }
      fragment.append(heading, table);
    }
    if (arrays.size) {
      const heading = document.createElement('div');
      heading.className = 'output-heading';
      heading.textContent = 'Arrays';
      const table = document.createElement('table');
      const numeric = text => text.split(/\D+/).filter(Boolean).map(Number);
      const compare = (a, b) => { const x = numeric(a[0]); const y = numeric(b[0]); for (let i = 0; i < Math.max(x.length, y.length); i++) if ((x[i] || 0) !== (y[i] || 0)) return (x[i] || 0) - (y[i] || 0); return 0; };
      for (const [name, cells] of [...arrays].sort((a, b) => a[0].localeCompare(b[0]))) {
        cells.sort(compare);
        const simple = cells.every(([index], position) => index === String(position));
        const shown = cells.slice(0, 200).map(([index, value]) => simple ? value : `${index}: ${value}`);
        const row = table.insertRow();
        row.insertCell().textContent = `${name} [${cells.length}]`;
        row.insertCell().textContent = `[${shown.join(', ')}${cells.length > 200 ? `, \u2026 ${cells.length - 200} more` : ''}]`;
      }
      fragment.append(heading, table);
    }
    if (!variables.length && !arrays.size) {
      const empty = document.createElement('div');
      empty.className = 'output-muted';
      empty.textContent = 'The program finished without top-level variables to show.';
      fragment.append(empty);
    }
    return fragment;
  }
  function renderOutput() {
    const output = $('run-output');
    const status = $('run-status');
    const entry = runs.get(roomId);
    const busy = entry?.state === 'running';
    $('run-code').disabled = busy || !state.entryPoint;
    $('run-code').textContent = busy ? 'Running\u2026' : '\u25B6 Run on cluster';
    status.className = `run-status ${entry?.state || ''}`;
    if (!entry) {
      status.textContent = '';
      output.replaceChildren();
      const hint = document.createElement('div');
      hint.className = 'output-muted';
      hint.textContent = 'Choose the main file and press Run. Results from your devices appear here. (Ctrl/Cmd + Enter)';
      output.append(hint);
      return;
    }
    const seconds = Math.round(((entry.finished || Date.now()) - entry.started) / 1000);
    const labels = { running: `Running on your devices\u2026 ${seconds}s`, success: `Finished in ${seconds}s`, failed: `Failed after ${seconds}s` };
    status.textContent = labels[entry.state];
    output.replaceChildren();
    const meta = document.createElement('div');
    meta.className = 'output-muted';
    meta.textContent = `Main file: ${entry.entryPoint} \u00B7 ${entry.fileCount} file${entry.fileCount === 1 ? '' : 's'}`;
    output.append(meta);
    if (entry.state === 'failed') {
      const error = document.createElement('pre');
      error.className = 'output-error';
      error.textContent = entry.error || 'The program could not be run.';
      output.append(error);
    } else if (entry.state === 'success') {
      output.append(resultTables(entry.result));
    }
  }

  async function runProject() {
    if (!roomId) throw new Error('Select a room first.');
    if (runs.get(roomId)?.state === 'running') return;
    if (!state.entryPoint || !(state.entryPoint in state.files)) throw new Error('Mark one file as main before running.');
    save();
    const runRoom = roomId;
    const entry = { state: 'running', started: Date.now(), entryPoint: state.entryPoint, fileCount: Object.keys(state.files).length };
    runs.set(runRoom, entry);
    renderOutput();
    const ticker = setInterval(() => { if (roomId === runRoom) renderOutput(); }, 1000);
    try {
      const job = await run({ files: { ...state.files }, entryPoint: state.entryPoint });
      entry.state = job.status === 'SUCCESS' ? 'success' : 'failed';
      entry.result = job.result;
      entry.error = job.error;
    } catch (error) {
      entry.state = 'failed';
      entry.error = error.message;
    } finally {
      clearInterval(ticker);
      entry.finished = Date.now();
      if (roomId === runRoom) renderOutput();
    }
  }

  function insert(text) {
    editor.focus();
    if (!document.execCommand('insertText', false, text)) editor.setRangeText(text, editor.selectionStart, editor.selectionEnd, 'end');
  }
  function indentSelection(outdent) {
    const { value, selectionStart, selectionEnd } = editor;
    const start = value.lastIndexOf('\n', selectionStart - 1) + 1;
    const end = selectionEnd > selectionStart && value[selectionEnd - 1] === '\n' ? selectionEnd - 1 : selectionEnd;
    const block = value.slice(start, end);
    const changed = block.split('\n').map(line => outdent ? line.replace(/^ {1,4}/, '') : '    ' + line).join('\n');
    editor.setSelectionRange(start, end);
    insert(changed);
    editor.setSelectionRange(start, start + changed.length);
  }
  editor.addEventListener('keydown', event => {
    const modifier = event.metaKey || event.ctrlKey;
    if (modifier && event.key === 'Enter') { event.preventDefault(); guard(runProject)(); return; }
    if (modifier && event.key.toLowerCase() === 's') { event.preventDefault(); save(); return; }
    if (event.key === 'Tab') {
      event.preventDefault();
      if (event.shiftKey || editor.value.slice(editor.selectionStart, editor.selectionEnd).includes('\n')) indentSelection(event.shiftKey);
      else insert('    ');
      return;
    }
    if (event.key === 'Enter' && !modifier && !event.altKey) {
      const before = editor.value.slice(0, editor.selectionStart);
      const line = before.slice(before.lastIndexOf('\n') + 1);
      const indent = line.match(/^ */)[0] + (/:\s*(#.*)?$/.test(line) ? '    ' : '');
      event.preventDefault();
      insert('\n' + indent);
    }
  });
  editor.addEventListener('input', () => {
    if (!state.active) return;
    state.files[state.active] = editor.value;
    paint();
    scheduleSave();
  });
  editor.addEventListener('scroll', syncScroll);
  for (const name of ['keyup', 'click', 'select']) editor.addEventListener(name, renderStatus);
  $('new-file').addEventListener('click', guard(newFile));
  $('upload-files').addEventListener('click', () => $('upload-files-input').click());
  $('upload-folder').addEventListener('click', () => $('upload-folder-input').click());
  for (const [id, fromFolder] of [['upload-files-input', false], ['upload-folder-input', true]]) {
    $(id).addEventListener('change', guard(async () => {
      const files = [...$(id).files];
      $(id).value = '';
      if (files.length) await addUploads(files, fromFolder);
    }));
  }
  $('entry-select').addEventListener('change', () => setMain($('entry-select').value));
  $('run-code').addEventListener('click', guard(runProject));
  window.addEventListener('beforeunload', save);

  return { setRoom, highlightPython };
}
