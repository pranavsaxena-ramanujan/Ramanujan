'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const crypto = require('node:crypto');
const { createApp } = require('../app');
const { createStore } = require('../store');

async function fixture(t, options = {}) {
  const store = createStore({ LOCAL_DB: ':memory:' });
  const calls = [];
  const fetchImpl = async (url, init) => {
    calls.push({ url: new URL(url), init });
    return new Response(JSON.stringify({ status: 'SUCCESS', data: null }), { headers: { 'content-type': 'application/json' } });
  };
  const app = createApp({ store, publicUrl: 'http://localhost:8090', homelabUrl: 'http://localhost:8888',
    releasesDir: '/nonexistent/releases', fetchImpl, ...options });
  const server = await new Promise(resolve => { const s = app.listen(0, '127.0.0.1', () => resolve(s)); });
  const base = `http://127.0.0.1:${server.address().port}`;
  t.after(async () => { await new Promise(resolve => server.close(resolve)); await store.pool.end(); });
  async function request(route, body, authorization, method) {
    const response = await fetch(base + route, { method: method || (body ? 'POST' : 'GET'),
      headers: { ...(body ? { 'Content-Type': 'application/json' } : {}), ...(authorization ? { Authorization: `Bearer ${authorization}` } : {}) },
      body: body ? JSON.stringify(body) : undefined });
    return { status: response.status, data: await response.json(), headers: response.headers };
  }
  async function register() {
    const result = await request('/api/session', {});
    assert.equal(result.status, 200);
    return result.data.apiToken;
  }
  async function room(owner) {
    const result = await request('/api/clusters', { name: 'Test cluster' }, owner);
    assert.equal(result.status, 201);
    return result.data;
  }
  return { store, calls, base, request, register, room };
}

test('anonymous private rooms retain a hashed management key without login or signup', async t => {
  const f = await fixture(t);
  assert.equal((await f.request('/api/clusters')).status, 401);
  const owner = await f.register();
  const room = await f.room(owner);
  assert.equal(room.clusterId.length, 36);
  const listed = await f.request('/api/clusters', null, owner);
  assert.equal(listed.data[0].room_id, room.roomId);
  assert.equal(listed.data[0].join_secret_hash, undefined);
  assert.equal(listed.data[0].joinSecret, undefined);
  const stored = await f.store.cluster(room.roomId);
  assert.notEqual(stored.join_secret_hash, room.joinSecret);
  const second = await f.register();
  assert.equal((await f.request(`/api/clusters/${room.roomId}/devices`, null, second)).status, 404);
  assert.equal((await f.request('/api/login', {})).status, 404);
  assert.equal((await f.request('/api/register', {})).status, 404);
  await f.request('/api/session', null, owner, 'DELETE');
  assert.equal((await f.request('/api/clusters', null, owner)).status, 401);
});

test('join, scoped polling and completion override forged host and cluster identity', async t => {
  const f = await fixture(t);
  const owner = await f.register();
  const room = await f.room(owner);
  const join = { roomId: room.roomId, joinSecret: room.joinSecret, name: 'Mac', platform: 'macos' };
  assert.equal((await f.request('/api/devices/join', { ...join, joinSecret: 'wrong' })).status, 403);
  const joined = await f.request('/api/devices/join', join);
  assert.equal(joined.status, 201);
  const worker = new URL(joined.data.workerUrl).pathname;
  const poll = await f.request(`${worker}/pings/open?uuid=forged&clusterId=other`, {});
  assert.equal(poll.status, 200);
  assert.equal(f.calls[0].url.searchParams.get('uuid'), joined.data.deviceId);
  assert.equal(f.calls[0].url.searchParams.get('clusterId'), room.clusterId);
  const complete = await f.request(`${worker}/task/complete`, { uuid: 'task-id', hostId: 'forged', clusterId: 'other', data: {} });
  assert.equal(complete.status, 200);
  const payload = JSON.parse(f.calls[1].init.body);
  assert.equal(payload.hostId, joined.data.deviceId);
  assert.equal(payload.clusterId, room.clusterId);
  await f.request(`${worker}/binary/stat?uuid=task-id&hostId=forged`, null);
  assert.equal(f.calls[2].url.searchParams.get('uuid'), joined.data.deviceId);
  assert.equal(f.calls[2].url.searchParams.get('hostId'), joined.data.deviceId);
  assert.equal((await f.request(`${worker}/llm/chain`, {})).status, 404);
  await f.request(`/api/clusters/${room.roomId}/devices/${joined.data.deviceId}`, null, owner, 'DELETE');
  assert.equal((await f.request(`${worker}/pings/open`, {})).status, 401);
});

test('owner LLM gateway tags requests and blocks local filesystem escape', async t => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'rj-model-test-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const file = path.join(root, 'weights.bin');
  await fs.writeFile(file, 'weights');
  const f = await fixture(t, { modelRoot: root });
  const owner = await f.register();
  const room = await f.room(owner);
  const route = `/api/clusters/${room.roomId}/homelab/llm/step`;
  const valid = await f.request(route, { clusterId: 'forged', affinity: 's1', graph: { embed: { file } }, files: [file] }, owner);
  assert.equal(valid.status, 200);
  assert.equal(JSON.parse(f.calls[0].init.body).clusterId, room.clusterId);
  const invalid = await f.request(route, { files: [__filename] }, owner);
  assert.equal(invalid.status, 403);
  assert.equal(f.calls.length, 1);
  assert.equal((await f.request(`/api/clusters/${room.roomId}/homelab/binary/fetch?path=${file}`, null, owner)).status, 404);
  assert.equal((await f.request(`/api/clusters/${room.roomId}/homelab/orchestrator/dump`, {
    name: 'output', path: '/arbitrary/server/file'
  }, owner)).status, 404);
});

test('cookie writes reject CSRF and production requires HTTPS', async t => {
  const f = await fixture(t);
  const owner = await f.register();
  const response = await fetch(f.base + '/api/clusters', { method: 'POST',
    headers: { Cookie: `cluster_session=${owner}`, 'Content-Type': 'application/json', Origin: 'https://evil.test' },
    body: JSON.stringify({ name: 'Attack' }) });
  assert.equal(response.status, 403);
  assert.throws(() => createApp({ store: f.store, publicUrl: 'http://insecure.test', homelabUrl: 'http://localhost:8888', production: true }), /HTTPS/);
});

test('chat requires online joined devices and uses selected cluster, not caller clusterId', async t => {
  const requests = [];
  const f = await fixture(t, { models: [{ id: 'tiny', name: 'Tiny' }], infer: async request => { requests.push(request); return 'Paris'; } });
  const owner = await f.register();
  const room = await f.room(owner);
  const route = `/api/clusters/${room.roomId}/chat`;
  assert.equal((await f.request(route, { model: 'tiny', question: 'Capital of France?' }, owner)).status, 409);
  const joined = await f.request('/api/devices/join', { roomId: room.roomId, joinSecret: room.joinSecret, name: 'Mac', platform: 'macos' });
  await f.store.touchDevice(joined.data.deviceId);
  const answer = await f.request(route, { model: 'tiny', question: 'Capital of France?', clusterId: 'forged' }, owner);
  assert.equal(answer.status, 200);
  assert.equal(answer.data.answer, 'Paris');
  assert.equal(requests[0].roomId, room.roomId);
  assert.equal(requests[0].apiToken, owner);
  const jobs = await f.request(`/api/clusters/${room.roomId}/jobs`, null, owner);
  assert.equal(jobs.data[0].status, 'SUCCESS');
});

test('download availability requires a real checksum-verified native artifact', async t => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'rj-release-test-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const bytes = Buffer.from('test installer');
  await fs.writeFile(path.join(root, 'Mac.pkg'), bytes);
  await fs.writeFile(path.join(root, 'manifest.json'), JSON.stringify({ macos: {
    file: 'Mac.pkg', sha256: crypto.createHash('sha256').update(bytes).digest('hex')
  } }));
  const f = await fixture(t, { releasesDir: root });
  const result = await f.request('/api/downloads');
  assert.equal(result.data.find(item => item.platform === 'macos').available, true);
  assert.equal(result.data.find(item => item.platform === 'windows').available, false);
  await fs.writeFile(path.join(root, 'Mac.pkg'), 'tampered');
  assert.equal((await f.request('/api/downloads')).status, 500);
});

test('general DAGs use existing middleware submission and cluster-scoped status', async t => {
  const calls = [];
  const f = await fixture(t, {
    middlewareUrl: 'http://middleware.test:8080',
    orchestratorUrl: 'http://orchestrator.test:8081',
    fetchImpl: async (url, init) => {
      const target = new URL(url);
      calls.push({ target, init });
      const data = target.pathname === '/run' ? { asyncId: 'existing-middleware-id' } : { taskStatus: 'SUCCESS', result: 4 };
      return new Response(JSON.stringify({ status: '200 OK', data }), { headers: { 'content-type': 'application/json' } });
    }
  });
  const owner = await f.register();
  const room = await f.room(owner);
  const route = `/api/clusters/${room.roomId}/jobs`;
  const submitted = await f.request(route, { code: 'a = 2 + 2', clusterId: 'forged' }, owner);
  assert.equal(submitted.status, 202);
  assert.equal(submitted.data.status, 'QUEUED');
  assert.equal(calls[0].target.origin, 'http://middleware.test:8080');
  assert.equal(calls[0].target.pathname, '/run');
  assert.equal(JSON.parse(calls[0].init.body).clusterId, room.clusterId);
  const jobs = await f.request(route, null, owner);
  assert.equal(jobs.data[0].status, 'SUCCESS');
  assert.equal(calls[1].target.pathname, '/status');
  assert.equal(calls[1].target.searchParams.get('uuid'), 'existing-middleware-id');
  assert.equal(calls[1].target.searchParams.get('clusterId'), room.clusterId);
});

test('multi-file projects are forwarded with the chosen main file and validated', async t => {
  const calls = [];
  const f = await fixture(t, {
    middlewareUrl: 'http://middleware.test:8080',
    fetchImpl: async (url, init) => {
      calls.push({ target: new URL(url), init });
      return new Response(JSON.stringify({ status: '200 OK', data: { asyncId: 'multi-id' } }), { headers: { 'content-type': 'application/json' } });
    }
  });
  const owner = await f.register();
  const room = await f.room(owner);
  const route = `/api/clusters/${room.roomId}/jobs`;
  const files = { 'app.py': 'from pkg.ops import add\nx = add(1, 2)\n', 'pkg/ops.py': 'def add(a, b):\n    return a + b\n' };
  const submitted = await f.request(route, { files, entryPoint: 'app.py' }, owner);
  assert.equal(submitted.status, 202);
  const body = JSON.parse(calls[0].init.body);
  assert.deepEqual(body.files, files);
  assert.equal(body.entryPoint, 'app.py');
  assert.equal(body.code, undefined);
  assert.equal(body.clusterId, room.clusterId);
  for (const bad of [
    { files, entryPoint: 'missing.py' },
    { files: { '../escape.py': 'x = 1' }, entryPoint: '../escape.py' },
    { files: { '/abs.py': 'x = 1' }, entryPoint: '/abs.py' },
    { files: { 'notes.txt': 'x' }, entryPoint: 'notes.txt' },
    { files: { 'main.py': 'f = open(\'x\')' }, entryPoint: 'main.py' },
    { files: {}, entryPoint: 'main.py' }
  ]) assert.equal((await f.request(route, bad, owner)).status, 400, JSON.stringify(bad));
  assert.equal(calls.length, 1);
});

test('rejected programs return a readable reason and stale jobs do not break the job list', async t => {
  let mode = 'reject';
  const f = await fixture(t, {
    middlewareUrl: 'http://middleware.test:8080',
    fetchImpl: async url => {
      const target = new URL(url);
      if (target.pathname === '/run' && mode === 'reject') {
        return new Response(JSON.stringify({ status: '500 Internal Server Error', data: { stackTrace: [{ className: 'x' }],
          message: 'Compilation error at line null character null: Error parsing Python code: Compilation error at line null character null: Function argument must be a variable name' } }),
          { status: 500, headers: { 'content-type': 'application/json' } });
      }
      if (target.pathname === '/run') return new Response(JSON.stringify({ status: '200 OK', data: { asyncId: 'lost-id' } }), { headers: { 'content-type': 'application/json' } });
      return new Response(JSON.stringify({ status: '500', data: null }), { status: 500, headers: { 'content-type': 'application/json' } });
    }
  });
  const owner = await f.register();
  const room = await f.room(owner);
  const route = `/api/clusters/${room.roomId}/jobs`;
  const rejected = await f.request(route, { files: { 'main.py': 'x = f(1 + 2)\n' }, entryPoint: 'main.py' }, owner);
  assert.equal(rejected.status, 422);
  assert.equal(rejected.data.error, 'Function argument must be a variable name');
  assert.equal(JSON.stringify(rejected.data).includes('stackTrace'), false);
  mode = 'accept';
  assert.equal((await f.request(route, { code: 'x = 1' }, owner)).status, 202);
  const jobs = await f.request(route, null, owner);
  assert.equal(jobs.status, 200);
  assert.deepEqual(jobs.data.map(job => job.status).sort(), ['FAILED', 'FAILED']);
  assert.match(jobs.data.map(job => JSON.parse(job.result_json).error).join(' '), /no longer tracks this job/);
});

test('binary relay preserves Content-Length for worker integrity checks', async t => {
  const bytes = Buffer.from('weights-bytes');
  const f = await fixture(t, { fetchImpl: async () => new Response(bytes, { headers: {
    'content-type': 'application/octet-stream', 'content-length': String(bytes.length), 'x-ramanujan-mtime': '123' } }) });
  const owner = await f.register();
  const room = await f.room(owner);
  const joined = await f.request('/api/devices/join', { roomId: room.roomId, joinSecret: room.joinSecret, name: 'Mac', platform: 'macos' });
  const response = await fetch(joined.data.workerUrl.replace('http://localhost:8090', f.base) + '/binary/fetch?path=/m/w.bin');
  assert.equal(response.status, 200);
  assert.equal(response.headers.get('content-length'), String(bytes.length));
  assert.equal(response.headers.get('x-ramanujan-mtime'), '123');
  assert.deepEqual(Buffer.from(await response.arrayBuffer()), bytes);
});

test('unreachable execution service returns an actionable availability error', async t => {
  const failure = new TypeError('fetch failed', { cause: { code: 'ECONNREFUSED' } });
  const f = await fixture(t, { fetchImpl: async () => { throw failure; } });
  const owner = await f.register();
  const room = await f.room(owner);
  const joined = await f.request('/api/devices/join', { roomId: room.roomId, joinSecret: room.joinSecret, name: 'Mac', platform: 'macos' });
  const worker = new URL(joined.data.workerUrl).pathname;
  const response = await f.request(`${worker}/pings/open`, {});
  assert.equal(response.status, 503);
  assert.match(response.data.error, /Start the local orchestrator/);
  assert.equal((await f.store.devices(room.clusterId))[0].last_seen, null);
});

test('missing native backend endpoint is not reported as a device library failure', async t => {
  const f = await fixture(t, {
    models: [{ id: 'tiny', name: 'Tiny' }],
    infer: async () => { throw Object.assign(new Error('Backend route missing'), { code: 'INFERENCE_BACKEND_NOT_READY' }); }
  });
  const owner = await f.register();
  const room = await f.room(owner);
  const joined = await f.request('/api/devices/join', { roomId: room.roomId, joinSecret: room.joinSecret, name: 'Mac', platform: 'macos' });
  await f.store.touchDevice(joined.data.deviceId);
  const response = await f.request(`/api/clusters/${room.roomId}/chat`, { model: 'tiny', question: 'Hello' }, owner);
  assert.equal(response.status, 503);
  assert.match(response.data.error, /cannot answer questions right now/);
});

test('chat sends the validated conversation history and reports dropped turns', async t => {
  const requests = [];
  const f = await fixture(t, { models: [{ id: 'tiny', name: 'Tiny' }],
    infer: async request => { requests.push(request); return { text: 'Ada', droppedTurns: 2 }; } });
  const owner = await f.register();
  const room = await f.room(owner);
  const joined = await f.request('/api/devices/join', { roomId: room.roomId, joinSecret: room.joinSecret, name: 'Mac', platform: 'macos' });
  await f.store.touchDevice(joined.data.deviceId);
  const route = `/api/clusters/${room.roomId}/chat`;
  const messages = [{ role: 'user', content: 'My name is Ada.' }, { role: 'assistant', content: 'Hi Ada!', extra: 1 }, { role: 'user', content: 'Who am I?' }];
  const answer = await f.request(route, { model: 'tiny', messages }, owner);
  assert.equal(answer.status, 200);
  assert.deepEqual(answer.data, { id: answer.data.id, answer: 'Ada', droppedTurns: 2 });
  assert.deepEqual(requests[0].messages, messages.map(({ role, content }) => ({ role, content })));
  for (const bad of [[], [{ role: 'assistant', content: 'x' }], messages.slice(0, 2), [{ role: 'user', content: 'a' }, { role: 'user', content: 'b' }, { role: 'user', content: 'c' }],
    [{ role: 'user', content: 'a' }, { role: 'assistant', content: 'b' }, { role: 'user', content: '  ' }], [{ role: 'user', content: 5 }]]) {
    assert.equal((await f.request(route, { model: 'tiny', messages: bad }, owner)).status, 400, JSON.stringify(bad));
  }
  assert.equal(requests.length, 1);
  const legacy = await f.request(route, { model: 'tiny', question: 'Hello' }, owner);
  assert.equal(legacy.status, 200);
  assert.deepEqual(requests[1].messages, [{ role: 'user', content: 'Hello' }]);
});

async function modelFixture(t, download) {
  const { createModelManager } = require('../user-models');
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'rj-user-models-'));
  t.after(() => fs.rm(dir, { recursive: true, force: true }));
  const conversions = [];
  let store;
  const lazyStore = new Proxy({}, { get: (target, name) => (...args) => store[name](...args) });
  const userModels = createModelManager({ store: lazyStore, dir, reserveBytes: 0, download,
    convert: async ({ gguf, shards, plan }) => {
      conversions.push(gguf);
      if ((await fs.readFile(gguf)).includes('broken')) throw new Error('unsupported architecture: mystery');
      await fs.mkdir(shards); await fs.writeFile(path.join(shards, 'shard-00'), 'weights');
      await fs.writeFile(path.join(shards, 'model-manifest.json'), '{}');
      await fs.mkdir(plan);
      await fs.writeFile(path.join(plan, 'gguf-metadata.json'), JSON.stringify({ 'general.architecture': 'llama',
        'llama.context_length': 512, 'tokenizer.chat_template': '<|start_header_id|>' }));
    } });
  const requests = [];
  const f = await fixture(t, { userModels, infer: async request => { requests.push(request); return 'ok'; } });
  store = f.store;
  await userModels.init();
  return { ...f, dir, userModels, conversions, requests };
}
async function upload(f, owner, body, name = 'My model') {
  const response = await fetch(`${f.base}/api/models/upload?name=${encodeURIComponent(name)}`, { method: 'POST',
    headers: { 'Content-Type': 'application/octet-stream', Authorization: `Bearer ${owner}` }, body });
  return { status: response.status, data: await response.json() };
}
async function waitForModel(f, owner, id) {
  for (let i = 0; i < 100; i++) {
    const model = (await f.request('/api/models', null, owner)).data.find(item => item.id === id);
    if (!model || ['READY', 'FAILED'].includes(model.status)) return model;
    await new Promise(resolve => setTimeout(resolve, 20));
  }
  throw new Error('model did not finish');
}

test('owners upload GGUF models that are converted, private to them and usable for chat', async t => {
  const f = await modelFixture(t);
  const owner = await f.register();
  const other = await f.register();
  const rejected = await upload(f, owner, 'not a gguf at all');
  assert.equal(rejected.status, 400);
  assert.match(rejected.data.error, /not a GGUF/);
  assert.deepEqual((await f.request('/api/models', null, owner)).data, []);
  assert.deepEqual(await fs.readdir(f.dir), []);

  const created = await upload(f, owner, 'GGUF\u0003rest-of-model');
  assert.equal(created.status, 201);
  assert.equal(created.data.status, 'QUEUED');
  const ready = await waitForModel(f, owner, created.data.id);
  assert.equal(ready.status, 'READY');
  assert.equal(ready.architecture, 'llama');
  assert.equal(ready.chatFormat, 'llama3');
  assert.equal(ready.sizeBytes, 18);
  await assert.rejects(fs.stat(path.join(f.dir, created.data.id, 'model.gguf')));
  assert.deepEqual((await f.request('/api/models', null, other)).data, []);
  assert.equal((await f.request(`/api/models/${created.data.id}`, null, other, 'DELETE')).status, 404);

  const room = await f.room(owner);
  const joined = await f.request('/api/devices/join', { roomId: room.roomId, joinSecret: room.joinSecret, name: 'Mac', platform: 'macos' });
  await f.store.touchDevice(joined.data.deviceId);
  assert.equal((await f.request(`/api/clusters/${room.roomId}/chat`, { model: created.data.id, question: 'Hi' }, owner)).status, 200);
  const model = f.requests[0].model;
  assert.equal(model.package, path.join(f.dir, created.data.id, 'shards'));
  assert.equal(model.chatFormat, 'llama3');
  assert.equal(model.maxContext, 512);
  assert.equal(model.maxNewTokens, 128);

  const shard = path.join(f.dir, created.data.id, 'shards', 'shard-00');
  const gateway = `/api/clusters/${room.roomId}/homelab/llm/step`;
  assert.equal((await f.request(gateway, { files: [shard] }, owner)).status, 200);
  const otherRoom = await f.room(other);
  assert.equal((await f.request(`/api/clusters/${otherRoom.roomId}/homelab/llm/step`, { files: [shard] }, other)).status, 403);

  assert.equal((await f.request(`/api/models/${created.data.id}`, null, owner, 'DELETE')).status, 200);
  await assert.rejects(fs.stat(path.join(f.dir, created.data.id)));
  assert.equal((await f.request(`/api/clusters/${room.roomId}/chat`, { model: created.data.id, question: 'Hi' }, owner)).status, 503);

  const broken = await upload(f, owner, 'GGUF broken');
  const failed = await waitForModel(f, owner, broken.data.id);
  assert.equal(failed.status, 'FAILED');
  assert.match(failed.detail, /unsupported architecture/);
});

test('URL imports download only public HTTPS models', async t => {
  const { Readable } = require('node:stream');
  const downloads = [];
  const f = await modelFixture(t, async url => {
    downloads.push(url);
    if (url.includes('missing')) throw new Error('The model URL returned HTTP 404');
    return { stream: Readable.from([Buffer.from('GGUF'), Buffer.from('-weights')]), size: 12 };
  });
  const owner = await f.register();
  for (const url of ['http://example.com/model.gguf', 'https://127.0.0.1/model.gguf', 'https://[::1]/model.gguf',
    'https://10.1.2.3/m.gguf', 'https://user:pw@example.com/m.gguf', 'https://localhost/m.gguf', 'file:///etc/passwd', 'not a url']) {
    assert.equal((await f.request('/api/models', { url }, owner)).status, 400, url);
  }
  assert.deepEqual(downloads, []);
  const created = await f.request('/api/models', { url: 'https://huggingface.co/org/repo/resolve/main/Tiny-Q4_K_M.gguf' }, owner);
  assert.equal(created.status, 202);
  assert.equal(created.data.name, 'Tiny-Q4_K_M');
  assert.equal((await waitForModel(f, owner, created.data.id)).status, 'READY');
  const missing = await f.request('/api/models', { url: 'https://example.com/missing.gguf', name: 'Missing' }, owner);
  const failed = await waitForModel(f, owner, missing.data.id);
  assert.equal(failed.status, 'FAILED');
  assert.match(failed.detail, /HTTP 404/);
});

test('public address checks block private, loopback and mapped networks', () => {
  const { isPublicAddress } = require('../user-models');
  for (const address of ['127.0.0.1', '10.0.0.8', '172.20.1.1', '192.168.1.1', '169.254.169.254', '100.64.0.1', '0.0.0.0',
    '::1', '::', 'fe80::1', 'fd00::1', '::ffff:127.0.0.1', '::ffff:10.0.0.1', 'not-an-ip']) {
    assert.equal(isPublicAddress(address), false, address);
  }
  for (const address of ['8.8.8.8', '104.16.1.1', '2606:4700::1111']) assert.equal(isPublicAddress(address), true, address);
});

test('deleting a converting model stops the conversion and a restart fails unfinished models', async t => {
  const { createModelManager } = require('../user-models');
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'rj-user-models-'));
  t.after(() => fs.rm(dir, { recursive: true, force: true }));
  const store = createStore({ LOCAL_DB: ':memory:' });
  t.after(() => store.pool.end());
  let started;
  const converting = new Promise(resolve => { started = resolve; });
  const manager = createModelManager({ store, dir, reserveBytes: 0, convert: ({ shards, signal }) => new Promise((resolve, reject) => {
    fs.mkdir(shards).then(() => started());
    signal.addEventListener('abort', () => reject(new Error('killed')));
  }) });
  await manager.init();
  await store.createOwner({ id: 'o1', email: 'o1@local.invalid', passwordHash: 'x' });
  const { Readable } = require('node:stream');
  const model = await manager.upload('o1', 'Slow', Readable.from([Buffer.from('GGUF-data')]), 9);
  await converting;
  assert.equal((await manager.list('o1'))[0].status, 'CONVERTING');
  await manager.remove('o1', model.id);
  assert.deepEqual(await manager.list('o1'), []);
  assert.deepEqual(await fs.readdir(dir), []);

  await store.createModel({ id: crypto.randomUUID(), ownerId: 'o1', name: 'Half', source: 'url', sourceUrl: 'https://x.test/a.gguf', status: 'DOWNLOADING' });
  await createModelManager({ store, dir, reserveBytes: 0 }).init();
  const [interrupted] = await manager.list('o1');
  assert.equal(interrupted.status, 'FAILED');
  assert.match(interrupted.detail, /restart/);
});
