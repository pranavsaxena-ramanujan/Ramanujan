'use strict';
const express = require('express');
const crypto = require('node:crypto');
const fs = require('node:fs/promises');
const path = require('node:path');
const { Readable } = require('node:stream');
const { pipeline } = require('node:stream/promises');
const { token, digest, hashSecret, verifySecret, HttpError, text, sessionToken, requireOrigin } = require('./security');
const asyncRoute = fn => (req, res, next) => Promise.resolve(fn(req, res)).catch(next);
const workerRoutes = new Set(['/pings/open', '/pings/heartbeat', '/pings/capacity', '/task/complete', '/binary/fetch', '/binary/stat', '/orchestrator/uploadBinary']);
const ownerRoutes = new Set(['/llm/chain', '/llm/step', '/llm/close', '/llm/capacity', '/llm/plan', '/llm/plan/release']);
const platforms = new Set(['windows', 'linux', 'macos', 'android']);
const dummyHash = '00000000000000000000000000000000:' + '00'.repeat(64);

const projectPath = /^(?!.*(?:^|\/)\.\.?(?:\/|$))[A-Za-z0-9_][A-Za-z0-9_.-]*(?:\/[A-Za-z0-9_][A-Za-z0-9_.-]*)*\.py$/;

// A multi-file Python project: relative .py paths mapped to source, plus the file to run.
/** Turns a middleware rejection (which embeds a Java exception) into a short message for the program author. */
function programError(result) {
  const raw = String(result?.data?.message || result?.error || result?.message || '');
  const cleaned = raw.replace(/Compilation error at line null character null:\s*/g, '')
    .replace(/Error parsing Python code:\s*/g, '').trim();
  return cleaned ? cleaned.slice(0, 600) : 'The cluster could not run this program.';
}
function pythonProject(files, entryPoint) {
  if (!files || typeof files !== 'object' || Array.isArray(files)) throw new HttpError(400, 'files must map file paths to source code');
  const entries = Object.entries(files);
  if (!entries.length || entries.length > 100) throw new HttpError(400, 'Submit between 1 and 100 Python files');
  let total = 0;
  for (const [name, source] of entries) {
    if (name.length > 200 || !projectPath.test(name)) throw new HttpError(400, `Invalid file path: ${name.slice(0, 200)}`);
    if (typeof source !== 'string') throw new HttpError(400, `File ${name} must contain text`);
    total += source.length;
  }
  if (total > 2000000) throw new HttpError(400, 'Project is larger than 2 MB');
  if (typeof entryPoint !== 'string' || !Object.hasOwn(files, entryPoint)) throw new HttpError(400, 'Choose which file is the main file');
  if (!files[entryPoint].trim()) throw new HttpError(400, 'The main file is empty');
  return { files, entryPoint };
}

function createApp({ store, publicUrl, orchestratorUrl, middlewareUrl, homelabUrl, modelRoot, releasesDir, models = [], userModels, infer, production = false, fetchImpl = fetch }) {
  const app = express();
  const publicOrigin = new URL(publicUrl).origin;
  if (production && !publicUrl.startsWith('https://')) throw new Error('Production PUBLIC_URL must use HTTPS');
  const backend = new URL(orchestratorUrl || homelabUrl);
  const middleware = new URL(middlewareUrl || orchestratorUrl || homelabUrl);
  if (![backend, middleware].every(url => ['http:', 'https:'].includes(url.protocol))) throw new Error('Invalid execution service URL');
  app.disable('x-powered-by');
  app.use((req, res, next) => {
    res.set('X-Content-Type-Options', 'nosniff');
    res.set('Referrer-Policy', 'no-referrer');
    res.set('Content-Security-Policy', "default-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'");
    res.set('Cache-Control', 'no-store');
    try { requireOrigin(req, publicOrigin); next(); } catch (error) { next(error); }
  });
  app.use(express.json({ limit: '32mb' }));
  const attempts = new Map();
  const activeChats = new Set();
  function checkRate(req) {
    const now = Date.now();
    for (const [key, value] of attempts) if (value.until < now) attempts.delete(key);
    const key = req.socket.remoteAddress;
    const entry = attempts.get(key) || { count: 0, until: now + 60_000 };
    if (++entry.count > 20 || attempts.size >= 10_000) throw new HttpError(429, 'Too many attempts; retry in a minute');
    attempts.set(key, entry);
  }
  async function owner(req) {
    const value = sessionToken(req);
    const session = value && await store.session(digest(value));
    if (!session) throw new HttpError(401, 'Private management key required');
    return session.owner_id;
  }
  async function roomForOwner(req) {
    const ownerId = await owner(req);
    const room = await store.cluster(req.params.roomId);
    if (!room || room.owner_id !== ownerId) throw new HttpError(404, 'Room not found');
    return room;
  }
  async function createManagementKey(res, ownerId) {
    const value = token();
    const lifetime = 10 * 365 * 86400000;
    await store.createSession(digest(value), ownerId, new Date(Date.now() + lifetime));
    res.json({ managementKey: value, apiToken: value, expiresIn: lifetime / 1000 });
  }
  app.post('/api/session', asyncRoute(async (req, res) => {
    checkRate(req);
    const id = crypto.randomUUID();
    await store.createOwner({ id, email: `anonymous-${id}@local.invalid`, passwordHash: 'disabled' });
    await createManagementKey(res, id);
  }));
  app.delete('/api/session', asyncRoute(async (req, res) => {
    await owner(req);
    const value = sessionToken(req);
    if (value) await store.deleteSession(digest(value));
    res.json({ success: true });
  }));
  app.get('/api/clusters', asyncRoute(async (req, res) => res.json(await store.clusters(await owner(req)))));
  app.post('/api/clusters', asyncRoute(async (req, res) => {
    const ownerId = await owner(req);
    const secret = token();
    const room = { id: crypto.randomUUID(), roomId: crypto.randomBytes(8).toString('hex'), ownerId,
      name: text(req.body.name, 'Cluster name'), secretHash: await hashSecret(secret) };
    await store.createCluster(room);
    res.status(201).json({ clusterId: room.id, roomId: room.roomId, name: room.name, joinSecret: secret });
  }));
  app.get('/api/clusters/:roomId/devices', asyncRoute(async (req, res) => res.json(await store.devices((await roomForOwner(req)).id))));
  app.delete('/api/clusters/:roomId/devices/:deviceId', asyncRoute(async (req, res) => {
    if (!await store.revokeDevice(req.params.deviceId, (await roomForOwner(req)).id)) throw new HttpError(404, 'Device not found');
    res.json({ success: true });
  }));
  app.post('/api/devices/join', asyncRoute(async (req, res) => {
    checkRate(req);
    const room = await store.cluster(text(req.body.roomId, 'Room ID', 32));
    const secret = text(req.body.joinSecret, 'Join secret', 256);
    const name = text(req.body.name, 'Device name');
    const platform = text(req.body.platform, 'Platform', 16);
    if (!platforms.has(platform)) throw new HttpError(400, 'Unsupported platform');
    const valid = await verifySecret(secret, room ? room.join_secret_hash : dummyHash);
    if (!room || !valid) throw new HttpError(403, 'Invalid room ID or join secret');
    const value = token();
    const id = crypto.randomUUID();
    await store.createDevice({ id, clusterId: room.id, tokenHash: digest(value), name, platform });
    res.status(201).json({ deviceId: id, clusterId: room.id, workerUrl: `${publicOrigin}/worker/${value}` });
  }));
  async function validateModelFiles(body, ownerId, capacityPlan = false) {
    const candidates = new Set();
    function walk(value, key) {
      if (key === 'file' && typeof value === 'string') candidates.add(value);
      if (key === 'files') {
        if (!Array.isArray(value)) throw new HttpError(400, 'files must be an array of paths');
        for (const file of value) {
          if (typeof file === 'string' && !capacityPlan) candidates.add(file);
          else if (capacityPlan && file && typeof file.path === 'string') candidates.add(file.path);
          else throw new HttpError(400, 'Invalid model file declaration');
        }
      }
      if (value && typeof value === 'object') for (const [name, child] of Object.entries(value)) walk(child, name);
    }
    walk(body);
    if (!candidates.size) return;
    if (!modelRoot && !userModels) throw new HttpError(503, 'MODEL_ROOT must be configured before opening LLM sessions');
    const root = modelRoot ? await fs.realpath(modelRoot) : null;
    for (const candidate of candidates) {
      let real;
      try { real = await fs.realpath(candidate); }
      catch (error) {
        if (error.code === 'ENOENT') throw new HttpError(400, 'Model file not found on the server');
        throw error;
      }
      const allowed = (root && real.startsWith(root + path.sep)) || (userModels && await userModels.ownsFile(ownerId, real));
      if (!allowed || !(await fs.stat(real)).isFile()) throw new HttpError(403, 'Model files must be inside MODEL_ROOT or one of your models');
    }
  }
  async function relay(req, res, route, clusterId, deviceId, ownerId) {
    const target = new URL(route, backend);
    const deviceUuidQuery = route.startsWith('/pings/') || route.startsWith('/binary/');
    for (const [key, value] of new URL(req.originalUrl, publicOrigin).searchParams) {
      if (!['clusterId', 'hostId'].includes(key) && !(deviceUuidQuery && key === 'uuid')) target.searchParams.append(key, value);
    }
    target.searchParams.set('clusterId', clusterId);
    if (deviceId) {
      target.searchParams.set('hostId', deviceId);
      if (deviceUuidQuery) target.searchParams.set('uuid', deviceId);
    }
    let body, duplex;
    const headers = {};
    if (!['GET', 'HEAD'].includes(req.method)) {
      if (req.is('application/json')) {
        const payload = { ...req.body, clusterId };
        if (deviceId) payload.hostId = deviceId;
        if (!deviceId) await validateModelFiles(payload, ownerId, route === '/llm/plan');
        body = JSON.stringify(payload);
        headers['Content-Type'] = 'application/json';
      } else {
        body = req; duplex = 'half';
        headers['Content-Type'] = req.get('content-type') || 'application/octet-stream';
      }
    }
    const controller = new AbortController();
    // Above the backend's 1800 s LLM request ceiling.
    const timer = setTimeout(() => controller.abort(), 31 * 60 * 1000);
    const abort = () => controller.abort();
    res.on('close', abort);
    try {
      const response = await fetchImpl(target, { method: req.method, headers, body, duplex, signal: controller.signal, redirect: 'error' });
      if (deviceId && response.ok) await store.touchDevice(deviceId);
      res.status(response.status).set('Content-Type', response.headers.get('content-type') || 'application/octet-stream');
      // Workers verify binary transfers against Content-Length, so it must survive the relay.
      for (const name of ['content-length', 'x-ramanujan-mtime']) {
        const value = response.headers.get(name);
        if (value !== null) res.set(name, value);
      }
      if (response.body) await pipeline(Readable.fromWeb(response.body), res);
      else res.end();
    } catch (error) {
      if (['UND_ERR_HEADERS_TIMEOUT', 'UND_ERR_BODY_TIMEOUT', 'UND_ERR_CONNECT_TIMEOUT'].includes(error.cause?.code)) {
        throw new HttpError(504, 'Execution service timed out. Active device work may still be running; check its status before starting another inference.');
      }
      if (['ECONNREFUSED', 'ENOTFOUND', 'ETIMEDOUT'].includes(error.cause?.code)) {
        throw new HttpError(503, 'The computation service is not running or reachable. Start the local orchestrator before starting device workers.');
      }
      throw error;
    } finally { clearTimeout(timer); res.off('close', abort); }
  }
  app.all('/worker/:deviceToken/*', asyncRoute(async (req, res) => {
    const route = '/' + req.params[0];
    if (!workerRoutes.has(route)) throw new HttpError(404, 'Worker endpoint not found');
    if (!['GET', 'POST'].includes(req.method)) throw new HttpError(405, 'Method not allowed');
    const device = await store.device(digest(req.params.deviceToken));
    if (!device) throw new HttpError(401, 'Device token invalid or revoked');
    await relay(req, res, route, device.cluster_id, device.id);
  }));
  app.all('/api/clusters/:roomId/homelab/*', asyncRoute(async (req, res) => {
    const room = await roomForOwner(req);
    const route = '/' + req.params[0];
    if (!ownerRoutes.has(route)) throw new HttpError(404, 'Owner endpoint not found');
    if (!['GET', 'POST'].includes(req.method)) throw new HttpError(405, 'Method not allowed');
    await relay(req, res, route, room.id, undefined, room.owner_id);
  }));
  app.post('/api/clusters/:roomId/jobs', asyncRoute(async (req, res) => {
    const room = await roomForOwner(req);
    if (JSON.stringify(req.body).includes('binaryFile')) throw new HttpError(400, 'Local file inputs are not accepted by this endpoint');
    if (req.body.csvInformationList?.length) throw new HttpError(400, 'CSV inputs require an administrator-provided model package; submit code without local file inputs');
    const program = req.body.files !== undefined ? pythonProject(req.body.files, req.body.entryPoint)
      : { code: text(req.body.code, 'Code', 1000000) };
    const sources = program.files ? Object.values(program.files) : [program.code];
    if (sources.some(source => /\b(?:open|load_binary)\s*\(/.test(source))) throw new HttpError(400, 'Local file inputs are not accepted by this endpoint');
    const id = crypto.randomUUID();
    await store.createJob(id, room.id);
    try {
      const response = await fetchImpl(new URL('/run?debug=false', middleware), {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...program, csvInformationList: [], clusterId: room.id }),
        signal: AbortSignal.timeout(15 * 60 * 1000), redirect: 'error'
      });
      const result = await response.json();
      const accepted = response.ok && typeof result.data?.asyncId === 'string';
      if (!accepted) {
        const error = programError(result);
        await store.finishJob(id, 'FAILED', { error });
        res.status(/compil|pars|syntax|not supported|must be/i.test(error) ? 422 : 502).json({ id, status: 'FAILED', error });
        return;
      }
      await store.finishJob(id, 'QUEUED', result);
      res.status(202).json({ id, status: 'QUEUED', result });
    } catch (error) {
      await store.finishJob(id, 'FAILED', { error: 'Execution service unavailable or timed out' });
      throw error;
    }
  }));
  app.get('/api/clusters/:roomId/jobs', asyncRoute(async (req, res) => {
    const room = await roomForOwner(req);
    const jobs = await store.jobs(room.id);
    for (const job of jobs) {
      if (!['QUEUED', 'PROCESSING'].includes(job.status)) continue;
      const previous = JSON.parse(job.result_json);
      const target = new URL('/status', middleware);
      target.searchParams.set('uuid', previous.data.asyncId);
      target.searchParams.set('clusterId', room.id);
      let response;
      try {
        response = await fetchImpl(target, { signal: AbortSignal.timeout(30000), redirect: 'error' });
      } catch {
        continue;
      }
      if (!response.ok) {
        const lost = { ...previous, error: 'The execution service no longer tracks this job (it may have restarted). Run it again.' };
        await store.finishJob(job.id, 'FAILED', lost);
        job.status = 'FAILED'; job.result_json = JSON.stringify(lost);
        continue;
      }
      const result = await response.json();
      const upstream = result.data?.taskStatus;
      const status = ['SUCCESS', 'COMPLETED'].includes(upstream) ? 'SUCCESS' :
        ['FAILURE', 'FAILED'].includes(upstream) ? 'FAILED' : 'PROCESSING';
      const combined = { ...previous, latest: result };
      await store.finishJob(job.id, status, combined);
      job.status = status; job.result_json = JSON.stringify(combined);
    }
    res.json(jobs);
  }));
  app.get('/api/models', asyncRoute(async (req, res) => {
    const ownerId = await owner(req);
    const catalog = models.map(({ id, name }) => ({ id, name, status: 'READY' }));
    res.json(userModels ? [...catalog, ...await userModels.list(ownerId)] : catalog);
  }));
  function requireUserModels() {
    if (!userModels) throw new HttpError(503, 'Adding models is not enabled on this server');
    return userModels;
  }
  app.post('/api/models/upload', asyncRoute(async (req, res) => {
    const ownerId = await owner(req);
    const manager = requireUserModels();
    if (req.is('application/json')) throw new HttpError(415, 'Upload the GGUF file as application/octet-stream');
    const name = text(req.query.name, 'Model name');
    const size = Number(req.get('content-length'));
    res.status(201).json(await manager.upload(ownerId, name, req, Number.isFinite(size) && size > 0 ? size : null));
  }));
  app.post('/api/models', asyncRoute(async (req, res) => {
    const ownerId = await owner(req);
    const manager = requireUserModels();
    const url = text(req.body.url, 'Model URL', 2048);
    let fallback = '';
    try { fallback = decodeURIComponent(new URL(url).pathname.split('/').pop()).replace(/\.gguf$/i, '').trim(); } catch { /* invalid URLs are rejected below */ }
    const name = text(req.body.name || fallback.slice(0, 120) || 'Model', 'Model name');
    res.status(202).json(await manager.importUrl(ownerId, name, url));
  }));
  app.delete('/api/models/:id', asyncRoute(async (req, res) => {
    const ownerId = await owner(req);
    await requireUserModels().remove(ownerId, req.params.id);
    res.json({ success: true });
  }));
  /** Validates a chat transcript: alternating user/assistant turns ending with the new question. */
  function chatMessages(body) {
    if (body.messages === undefined) return [{ role: 'user', content: text(body.question, 'Question', 16000) }];
    const messages = body.messages;
    if (!Array.isArray(messages) || !messages.length || messages.length > 99) throw new HttpError(400, 'messages must hold between 1 and 99 chat turns');
    let total = 0;
    messages.forEach((message, index) => {
      const role = index % 2 === 0 ? 'user' : 'assistant';
      if (!message || message.role !== role) throw new HttpError(400, 'Chat turns must alternate user and assistant, starting with user');
      if (typeof message.content !== 'string' || message.content.length > 16000) throw new HttpError(400, 'Each chat turn must be text of at most 16000 characters');
      total += message.content.length;
    });
    if (messages.length % 2 === 0) throw new HttpError(400, 'The last chat turn must be the user\'s question');
    if (!messages.at(-1).content.trim()) throw new HttpError(400, 'Question is required');
    if (total > 200000) throw new HttpError(400, 'This conversation is too long; start a new chat');
    return messages.map(({ role, content }) => ({ role, content }));
  }
  app.post('/api/clusters/:roomId/chat', asyncRoute(async (req, res) => {
    const room = await roomForOwner(req);
    const messages = chatMessages(req.body);
    const question = messages.at(-1).content;
    const model = models.find(item => item.id === req.body.model) || await userModels?.resolve(room.owner_id, req.body.model);
    if (!model || !infer) throw new HttpError(503, 'No model configured for inference');
    if (activeChats.has(room.id)) throw new HttpError(409, 'This cluster is answering another question');
    const devices = await store.devices(room.id);
    if (!devices.some(device => {
      const value = device.last_seen;
      const lastSeen = value instanceof Date ? value : new Date(typeof value === 'string' && !value.includes('T') ? value + 'Z' : value);
      return !device.revoked && value && Date.now() - lastSeen.getTime() < 60000;
    })) {
      throw new HttpError(409, 'Start a client joined to this room before asking a question');
    }
    activeChats.add(room.id);
    const id = crypto.randomUUID();
    try {
      await store.createJob(id, room.id);
      const result = await infer({ model, messages, roomId: room.room_id, apiToken: sessionToken(req), publicUrl: publicOrigin });
      const { text: answer, droppedTurns = 0 } = typeof result === 'string' ? { text: result } : result;
      await store.finishJob(id, 'SUCCESS', { question, model: model.id, answer, turns: messages.length, droppedTurns });
      res.json({ id, answer, droppedTurns });
    } catch (error) {
      await store.finishJob(id, 'FAILED', { error: error.message });
      if (error.code === 'INFERENCE_BACKEND_NOT_READY') {
        throw new HttpError(503, 'The service cannot answer questions right now. Please try again later.');
      }
      if (error.code === 'PROMPT_TOO_LONG') {
        throw new HttpError(400, 'Question is too long for this model\'s context window. Please shorten it or start a new chat.');
      }
      throw new HttpError(502, 'No answer was produced. Check that a device in this room is running the client.');
    } finally { activeChats.delete(room.id); }
  }));
  async function downloads() {
    let manifest;
    try { manifest = JSON.parse(await fs.readFile(path.join(releasesDir, 'manifest.json'), 'utf8')); }
    catch (error) { if (error.code === 'ENOENT') return {}; throw error; }
    const available = {};
    for (const platform of platforms) {
      const entry = manifest[platform];
      if (!entry || entry.available === false) continue;
      if (typeof entry.file !== 'string' || path.basename(entry.file) !== entry.file || !/^[a-f0-9]{64}$/.test(entry.sha256)) throw new Error('Invalid installer release manifest');
      const hash = crypto.createHash('sha256');
      for await (const chunk of require('node:fs').createReadStream(path.join(releasesDir, entry.file))) hash.update(chunk);
      if (hash.digest('hex') !== entry.sha256) throw new Error(`Installer checksum mismatch: ${platform}`);
      available[platform] = { ...entry, url: `/downloads/${platform}` };
    }
    return available;
  }
  app.get('/api/downloads', asyncRoute(async (req, res) => {
    const available = await downloads();
    res.json([...platforms].map(platform => ({ platform, available: Boolean(available[platform]), ...available[platform] })));
  }));
  app.get('/downloads/:platform', asyncRoute(async (req, res) => {
    const entry = (await downloads())[req.params.platform];
    if (!entry) throw new HttpError(404, 'This installer is not available yet');
    res.download(path.join(releasesDir, entry.file), entry.file);
  }));
  app.get('/health', asyncRoute(async (req, res) => {
    await store.pool.execute('SELECT 1');
    res.json({ status: 'healthy' });
  }));
  app.use(express.static(path.join(__dirname, 'public')));
  app.use((req, res) => res.status(404).json({ error: 'Endpoint not found' }));
  app.use((error, req, res, next) => {
    if (res.headersSent) return next(error);
    const status = error.status || 500;
    if (status >= 500) console.error('Portal request failed:', error.cause?.code || error.code || error.name);
    res.status(status).json({ error: error instanceof HttpError || status < 500 ? error.message : 'Service unavailable; check server logs' });
  });
  return app;
}
module.exports = { createApp };
