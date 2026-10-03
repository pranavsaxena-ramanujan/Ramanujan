'use strict';
// Owner-provided GGUF models: browser uploads or HTTPS downloads, converted into shard packages.
const { spawn } = require('node:child_process');
const crypto = require('node:crypto');
const dns = require('node:dns');
const fs = require('node:fs');
const https = require('node:https');
const net = require('node:net');
const path = require('node:path');
const { Transform } = require('node:stream');
const { pipeline } = require('node:stream/promises');
const { HttpError } = require('./security');
const { detectChatFormat, defaultGenerationPrefix } = require('./inference');

const fsp = fs.promises;
const GB = 1024 ** 3;
const IN_PROGRESS = ['UPLOADING', 'DOWNLOADING', 'QUEUED', 'CONVERTING'];
const MAX_IN_PROGRESS = 3;

// Separate lists: Node matches IPv4 addresses against IPv4-mapped IPv6 rules.
const blockedV4 = new net.BlockList();
const blockedV6 = new net.BlockList();
for (const [address, prefix] of [['0.0.0.0', 8], ['10.0.0.0', 8], ['100.64.0.0', 10], ['127.0.0.0', 8], ['169.254.0.0', 16],
  ['172.16.0.0', 12], ['192.0.0.0', 24], ['192.168.0.0', 16], ['198.18.0.0', 15], ['224.0.0.0', 4], ['240.0.0.0', 4]]) {
  blockedV4.addSubnet(address, prefix, 'ipv4');
}
for (const [address, prefix] of [['::', 127], ['::ffff:0:0', 96], ['64:ff9b::', 96], ['2002::', 16], ['fc00::', 7],
  ['fe80::', 10], ['ff00::', 8]]) {
  blockedV6.addSubnet(address, prefix, 'ipv6');
}
function isPublicAddress(address) {
  const family = net.isIP(address);
  if (family === 4) return !blockedV4.check(address, 'ipv4');
  return family === 6 && !blockedV6.check(address, 'ipv6');
}
function privateAddressError() {
  return Object.assign(new HttpError(400, 'Model URLs must point to a public internet host'), { code: 'EPRIVATEADDRESS' });
}
// Every connection (including redirects) resolves through this, so DNS cannot point the download at the server's network.
function publicLookup(hostname, options, callback) {
  dns.lookup(hostname, { ...options, all: true }, (error, addresses) => {
    if (error) return callback(error);
    if (!addresses.length || addresses.some(entry => !isPublicAddress(entry.address))) return callback(privateAddressError());
    if (options.all) callback(null, addresses);
    else callback(null, addresses[0].address, addresses[0].family);
  });
}
function checkModelUrl(value) {
  let url;
  try { url = new URL(value); } catch { throw new HttpError(400, 'Enter a valid https:// URL to a .gguf file'); }
  if (url.protocol !== 'https:') throw new HttpError(400, 'Model URLs must use https://');
  if (url.username || url.password) throw new HttpError(400, 'Model URLs must not contain credentials');
  const host = url.hostname.replace(/^\[|\]$/g, '');
  if (net.isIP(host) && !isPublicAddress(host)) throw privateAddressError();
  if (/^localhost$|\.localhost$|\.local$|\.internal$/i.test(host)) throw privateAddressError();
  return url;
}

/** Opens an HTTPS download, following up to five redirects, each re-checked against private networks. */
async function openDownload(value, signal) {
  let url = checkModelUrl(value);
  for (let hop = 0; hop <= 5; hop++) {
    const response = await new Promise((resolve, reject) => {
      const request = https.get(url, { lookup: publicLookup, signal, headers: { 'User-Agent': 'Ramanujan-Compute-Cluster' } }, resolve);
      request.setTimeout(60000, () => request.destroy(new Error('Model download timed out')));
      request.once('error', reject);
    });
    if ([301, 302, 303, 307, 308].includes(response.statusCode) && response.headers.location) {
      response.resume();
      url = checkModelUrl(new URL(response.headers.location, url).href);
      continue;
    }
    if (response.statusCode !== 200) {
      response.resume();
      throw new HttpError(400, `The model URL returned HTTP ${response.statusCode}`);
    }
    response.setTimeout(120000, () => response.destroy(new Error('Model download stalled')));
    const size = Number(response.headers['content-length']);
    return { stream: response, size: Number.isFinite(size) && size > 0 ? size : null };
  }
  throw new HttpError(400, 'The model URL redirected too many times');
}

async function freeBytes(dir) {
  const stats = await fsp.statfs(dir);
  return stats.bavail * stats.bsize;
}
const formatBytes = bytes => bytes >= GB ? `${(bytes / GB).toFixed(1)} GB` : `${Math.max(0, Math.round(bytes / 1024 ** 2))} MB`;

function runPython({ python, cwd, args, signal }) {
  return new Promise((resolve, reject) => {
    const env = { PYTHONPATH: cwd };
    for (const key of ['PATH', 'HOME', 'LANG', 'LC_ALL', 'VIRTUAL_ENV', 'SYSTEMROOT', 'TMPDIR']) if (process.env[key]) env[key] = process.env[key];
    const child = spawn(python, args, { cwd, env, signal, stdio: ['ignore', 'ignore', 'pipe'] });
    let stderr = '';
    child.stderr.on('data', chunk => { stderr = (stderr + chunk).slice(-4000); });
    child.once('error', reject);
    child.once('close', code => {
      if (code === 0) return resolve();
      const lines = stderr.trim().split('\n').filter(Boolean);
      reject(new Error(lines.at(-1) || `converter exited with code ${code}`));
    });
  });
}

function defaultConverter({ converterDir, python, capacityAware }) {
  return async ({ gguf, shards, plan, signal }) => {
    await runPython({ python, cwd: converterDir, signal,
      args: ['-m', 'ramanujan_shards.emit_gguf', '--gguf', gguf, '--output-dir', shards,
        ...(capacityAware ? ['--per-layer'] : ['--shards', '4'])] });
    await runPython({ python, cwd: converterDir, signal,
      args: ['-m', 'ramanujan_shards.gguf_ir_plan', '--package', shards, '--gguf', gguf, '--output-dir', plan] });
  };
}

async function directoryBytes(dir) {
  let total = 0;
  let entries;
  try { entries = await fsp.readdir(dir, { withFileTypes: true }); } catch { return 0; }
  for (const entry of entries) {
    const child = path.join(dir, entry.name);
    if (entry.isDirectory()) total += await directoryBytes(child);
    else if (entry.isFile()) total += (await fsp.stat(child).catch(() => ({ size: 0 }))).size;
  }
  return total;
}

function publicModel(row) {
  return { id: row.id, name: row.name, owned: true, source: row.source, sourceUrl: row.source_url || undefined, status: row.status,
    progress: Number(row.progress) || 0, detail: row.detail || undefined, sizeBytes: row.size_bytes == null ? undefined : Number(row.size_bytes),
    architecture: row.architecture || undefined, chatFormat: row.chat_format || undefined, createdAt: row.created_at };
}

function createModelManager({ store, dir, converterDir, python = 'python3', download = openDownload, convert, cache,
  reserveBytes = 2 * GB, maxContext = 1024, maxNewTokens = 128, capacityAware = false }) {
  const runConvert = convert || defaultConverter({ converterDir, python, capacityAware });
  const jobs = new Map();
  const queue = [];
  let converting = false;
  let realDir;
  const modelDir = id => path.join(dir, id);

  async function init() {
    await fsp.mkdir(dir, { recursive: true, mode: 0o700 });
    realDir = await fsp.realpath(dir);
    for (const row of await store.modelsWithStatus(IN_PROGRESS)) {
      await store.updateModel(row.id, { status: 'FAILED', progress: 0, detail: 'Interrupted by a server restart. Delete it and add it again.' });
      await fsp.rm(modelDir(row.id), { recursive: true, force: true });
    }
  }

  function progressWriter(id) {
    let last = 0;
    return (fraction, force = false) => {
      const now = Date.now();
      if (!force && now - last < 1000) return;
      last = now;
      store.updateModel(id, { progress: Math.max(0, Math.min(1, fraction)) }).catch(() => {});
    };
  }

  async function ensureRoom(bytes, label) {
    const available = await freeBytes(dir) - reserveBytes;
    if (available < bytes) throw new HttpError(507, `Not enough server disk space ${label} (needs about ${formatBytes(bytes)}, ${formatBytes(available)} free)`);
  }

  /** Streams a GGUF to <id>/model.gguf, checking the magic bytes and keeping free disk above the reserve. */
  async function receive(id, stream, size) {
    // The GGUF plus its shard package need roughly twice the file size.
    if (size) await ensureRoom(size * 2, 'for this model');
    await fsp.mkdir(modelDir(id), { recursive: true, mode: 0o700 });
    const report = progressWriter(id);
    let bytes = 0, header = Buffer.alloc(0), nextDiskCheck = 0;
    const meter = new Transform({
      transform(chunk, encoding, callback) {
        if (header.length < 4) {
          header = Buffer.concat([header, chunk.subarray(0, 4 - header.length)]);
          if (header.length === 4 && header.toString('latin1') !== 'GGUF') return callback(new HttpError(400, 'This is not a GGUF file'));
        }
        bytes += chunk.length;
        if (size) report(bytes / size);
        if (bytes < nextDiskCheck) return callback(null, chunk);
        nextDiskCheck = bytes + 256 * 1024 ** 2;
        freeBytes(dir).then(free => {
          callback(free < reserveBytes ? new HttpError(507, 'The server ran out of disk space while receiving the model') : null, chunk);
        }, callback);
      }
    });
    await pipeline(stream, meter, fs.createWriteStream(path.join(modelDir(id), 'model.gguf'), { mode: 0o600 }));
    if (header.toString('latin1') !== 'GGUF') throw new HttpError(400, 'This is not a GGUF file');
    if (size && bytes !== size) throw new HttpError(400, 'The model transfer was incomplete');
    report(1, true);
    return bytes;
  }

  async function failed(id, error) {
    await fsp.rm(modelDir(id), { recursive: true, force: true }).catch(() => {});
    const detail = String(error?.message || error || 'Failed').slice(0, 600);
    await store.updateModel(id, { status: 'FAILED', progress: 0, detail }).catch(() => {});
  }

  function enqueue(id) {
    queue.push(id);
    pump();
  }
  async function pump() {
    if (converting) return;
    converting = true;
    try {
      while (queue.length) await convertOne(queue.shift());
    } finally { converting = false; }
  }
  async function convertOne(id) {
    const gguf = path.join(modelDir(id), 'model.gguf');
    const shards = path.join(modelDir(id), 'shards');
    const plan = path.join(modelDir(id), 'ir-plan');
    const controller = new AbortController();
    let finish;
    jobs.set(id, { controller, done: new Promise(resolve => { finish = resolve; }) });
    const report = progressWriter(id);
    let ticker;
    try {
      const size = (await fsp.stat(gguf)).size;
      await ensureRoom(size * 1.05, 'to convert this model');
      await store.updateModel(id, { status: 'CONVERTING', progress: 0, detail: null });
      // The converter stages shards in a temporary sibling directory, so measure everything beside the GGUF.
      ticker = setInterval(() => directoryBytes(modelDir(id)).then(bytes => report(0.95 * (bytes - size) / size)), 2000);
      await runConvert({ gguf, shards, plan, signal: controller.signal });
      clearInterval(ticker);
      const metadata = JSON.parse(await fsp.readFile(path.join(plan, 'gguf-metadata.json'), 'utf8'));
      const architecture = String(metadata['general.architecture'] || 'unknown').slice(0, 64);
      const template = metadata['tokenizer.chat_template'];
      const chatFormat = detectChatFormat(template);
      const contextLength = Number(metadata[`${architecture}.context_length`]) || null;
      await fsp.rm(gguf, { force: true });
      if (cache) {
        await store.updateModel(id, { detail: 'Saving converted model to private cloud storage', progress: 0.95 });
        await cache.save(id, modelDir(id), controller.signal);
      }
      await store.updateModel(id, { status: 'READY', progress: 1, detail: null, architecture, chat_format: chatFormat,
        generation_prefix: defaultGenerationPrefix(chatFormat, template), context_length: contextLength });
    } catch (error) {
      if (!controller.signal.aborted) await failed(id, error);
    } finally {
      clearInterval(ticker);
      jobs.delete(id);
      finish();
    }
  }

  async function checkCapacity(ownerId) {
    const rows = await store.ownerModels(ownerId);
    if (rows.filter(row => IN_PROGRESS.includes(row.status)).length >= MAX_IN_PROGRESS) {
      throw new HttpError(429, `Wait for your other ${MAX_IN_PROGRESS} models to finish before adding another`);
    }
  }

  async function upload(ownerId, name, stream, size) {
    await checkCapacity(ownerId);
    const id = crypto.randomUUID();
    await store.createModel({ id, ownerId, name, source: 'upload', status: 'UPLOADING' });
    try {
      const bytes = await receive(id, stream, size);
      await store.updateModel(id, { status: 'QUEUED', progress: 0, size_bytes: bytes });
    } catch (error) {
      await fsp.rm(modelDir(id), { recursive: true, force: true }).catch(() => {});
      await store.deleteModel(id).catch(() => {});
      throw error;
    }
    enqueue(id);
    return publicModel(await store.ownerModel(id, ownerId));
  }

  async function importUrl(ownerId, name, url) {
    checkModelUrl(url);
    await checkCapacity(ownerId);
    const id = crypto.randomUUID();
    await store.createModel({ id, ownerId, name, source: 'url', sourceUrl: url, status: 'DOWNLOADING' });
    const controller = new AbortController();
    let finish;
    jobs.set(id, { controller, done: new Promise(resolve => { finish = resolve; }) });
    (async () => {
      try {
        const { stream, size } = await download(url, controller.signal);
        if (size) await store.updateModel(id, { size_bytes: size });
        const bytes = await receive(id, stream, size);
        await store.updateModel(id, { status: 'QUEUED', progress: 0, size_bytes: bytes });
        jobs.delete(id);
        finish();
        enqueue(id);
      } catch (error) {
        if (!controller.signal.aborted) await failed(id, error.code === 'EPRIVATEADDRESS' ? error : new Error(`Download failed: ${error.message}`));
        jobs.delete(id);
        finish();
      }
    })();
    return publicModel(await store.ownerModel(id, ownerId));
  }

  async function remove(ownerId, id) {
    const row = await store.ownerModel(id, ownerId);
    if (!row) throw new HttpError(404, 'Model not found');
    const queued = queue.indexOf(id);
    if (queued !== -1) queue.splice(queued, 1);
    const job = jobs.get(id);
    if (job) {
      job.controller.abort();
      await job.done;
    }
    if (cache) await cache.remove(id);
    await fsp.rm(modelDir(id), { recursive: true, force: true });
    await store.deleteModel(id);
  }

  async function list(ownerId) { return (await store.ownerModels(ownerId)).map(publicModel); }

  /** The inference settings for one of the owner's READY models, or null. */
  async function resolve(ownerId, id) {
    if (typeof id !== 'string' || !/^[0-9a-f-]{36}$/.test(id)) return null;
    const row = await store.ownerModel(id, ownerId);
    if (!row || row.status !== 'READY') return null;
    try {
      await fsp.access(path.join(modelDir(id), 'shards', 'model-manifest.json'));
      await fsp.access(path.join(modelDir(id), 'ir-plan', 'gguf-metadata.json'));
    } catch (error) {
      if (error.code !== 'ENOENT') throw error;
      if (!cache) throw new HttpError(503, 'Converted model files are missing; delete the model and add it again');
      await cache.restore(id, modelDir(id), reserveBytes);
    }
    const context = Math.min(maxContext, Number(row.context_length) || maxContext);
    return { id: row.id, name: row.name, package: path.join(modelDir(id), 'shards'), metadata: path.join(modelDir(id), 'ir-plan', 'gguf-metadata.json'),
      chatFormat: row.chat_format, generationPrefix: row.generation_prefix ?? undefined, maxContext: context,
      maxNewTokens: Math.min(maxNewTokens, Math.floor(context / 2)), requestTimeout: 1800 };
  }

  /** True when a resolved file path belongs to one of the owner's READY models. */
  async function ownsFile(ownerId, realPath) {
    if (!realDir || !realPath.startsWith(realDir + path.sep)) return false;
    const id = realPath.slice(realDir.length + 1).split(path.sep)[0];
    const row = await store.ownerModel(id, ownerId);
    return Boolean(row && row.status === 'READY');
  }

  return { init, upload, importUrl, remove, list, resolve, ownsFile, dir };
}

module.exports = { createModelManager, openDownload, checkModelUrl, isPublicAddress };
