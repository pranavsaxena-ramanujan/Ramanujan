'use strict';
const crypto = require('node:crypto');
const fs = require('node:fs/promises');
const path = require('node:path');

const validId = id => typeof id === 'string' && /^[0-9a-f-]{36}$/.test(id);
const validFile = name => typeof name === 'string' && /^(shards|ir-plan)\/[A-Za-z0-9_.\/-]+$/.test(name)
  && !name.split('/').some(part => !part || part === '.' || part === '..');

async function filesIn(dir, prefix = '') {
  const files = [];
  for (const entry of await fs.readdir(dir, { withFileTypes: true })) {
    const name = prefix + entry.name;
    if (entry.isDirectory()) files.push(...await filesIn(path.join(dir, entry.name), name + '/'));
    else if (entry.isFile()) files.push(name);
    else throw new Error('Model cache does not accept symlinks or special files');
  }
  return files;
}

/** Private durable packages. The manifest is published last, after all object uploads complete. */
function createModelCache({ bucket, prefix = 'models' }) {
  if (!/^[A-Za-z0-9_-]+$/.test(prefix)) throw new Error('Invalid model cache prefix');
  const pending = new Map();
  function base(id) {
    if (!validId(id)) throw new Error('Invalid cached model ID');
    return `${prefix}/${id}/`;
  }
  async function save(id, dir, signal) {
    const root = base(id);
    const entries = [];
    for (const name of await filesIn(dir)) {
      if (!validFile(name)) continue;
      signal?.throwIfAborted();
      const filename = path.join(dir, name);
      const stat = await fs.stat(filename);
      await bucket.upload(filename, { destination: root + name, resumable: stat.size > 8 * 1024 ** 2,
        validation: 'crc32c' });
      entries.push({ name, size: stat.size });
    }
    signal?.throwIfAborted();
    if (!entries.some(entry => entry.name === 'ir-plan/gguf-metadata.json')
        || !entries.some(entry => entry.name === 'shards/model-manifest.json')) {
      throw new Error('Model cache requires metadata and a shard manifest');
    }
    await bucket.file(root + 'manifest.json').save(JSON.stringify({ version: 1, files: entries }),
      { resumable: false, contentType: 'application/json', validation: 'crc32c' });
  }
  async function restore(id, dir, reserveBytes = 0) {
    if (pending.has(id)) return pending.get(id);
    const operation = (async () => {
      const root = base(id);
      const [data] = await bucket.file(root + 'manifest.json').download({ validation: 'crc32c' });
      const manifest = JSON.parse(data.toString('utf8'));
      if (manifest.version !== 1 || !Array.isArray(manifest.files) || !manifest.files.length
          || manifest.files.length > 100000
          || manifest.files.some(entry => !validFile(entry.name) || !Number.isSafeInteger(entry.size) || entry.size < 0)
          || new Set(manifest.files.map(entry => entry.name)).size !== manifest.files.length
          || !manifest.files.some(entry => entry.name === 'ir-plan/gguf-metadata.json')
          || !manifest.files.some(entry => entry.name === 'shards/model-manifest.json')) {
        throw new Error('Invalid cached model manifest');
      }
      await fs.mkdir(path.dirname(dir), { recursive: true, mode: 0o700 });
      const stats = await fs.statfs(path.dirname(dir));
      const bytes = manifest.files.reduce((sum, entry) => sum + entry.size, 0);
      if (!Number.isSafeInteger(bytes) || stats.bavail * stats.bsize < bytes + reserveBytes) {
        throw new Error('Not enough disk space to restore the cached model');
      }
      const staging = dir + '.restore-' + crypto.randomUUID();
      await fs.mkdir(staging, { mode: 0o700 });
      try {
        for (const entry of manifest.files) {
          const target = path.join(staging, entry.name);
          await fs.mkdir(path.dirname(target), { recursive: true, mode: 0o700 });
          await bucket.file(root + entry.name).download({ destination: target, validation: 'crc32c' });
          if ((await fs.stat(target)).size !== entry.size) throw new Error('Incomplete cached model download');
          await fs.chmod(target, 0o600);
        }
        await fs.rm(dir, { recursive: true, force: true });
        await fs.rename(staging, dir);
      } finally { await fs.rm(staging, { recursive: true, force: true }); }
    })();
    pending.set(id, operation);
    try { await operation; } finally { pending.delete(id); }
  }
  async function remove(id) {
    if (pending.has(id)) await pending.get(id);
    await bucket.deleteFiles({ prefix: base(id) });
  }
  return { save, restore, remove };
}

module.exports = { createModelCache };
