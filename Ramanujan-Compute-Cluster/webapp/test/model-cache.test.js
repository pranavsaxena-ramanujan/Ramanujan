'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { createModelCache } = require('../model-cache');

function memoryBucket() {
  const objects = new Map();
  return {
    objects,
    upload: async (filename, { destination }) => objects.set(destination, await fs.readFile(filename)),
    file: name => ({
      save: async data => objects.set(name, Buffer.from(data)),
      download: async ({ destination } = {}) => {
        if (!objects.has(name)) throw new Error('Object not found');
        if (destination) await fs.writeFile(destination, objects.get(name));
        return [objects.get(name)];
      }
    }),
    deleteFiles: async ({ prefix }) => { for (const name of objects.keys()) if (name.startsWith(prefix)) objects.delete(name); }
  };
}

test('private cloud cache saves packages, restores after disk loss and deletes only the model prefix', async t => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'rj-cache-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const dir = path.join(root, 'model');
  await fs.mkdir(path.join(dir, 'shards', 'shard-00'), { recursive: true });
  await fs.mkdir(path.join(dir, 'ir-plan'), { recursive: true });
  await fs.writeFile(path.join(dir, 'shards', 'model-manifest.json'), '{}');
  await fs.writeFile(path.join(dir, 'shards', 'shard-00', 'weights.bin'), 'weights');
  await fs.writeFile(path.join(dir, 'ir-plan', 'gguf-metadata.json'), '{"general.architecture":"qwen2"}');
  const bucket = memoryBucket();
  bucket.objects.set('other/keep', Buffer.from('keep'));
  const cache = createModelCache({ bucket });
  const id = '12345678-1234-1234-1234-123456789abc';
  await cache.save(id, dir);
  assert.ok(bucket.objects.has(`models/${id}/manifest.json`));
  await fs.rm(dir, { recursive: true });
  await Promise.all([cache.restore(id, dir), cache.restore(id, dir)]);
  assert.equal(await fs.readFile(path.join(dir, 'shards', 'shard-00', 'weights.bin'), 'utf8'), 'weights');
  await cache.remove(id);
  assert.deepEqual([...bucket.objects.keys()], ['other/keep']);
});

test('cache rejects path traversal, incomplete transfers, invalid IDs and insufficient disk', async t => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'rj-cache-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const bucket = memoryBucket();
  const cache = createModelCache({ bucket });
  const id = '12345678-1234-1234-1234-123456789abc';
  const manifest = files => Buffer.from(JSON.stringify({ version: 1, files }));
  bucket.objects.set(`models/${id}/manifest.json`, manifest([{ name: 'shards/../../escape', size: 1 }]));
  await assert.rejects(cache.restore(id, path.join(root, 'model')), /Invalid cached/);
  await assert.rejects(cache.restore('../escape', path.join(root, 'model')), /Invalid cached model ID/);
  const entries = [{ name: 'shards/model-manifest.json', size: 2 }, { name: 'ir-plan/gguf-metadata.json', size: 2 }];
  bucket.objects.set(`models/${id}/manifest.json`, manifest(entries));
  bucket.objects.set(`models/${id}/shards/model-manifest.json`, Buffer.from('x'));
  await assert.rejects(cache.restore(id, path.join(root, 'model'), Number.MAX_SAFE_INTEGER), /disk space/);
  await assert.rejects(cache.restore(id, path.join(root, 'model')), /Incomplete/);
  assert.deepEqual(await fs.readdir(root), []);
});
