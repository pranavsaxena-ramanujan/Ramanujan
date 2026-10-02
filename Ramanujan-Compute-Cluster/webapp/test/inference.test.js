'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { loadModels, createInference, trimAtStop } = require('../inference');

test('model catalog resolves paths and rejects paths outside its root', async t => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'rj-catalog-test-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  await fs.mkdir(path.join(root, 'shards'));
  await fs.writeFile(path.join(root, 'metadata.json'), '{}');
  const catalog = path.join(root, 'models.json');
  const model = { id: 'tiny', name: 'Tiny', package: path.join(root, 'shards'), metadata: path.join(root, 'metadata.json') };
  await fs.writeFile(catalog, JSON.stringify([model]));
  assert.deepEqual(await loadModels(catalog, root), [{ ...model, maxContext: 1024, maxNewTokens: 128, requestTimeout: 120,
    package: await fs.realpath(model.package), metadata: await fs.realpath(model.metadata) }]);
  await fs.writeFile(catalog, JSON.stringify([{ ...model, metadata: __filename }]));
  await assert.rejects(loadModels(catalog, root), /inside MODEL_ROOT/);
  await fs.writeFile(catalog, JSON.stringify([{ ...model, requestTimeout: 1801 }]));
  await assert.rejects(loadModels(catalog, root), /requestTimeout/);
});

test('driver uses the room gateway and does not inherit database credentials', async t => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'rj-driver-test-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const runner = path.join(root, 'driver.js');
  await fs.writeFile(runner, `
    if (process.env.DB_PASSWORD) process.exit(9);
    if (process.env.RAMANUJAN_PORTAL_TOKEN !== 'test-capability') process.exit(10);
    if (!process.argv.includes('https://portal.test/api/clusters/room1/homelab')) process.exit(11);
    if (!process.argv.includes('User: Question? Assistant:')) process.exit(12);
    if (process.argv[process.argv.indexOf('--timeout') + 1] !== '1800') process.exit(13);
    console.log(JSON.stringify({event:'completion',text:'An answer.'}));
  `);
  const previous = process.env.DB_PASSWORD;
  process.env.DB_PASSWORD = 'test-only-not-a-real-password';
  t.after(() => {
    if (previous === undefined) delete process.env.DB_PASSWORD;
    else process.env.DB_PASSWORD = previous;
  });
  const infer = createInference({ runner, python: process.execPath });
  const answer = await infer({ model: { package: '/model/shards', metadata: '/model/meta.json',
    promptTemplate: 'User: {question} Assistant:', requestTimeout: 1800 }, question: 'Question?', roomId: 'room1',
  apiToken: 'test-capability', publicUrl: 'https://portal.test' });
  assert.equal(answer, 'An answer.');
});

test('driver failure and missing generated text are not success-shaped answers', async t => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'rj-failed-driver-test-'));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const runner = path.join(root, 'driver.js');
  const request = { model: { package: '/model', metadata: '/meta' }, question: 'Question',
    roomId: 'room', apiToken: 'test-capability', publicUrl: 'https://portal.test' };
  await fs.writeFile(runner, 'process.exit(1);');
  await assert.rejects(createInference({ runner, python: process.execPath })(request), /inference failed/);
  await fs.writeFile(runner, 'console.log(JSON.stringify({event:"model"}));');
  await assert.rejects(createInference({ runner, python: process.execPath })(request), /generated text/);
  await fs.writeFile(runner, 'console.error("homelab /llm/chain failed with HTTP 404"); process.exit(1);');
  await assert.rejects(createInference({ runner, python: process.execPath })(request),
    error => error.code === 'INFERENCE_BACKEND_NOT_READY');
});

test('generated text is cut at the first end-of-turn token', () => {
  assert.equal(trimAtStop(' 4<|im_end|>junk', ['</s>', '<|im_end|>']), '4');
  assert.equal(trimAtStop('plain', ['<|im_end|>']), 'plain');
});
