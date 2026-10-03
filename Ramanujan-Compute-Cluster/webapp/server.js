'use strict';
const os = require('node:os');
const path = require('node:path');
const { createStore } = require('./store');
const { createApp } = require('./app');
const { loadModels, createInference } = require('./inference');
const { createModelManager } = require('./user-models');
const { createModelCache } = require('./model-cache');
const { createBackendFetch } = require('./backend-fetch');

async function main() {
  const models = await loadModels(process.env.MODELS_FILE, process.env.MODEL_ROOT);
  const store = createStore();
  const python = process.env.PYTHON || 'python3';
  const capacityAware = process.env.RAMANUJAN_CAPACITY_AWARE === '1';
  const backend = createBackendFetch();
  const cache = process.env.MODEL_CACHE_BUCKET ? createModelCache({
    bucket: new (require('@google-cloud/storage').Storage)().bucket(process.env.MODEL_CACHE_BUCKET)
  }) : undefined;
  const userModels = process.env.USER_MODELS === 'off' ? undefined : createModelManager({
    store, python, cache, dir: path.resolve(process.env.USER_MODEL_DIR || path.join(os.homedir(), '.ramanujan', 'portal-models')),
    converterDir: path.resolve(__dirname, '../../sharded-llm/converter'), capacityAware
  });
  await userModels?.init();
  const app = createApp({
    store, models, userModels, fetchImpl: backend.fetch, publicUrl: process.env.PUBLIC_URL || 'http://localhost:8090',
    middlewareUrl: process.env.MIDDLEWARE_URL || 'http://127.0.0.1:8888',
    orchestratorUrl: process.env.ORCHESTRATOR_URL || process.env.MIDDLEWARE_URL || 'http://127.0.0.1:8888', modelRoot: process.env.MODEL_ROOT,
    releasesDir: process.env.RELEASES_DIR || path.resolve(__dirname, '../installer/releases'),
    production: process.env.NODE_ENV === 'production',
    infer: createInference({ runner: path.resolve(__dirname, '../../sharded-llm/run_gguf_shards.py'), python, capacityAware })
  });
  const server = app.listen(Number(process.env.PORT || 8090), process.env.HOST || '127.0.0.1', () => {
    console.log(`Ramanujan Compute Cluster portal: ${process.env.PUBLIC_URL || 'http://localhost:8090'}`);
  });
  // Model uploads can take longer than Node's default five-minute request timeout.
  server.requestTimeout = 0;
  function shutdown() { server.close(async () => { await backend.close(); await store.pool.end(); }); }
  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
}
main().catch(error => { console.error('Portal startup failed:', error.message); process.exitCode = 1; });
