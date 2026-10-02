'use strict';
const path = require('node:path');
const { createStore } = require('./store');
const { createApp } = require('./app');
const { loadModels, createInference } = require('./inference');

async function main() {
  const models = await loadModels(process.env.MODELS_FILE, process.env.MODEL_ROOT);
  const store = createStore();
  const app = createApp({
    store, models, publicUrl: process.env.PUBLIC_URL || 'http://localhost:8090',
    middlewareUrl: process.env.MIDDLEWARE_URL || 'http://127.0.0.1:8888',
    orchestratorUrl: process.env.ORCHESTRATOR_URL || process.env.MIDDLEWARE_URL || 'http://127.0.0.1:8888', modelRoot: process.env.MODEL_ROOT,
    releasesDir: process.env.RELEASES_DIR || path.resolve(__dirname, '../installer/releases'),
    production: process.env.NODE_ENV === 'production',
    infer: createInference({ runner: path.resolve(__dirname, '../../sharded-llm/run_gguf_shards.py'), python: process.env.PYTHON || 'python3' })
  });
  const server = app.listen(Number(process.env.PORT || 8090), process.env.HOST || '127.0.0.1', () => {
    console.log(`Ramanujan Compute Cluster portal: ${process.env.PUBLIC_URL || 'http://localhost:8090'}`);
  });
  function shutdown() { server.close(async () => { await store.pool.end(); }); }
  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
}
main().catch(error => { console.error('Portal startup failed:', error.message); process.exitCode = 1; });
