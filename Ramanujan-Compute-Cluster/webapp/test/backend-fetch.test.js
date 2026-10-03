'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const http = require('node:http');
const { createBackendFetch, BACKEND_TIMEOUT_MS } = require('../backend-fetch');

test('backend headers and body timeouts cover cold loading beyond the default five minutes', async t => {
  assert.ok(BACKEND_TIMEOUT_MS > 30 * 60 * 1000);
  const server = http.createServer((req, res) => {
    const timer = setTimeout(() => res.end('ready'), 1500);
    req.on('close', () => clearTimeout(timer));
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(() => new Promise(resolve => { server.close(resolve); server.closeAllConnections(); }));
  const short = createBackendFetch({ timeout: 200 });
  const long = createBackendFetch({ timeout: 4000 });
  t.after(async () => { await short.close(); await long.close(); });
  const url = `http://127.0.0.1:${server.address().port}/llm/chain`;
  await assert.rejects(short.fetch(url), error => error.cause?.code === 'UND_ERR_HEADERS_TIMEOUT');
  assert.equal(await (await long.fetch(url)).text(), 'ready');
});
