'use strict';
const { Agent, fetch } = require('undici');

const BACKEND_TIMEOUT_MS = 31 * 60 * 1000;

function createBackendFetch({ timeout = BACKEND_TIMEOUT_MS } = {}) {
  if (!Number.isSafeInteger(timeout) || timeout <= 0) throw new Error('Backend timeout must be a positive integer');
  // Node's default fetch headers timeout is five minutes, shorter than cold model loading.
  const dispatcher = new Agent({ headersTimeout: timeout, bodyTimeout: timeout });
  return {
    fetch: (url, init) => fetch(url, { ...init, dispatcher }),
    close: () => dispatcher.close()
  };
}

module.exports = { createBackendFetch, BACKEND_TIMEOUT_MS };
