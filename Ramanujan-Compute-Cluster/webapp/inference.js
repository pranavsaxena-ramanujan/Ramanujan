'use strict';
const { spawn } = require('node:child_process');
const fs = require('node:fs/promises');
const path = require('node:path');

async function loadModels(filename, modelRoot) {
  if (!filename) return [];
  const models = JSON.parse(await fs.readFile(filename, 'utf8'));
  if (!Array.isArray(models) || !modelRoot) throw new Error('MODELS_FILE requires an array and MODEL_ROOT');
  const root = await fs.realpath(modelRoot);
  const ids = new Set();
  for (const model of models) {
    if (!/^[a-zA-Z0-9_-]+$/.test(model.id) || ids.has(model.id) || typeof model.name !== 'string') throw new Error('Invalid model catalog');
    ids.add(model.id);
    for (const key of ['package', 'metadata']) {
      const real = await fs.realpath(model[key]);
      if (!real.startsWith(root + path.sep)) throw new Error('Model catalog paths must be inside MODEL_ROOT');
      model[key] = real;
    }
    if (model.promptTemplate && (typeof model.promptTemplate !== 'string' || !model.promptTemplate.includes('{question}'))) throw new Error('promptTemplate must contain {question}');
    model.maxContext = model.maxContext ?? 1024;
    model.maxNewTokens = model.maxNewTokens ?? 128;
    model.requestTimeout = model.requestTimeout ?? 120;
    if (!Number.isInteger(model.requestTimeout) || model.requestTimeout < 1 || model.requestTimeout > 1800) {
      throw new Error('requestTimeout must be an integer between 1 and 1800 seconds');
    }
    for (const key of ['maxContext', 'maxNewTokens']) {
      if (!Number.isInteger(model[key]) || model[key] < 1 || model[key] > 32768) throw new Error(`${key} must be an integer between 1 and 32768`);
    }
    if (model.maxNewTokens >= model.maxContext) throw new Error('maxNewTokens must be smaller than maxContext');
    if (model.stopSequences !== undefined && (!Array.isArray(model.stopSequences)
        || model.stopSequences.some(s => typeof s !== 'string' || !s))) throw new Error('stopSequences must be non-empty strings');
  }
  return models;
}

const DEFAULT_STOPS = ['<|im_end|>', '<|endoftext|>', '<|eot_id|>', '</s>'];

function trimAtStop(text, stops) {
  let end = text.length;
  for (const stop of stops) {
    const index = text.indexOf(stop);
    if (index !== -1 && index < end) end = index;
  }
  return text.slice(0, end).trim();
}

function createInference({ runner, python = 'python3' }) {
  return ({ model, question, roomId, apiToken, publicUrl }) => new Promise((resolve, reject) => {
    const prompt = model.promptTemplate ? model.promptTemplate.replace('{question}', question) : question;
    const env = { RAMANUJAN_PORTAL_TOKEN: apiToken };
    for (const key of ['PATH', 'HOME', 'LANG', 'LC_ALL', 'PYTHONPATH', 'VIRTUAL_ENV', 'SYSTEMROOT']) {
      if (process.env[key]) env[key] = process.env[key];
    }
    const child = spawn(python, [runner, '--runtime', 'native', '--homelab', `${publicUrl}/api/clusters/${roomId}/homelab`,
      '--package', model.package, '--metadata', model.metadata, '--prompt', prompt, '--max-new-tokens', String(model.maxNewTokens ?? 128),
      '--max-context', String(model.maxContext ?? 1024), '--timeout', String(model.requestTimeout ?? 120)],
    { env, stdio: ['ignore', 'pipe', 'pipe'] });
    let stdout = '', stderr = '';
    // The first request may wait for a device to cache the whole model.
    const timer = setTimeout(() => child.kill('SIGTERM'), ((model.requestTimeout ?? 120) + 600) * 1000);
    child.stdout.on('data', chunk => {
      stdout += chunk.toString();
      if (stdout.length > 4000000) child.kill('SIGTERM');
    });
    child.stderr.on('data', chunk => { stderr = (stderr + chunk.toString()).slice(-16000); });
    child.once('error', error => { clearTimeout(timer); reject(error); });
    child.once('close', code => {
      clearTimeout(timer);
      if (code !== 0) {
        console.error('Inference worker failed:', stderr.replaceAll(apiToken, '[redacted]'));
        const error = new Error('LLM inference failed; check that joined devices can load the native runtime and model');
        if (/homelab \/llm\/(?:chain|step) failed with HTTP 404/.test(stderr)) {
          error.code = 'INFERENCE_BACKEND_NOT_READY';
        } else if (/prompt plus new tokens must fit in --max-context/.test(stderr)) {
          error.code = 'PROMPT_TOO_LONG';
        }
        reject(error);
        return;
      }
      try {
        const events = stdout.split('\n').filter(line => line.trim().startsWith('{')).map(line => JSON.parse(line));
        const result = events.findLast(event => event.event === 'completion');
        if (!result || typeof result.text !== 'string') throw new Error('Native runner did not return generated text');
        resolve(trimAtStop(result.text, model.stopSequences ?? DEFAULT_STOPS));
      } catch (error) { reject(error); }
    });
  });
}
module.exports = { loadModels, createInference, trimAtStop };
