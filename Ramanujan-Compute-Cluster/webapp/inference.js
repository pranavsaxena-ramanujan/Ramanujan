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
    configureChat(model);
  }
  return models;
}

// Turn templates per chat format. No BOS text: the GGUF tokenizer adds it when the model wants it.
const CHAT_FORMATS = {
  chatml: { user: '<|im_start|>user\n{content}<|im_end|>\n', assistant: '<|im_start|>assistant\n{content}<|im_end|>\n',
    generation: '<|im_start|>assistant\n', stops: ['<|im_end|>', '<|endoftext|>'] },
  llama3: { user: '<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>',
    assistant: '<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>',
    generation: '<|start_header_id|>assistant<|end_header_id|>\n\n', stops: ['<|eot_id|>', '<|end_of_text|>'] },
  gemma: { user: '<start_of_turn>user\n{content}<end_of_turn>\n', assistant: '<start_of_turn>model\n{content}<end_of_turn>\n',
    generation: '<start_of_turn>model\n', stops: ['<end_of_turn>', '<eos>'] },
  phi3: { user: '<|user|>\n{content}<|end|>\n', assistant: '<|assistant|>\n{content}<|end|>\n',
    generation: '<|assistant|>\n', stops: ['<|end|>', '<|endoftext|>'] },
  zephyr: { user: '<|user|>\n{content}</s>\n', assistant: '<|assistant|>\n{content}</s>\n',
    generation: '<|assistant|>\n', stops: ['</s>', '<|user|>'] },
  mistral: { user: '[INST] {content} [/INST]', assistant: ' {content}</s>', generation: '', stops: ['</s>', '[INST]'] },
  plain: { user: 'User: {content}\n', assistant: 'Assistant: {content}\n', generation: 'Assistant:', stops: ['\nUser:', '</s>'] }
};

/** Guesses the chat format from a GGUF Jinja chat template (or a legacy promptTemplate). */
function detectChatFormat(template) {
  const value = typeof template === 'string' ? template : '';
  if (value.includes('<|im_start|>')) return 'chatml';
  if (value.includes('<|start_header_id|>')) return 'llama3';
  if (value.includes('<start_of_turn>')) return 'gemma';
  if (value.includes('<|end|>') && value.includes('<|assistant|>')) return 'phi3';
  if (value.includes('<|user|>')) return 'zephyr';
  if (value.includes('[INST]')) return 'mistral';
  return 'plain';
}

/** Generation prefix for models whose template has a thinking block: an empty one skips the reasoning. */
function defaultGenerationPrefix(format, template) {
  const base = CHAT_FORMATS[format].generation;
  return format === 'chatml' && typeof template === 'string' && template.includes('<think>') ? base + '<think>\n\n</think>\n\n' : base;
}

function configureChat(model) {
  if (model.chatFormat !== undefined && !Object.hasOwn(CHAT_FORMATS, model.chatFormat)) {
    throw new Error(`chatFormat must be one of ${Object.keys(CHAT_FORMATS).join(', ')}`);
  }
  if (model.generationPrefix !== undefined && typeof model.generationPrefix !== 'string') throw new Error('generationPrefix must be a string');
  if (!model.chatFormat && model.promptTemplate) {
    const format = detectChatFormat(model.promptTemplate);
    const user = CHAT_FORMATS[format].user.replace('{content}', '{question}');
    // A legacy single-turn template becomes a multi-turn format when it starts with that format's user turn.
    if (format !== 'plain' && model.promptTemplate.startsWith(user)) {
      model.chatFormat = format;
      model.generationPrefix ??= model.promptTemplate.slice(user.length);
    }
  }
  if (!model.chatFormat && !model.promptTemplate) model.chatFormat = 'plain';
  if (model.chatFormat) model.generationPrefix ??= CHAT_FORMATS[model.chatFormat].generation;
  return model;
}

/**
 * Renders a conversation as chat turns for the runner, which drops the oldest pairs that do not
 * fit the context window. `messages` alternate user/assistant and end with the user's question.
 */
function chatTurns(model, messages) {
  if (!model.chatFormat && model.promptTemplate) {
    return { turns: [model.promptTemplate.replace('{question}', () => messages.at(-1).content)], suffix: '' };
  }
  const format = CHAT_FORMATS[model.chatFormat || 'plain'];
  return { turns: messages.map(message => format[message.role].replace('{content}', () => message.content)),
    suffix: model.generationPrefix ?? format.generation };
}

function stopSequences(model) {
  return model.stopSequences ?? [...new Set([...(model.chatFormat ? CHAT_FORMATS[model.chatFormat].stops : []), ...DEFAULT_STOPS])];
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

function createInference({ runner, python = 'python3', capacityAware = false }) {
  return ({ model, messages, question, roomId, apiToken, publicUrl }) => new Promise((resolve, reject) => {
    const chat = chatTurns(model, messages ?? [{ role: 'user', content: question }]);
    const env = { RAMANUJAN_PORTAL_TOKEN: apiToken };
    for (const key of ['PATH', 'HOME', 'LANG', 'LC_ALL', 'PYTHONPATH', 'VIRTUAL_ENV', 'SYSTEMROOT']) {
      if (process.env[key]) env[key] = process.env[key];
    }
    const child = spawn(python, [runner, '--runtime', 'native', '--homelab', `${publicUrl}/api/clusters/${roomId}/homelab`,
      '--package', model.package, '--metadata', model.metadata, '--prompt-turns', '-', '--max-new-tokens', String(model.maxNewTokens ?? 128),
      '--max-context', String(model.maxContext ?? 1024), '--timeout', String(model.requestTimeout ?? 120),
      ...(capacityAware ? ['--capacity-aware'] : [])],
    { env, stdio: ['pipe', 'pipe', 'pipe'] });
    child.stdin.on('error', () => {});
    child.stdin.end(JSON.stringify(chat));
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
        const placement = events.findLast(event => event.event === 'capacity-placement');
        if (placement) console.log('LLM capacity placement:', JSON.stringify(placement.devices));
        const result = events.findLast(event => event.event === 'completion');
        if (!result || typeof result.text !== 'string') throw new Error('Native runner did not return generated text');
        const fit = events.findLast(event => event.event === 'prompt-fit');
        resolve({ text: trimAtStop(result.text, stopSequences(model)), droppedTurns: fit?.droppedTurns ?? 0 });
      } catch (error) { reject(error); }
    });
  });
}
module.exports = { loadModels, createInference, trimAtStop, CHAT_FORMATS, detectChatFormat, defaultGenerationPrefix, configureChat, chatTurns };
