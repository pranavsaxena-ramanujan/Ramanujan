'use strict';
const crypto = require('node:crypto');
const scrypt = require('node:util').promisify(crypto.scrypt);
const token = () => crypto.randomBytes(32).toString('base64url');
const digest = value => crypto.createHash('sha256').update(value).digest('hex');
async function hashSecret(secret) {
  const salt = crypto.randomBytes(16).toString('hex');
  return `${salt}:${(await scrypt(secret, salt, 64)).toString('hex')}`;
}
async function verifySecret(secret, stored) {
  const [salt, hex] = stored.split(':');
  const expected = Buffer.from(hex, 'hex');
  const actual = await scrypt(secret, salt, 64);
  return expected.length === actual.length && crypto.timingSafeEqual(expected, actual);
}
class HttpError extends Error {
  constructor(status, message) { super(message); this.status = status; }
}
function text(value, name, max = 120) {
  if (typeof value !== 'string' || !value.trim() || value.length > max) throw new HttpError(400, `${name} is required (maximum ${max} characters)`);
  return value.trim();
}
function sessionToken(req) {
  const header = req.get('authorization');
  if (header) return header.startsWith('Bearer ') ? header.slice(7) : '';
  const match = (req.get('cookie') || '').match(/(?:^|;\s*)cluster_session=([A-Za-z0-9_-]+)/);
  return match ? match[1] : '';
}
function requireOrigin(req, origin) {
  const supplied = req.get('origin');
  if (supplied && supplied !== origin) throw new HttpError(403, 'Cross-origin request rejected');
  if (req.get('cookie') && !req.get('authorization') && !supplied && !['GET', 'HEAD'].includes(req.method)) {
    throw new HttpError(403, 'Origin header required for cookie-authenticated writes');
  }
}
module.exports = { token, digest, hashSecret, verifySecret, HttpError, text, sessionToken, requireOrigin };
