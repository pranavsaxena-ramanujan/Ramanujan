'use strict';
const fs = require('node:fs');
const path = require('node:path');

class SqlStore {
  constructor(pool) { this.pool = pool; }
  async one(sql, args) {
    const [rows] = await this.pool.execute(sql, args);
    return rows[0] || null;
  }
  ownerByEmail(email) { return this.one('SELECT * FROM cluster_owner WHERE email = ?', [email]); }
  createOwner(o) {
    return this.pool.execute('INSERT INTO cluster_owner (id,email,password_hash) VALUES (?,?,?)', [o.id, o.email, o.passwordHash]);
  }
  async createSession(hash, ownerId, expiresAt) {
    await this.pool.execute('DELETE FROM cluster_session WHERE expires_at <= CURRENT_TIMESTAMP');
    await this.pool.execute('INSERT INTO cluster_session VALUES (?,?,?)', [hash, ownerId, expiresAt]);
  }
  session(hash) {
    return this.one('SELECT owner_id FROM cluster_session WHERE token_hash = ? AND expires_at > CURRENT_TIMESTAMP', [hash]);
  }
  deleteSession(hash) { return this.pool.execute('DELETE FROM cluster_session WHERE token_hash = ?', [hash]); }
  createCluster(r) {
    return this.pool.execute('INSERT INTO compute_cluster (id,room_id,owner_id,name,join_secret_hash) VALUES (?,?,?,?,?)',
      [r.id, r.roomId, r.ownerId, r.name, r.secretHash]);
  }
  cluster(roomId) { return this.one('SELECT * FROM compute_cluster WHERE room_id = ?', [roomId]); }
  async clusters(ownerId) {
    const [rows] = await this.pool.execute('SELECT id,room_id,name,created_at FROM compute_cluster WHERE owner_id = ? ORDER BY created_at DESC', [ownerId]);
    return rows;
  }
  async devices(clusterId) {
    const [rows] = await this.pool.execute('SELECT id,name,platform,last_seen,revoked FROM cluster_device WHERE cluster_id = ?', [clusterId]);
    return rows;
  }
  createDevice(d) {
    return this.pool.execute('INSERT INTO cluster_device (id,cluster_id,token_hash,name,platform) VALUES (?,?,?,?,?)',
      [d.id, d.clusterId, d.tokenHash, d.name, d.platform]);
  }
  device(hash) { return this.one('SELECT id,cluster_id FROM cluster_device WHERE token_hash = ? AND revoked = FALSE', [hash]); }
  touchDevice(id) { return this.pool.execute('UPDATE cluster_device SET last_seen = CURRENT_TIMESTAMP WHERE id = ?', [id]); }
  async revokeDevice(id, clusterId) {
    const [result] = await this.pool.execute('UPDATE cluster_device SET revoked = TRUE WHERE id = ? AND cluster_id = ?', [id, clusterId]);
    return result.affectedRows > 0;
  }
  createJob(id, clusterId) {
    return this.pool.execute("INSERT INTO cluster_job (id,cluster_id,status) VALUES (?,?,'RUNNING')", [id, clusterId]);
  }
  finishJob(id, status, result) {
    return this.pool.execute('UPDATE cluster_job SET status = ?,result_json = ? WHERE id = ?', [status, JSON.stringify(result), id]);
  }
  async jobs(clusterId) {
    const [rows] = await this.pool.execute('SELECT id,status,result_json,created_at FROM cluster_job WHERE cluster_id = ? ORDER BY created_at DESC LIMIT 50', [clusterId]);
    return rows;
  }
}

function createStore(env = process.env) {
  if (env.LOCAL_DB) {
    if (env.NODE_ENV === 'production') throw new Error('Production requires MySQL; LOCAL_DB is for local development only');
    const { DatabaseSync } = require('node:sqlite');
    if (env.LOCAL_DB !== ':memory:') fs.mkdirSync(path.dirname(path.resolve(env.LOCAL_DB)), { recursive: true });
    const db = new DatabaseSync(env.LOCAL_DB);
    db.exec('PRAGMA foreign_keys = ON; PRAGMA journal_mode = WAL;');
    db.exec(fs.readFileSync(path.join(__dirname, 'schema.sql'), 'utf8'));
    if (env.LOCAL_DB !== ':memory:') fs.chmodSync(env.LOCAL_DB, 0o600);
    return new SqlStore({
      execute: async (sql, args = []) => {
        const stmt = db.prepare(sql);
        const normalized = args.map(value => value instanceof Date ? value.toISOString().slice(0, 19).replace('T', ' ') : value);
        if (/^\s*SELECT/i.test(sql)) return [stmt.all(...normalized)];
        const result = stmt.run(...normalized);
        return [{ affectedRows: result.changes }];
      },
      end: async () => db.close()
    });
  }
  for (const key of ['DB_HOST', 'DB_USER', 'DB_NAME']) if (!env[key]) throw new Error(`${key} is required`);
  const pool = require('mysql2/promise').createPool({
    host: env.DB_HOST, port: Number(env.DB_PORT || 3306), user: env.DB_USER,
    password: env.DB_PASSWORD, database: env.DB_NAME, connectionLimit: 10, timezone: 'Z',
    ssl: env.DB_TLS === 'true' ? { rejectUnauthorized: true } : undefined
  });
  return new SqlStore(pool);
}
module.exports = { SqlStore, createStore };
