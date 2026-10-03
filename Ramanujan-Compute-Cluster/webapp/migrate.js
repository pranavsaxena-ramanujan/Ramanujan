'use strict';
const fs = require('node:fs/promises');
const path = require('node:path');
const { createStore } = require('./store');
async function main() {
  const store = createStore();
  try {
    const statements = (await fs.readFile(path.join(__dirname, 'schema.sql'), 'utf8')).split(';').filter(sql => sql.trim());
    for (const statement of statements) await store.pool.execute(statement);
    console.log('Cluster portal tables are ready.');
  } finally { await store.pool.end(); }
}
main().catch(error => { console.error('Migration failed:', error.code || error.name); process.exitCode = 1; });
