// node-postgres: simple + extended (parameterized) queries, named prepared statements.
const { Client } = require('pg');
(async () => {
  const c = new Client({ host: '127.0.0.1', port: +(process.env.JOULEDB_PG_PORT || 5433), user: 'postgres', database: 'postgres' });
  const res = {}; let ok = true;
  const check = async (n, f) => { try { await f(); res[n] = 'ok'; } catch (e) { ok = false; res[n] = 'FAIL ' + String(e.message).slice(0, 160); } };
  await c.connect();
  await check('ddl', async () => { await c.query('DROP TABLE IF EXISTS np_t'); await c.query('CREATE TABLE np_t (id INT, name TEXT)'); });
  await check('insert_params', () => c.query('INSERT INTO np_t VALUES ($1, $2)', [1, 'a']));
  await check('select_params', async () => { const r = await c.query('SELECT name FROM np_t WHERE id = $1', [1]); if (r.rows[0].name !== 'a') throw new Error(JSON.stringify(r.rows)); });
  await check('named_prepared', async () => { for (let i = 0; i < 3; i++) { const r = await c.query({ name: 'q1', text: 'SELECT id FROM np_t WHERE name = $1', values: ['a'] }); if (Number(r.rows[0].id) !== 1) throw new Error(JSON.stringify(r.rows)); } });
  await check('multi_statement', () => c.query('SELECT 1; SELECT 2'));
  await c.end();
  for (const [k, v] of Object.entries(res)) console.log(`node-pg ${k}: ${v}`);
  process.exit(ok ? 0 : 1);
})().catch(e => { console.log('node-pg connect: FAIL ' + e.message); process.exit(1); });
