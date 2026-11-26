# `@openie/jouledb` (`v0.8.0`)

Official zero-dependency TypeScript / JavaScript / Deno / Bun / Edge SDK for **[JouleDB v0.8.0](https://jouledb.org)**.

```js
import { connect } from "@openie/jouledb";

const db = connect("http://127.0.0.1:8080");
const res = await db.query("SELECT region, SUM(revenue) FROM sales GROUP BY region");
console.log(res.rows, res.energy_joules);

const cert = await db.migrateCertify();
const trust = await db.trustAudit();
```
