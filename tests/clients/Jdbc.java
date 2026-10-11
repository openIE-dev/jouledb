import java.sql.*;
// pgJDBC: connect (SET/SHOW on startup), PreparedStatement (server-side after 5 uses), batch, metadata.
public class Jdbc {
  static boolean ok = true;
  interface Check { void run() throws Exception; }
  static void check(String n, Check c) {
    try { c.run(); System.out.println("jdbc " + n + ": ok"); }
    catch (Exception e) { ok = false; System.out.println("jdbc " + n + ": FAIL " + String.valueOf(e.getMessage()).split("\n")[0]); }
  }
  public static void main(String[] a) throws Exception {
    String port = System.getenv().getOrDefault("JOULEDB_PG_PORT", "5433");
    Connection c = DriverManager.getConnection("jdbc:postgresql://127.0.0.1:" + port + "/postgres?sslmode=disable", "postgres", "");
    c.setAutoCommit(true);
    Statement s = c.createStatement();
    check("ddl", () -> { s.execute("DROP TABLE IF EXISTS jd_t"); s.execute("CREATE TABLE jd_t (id INT, name TEXT, x FLOAT)"); });
    check("prepared_insert", () -> { try (PreparedStatement p = c.prepareStatement("INSERT INTO jd_t VALUES (?, ?, ?)")) {
      for (int i = 1; i <= 6; i++) { p.setInt(1, i); p.setString(2, "n" + i); p.setDouble(3, i + 0.5); p.executeUpdate(); } } });
    check("batch", () -> { try (PreparedStatement p = c.prepareStatement("INSERT INTO jd_t VALUES (?, ?, ?)")) {
      p.setInt(1, 7); p.setString(2, "n7"); p.setDouble(3, 7.5); p.addBatch(); p.setInt(1, 8); p.setString(2, "n8"); p.setDouble(3, 8.5); p.addBatch(); p.executeBatch(); } });
    check("prepared_select", () -> { try (PreparedStatement p = c.prepareStatement("SELECT name, x FROM jd_t WHERE id = ?")) {
      for (int i = 0; i < 6; i++) { p.setInt(1, 3); ResultSet r = p.executeQuery(); if (!r.next() || !r.getString(1).equals("n3") || Math.abs(r.getDouble(2) - 3.5) > 1e-9) throw new Exception("bad row"); } } });
    check("aggregate", () -> { ResultSet r = s.executeQuery("SELECT count(*), sum(x) FROM jd_t"); r.next(); if (r.getLong(1) != 8) throw new Exception("count " + r.getLong(1)); });
    check("transaction", () -> { c.setAutoCommit(false); s.execute("INSERT INTO jd_t VALUES (9, 'n9', 9.5)"); c.commit(); c.setAutoCommit(true); });
    check("metadata_version", () -> { String v = c.getMetaData().getDatabaseProductVersion(); if (v == null || v.isEmpty()) throw new Exception("no version"); });
    c.close();
    System.exit(ok ? 0 : 1);
  }
}
