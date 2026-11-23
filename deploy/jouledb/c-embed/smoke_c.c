#include "jouledb.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    const char *db_path = (argc > 1) ? argv[1] : "/tmp/jouledb_c_embed_smoke";
    JouleDBHandle *db = joule_db_open(db_path);
    assert(db != NULL && "joule_db_open must succeed");

    char *err = NULL;
    assert(joule_db_exec(db, "DROP TABLE IF EXISTS actuators", &err) == JOULE_OK);
    if (joule_db_exec(db, "CREATE TABLE actuators (id INT PRIMARY KEY, joint TEXT, torque_nm REAL)", &err) != JOULE_OK) {
        fprintf(stderr, "CREATE TABLE failed: %s\n", err ? err : "unknown");
        return 1;
    }
    assert(joule_db_exec(db, "INSERT INTO actuators VALUES (1, 'shoulder_pitch', 14.25)", &err) == JOULE_OK);
    assert(joule_db_exec(db, "INSERT INTO actuators VALUES (2, 'wrist_yaw', 3.75)", &err) == JOULE_OK);

    JouleDBStmt *stmt = joule_db_prepare(db, "SELECT id, joint, torque_nm FROM actuators WHERE id = $1");
    assert(stmt != NULL);
    assert(joule_db_bind_int(stmt, 1, 1) == JOULE_OK);
    assert(joule_db_step(stmt) == JOULE_ROW);
    assert(joule_db_column_int(stmt, 0) == 1);
    const char *joint = joule_db_column_text(stmt, 1);
    assert(joint != NULL && strcmp(joint, "shoulder_pitch") == 0);
    double torque = joule_db_column_double(stmt, 2);
    assert(torque > 14.2 && torque < 14.3);
    assert(joule_db_step(stmt) == JOULE_DONE);
    assert(joule_db_finalize(stmt) == JOULE_OK);

    joule_db_close(db);
    printf("C99_EMBED_SMOKE_OK joint=%s torque=%.2f\n", "shoulder_pitch", torque);
    return 0;
}
