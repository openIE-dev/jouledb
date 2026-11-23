/* JouleDB C99 Embedded Header (`libjoule_db_c`)
 *
 * Embed JouleDB directly inside C, C++, Rust, Zig, Python ctypes, ROS2 nodes,
 * or embedded Linux/RTOS firmwares with zero external daemon.
 */
#ifndef JOULEDB_H
#define JOULEDB_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define JOULE_OK    0
#define JOULE_ERROR 1
#define JOULE_ROW   100
#define JOULE_DONE  101

typedef struct JouleDBHandle JouleDBHandle;
typedef struct JouleDBStmt   JouleDBStmt;

JouleDBHandle *joule_db_open(const char *path);
void           joule_db_close(JouleDBHandle *handle);
int            joule_db_exec(JouleDBHandle *handle, const char *sql, char **err_msg);
const char    *joule_db_errmsg(JouleDBHandle *handle);

JouleDBStmt   *joule_db_prepare(JouleDBHandle *handle, const char *sql);
int            joule_db_bind_int(JouleDBStmt *stmt, int index, int64_t value);
int            joule_db_bind_double(JouleDBStmt *stmt, int index, double value);
int            joule_db_bind_text(JouleDBStmt *stmt, int index, const char *value);
int            joule_db_bind_null(JouleDBStmt *stmt, int index);

int            joule_db_step(JouleDBStmt *stmt);
int            joule_db_column_count(JouleDBStmt *stmt);
const char    *joule_db_column_name(JouleDBStmt *stmt, int col);
int64_t        joule_db_column_int(JouleDBStmt *stmt, int col);
double         joule_db_column_double(JouleDBStmt *stmt, int col);
const char    *joule_db_column_text(JouleDBStmt *stmt, int col);
int            joule_db_column_is_null(JouleDBStmt *stmt, int col);
int            joule_db_reset(JouleDBStmt *stmt);
int            joule_db_finalize(JouleDBStmt *stmt);

#ifdef __cplusplus
}
#endif

#endif /* JOULEDB_H */
