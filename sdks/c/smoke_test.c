#include "jouledb_universal.h"
#include <stdlib.h>

int main(void) {
    if (strcmp(JOULEDB_SDK_VERSION, "1.0.0") != 0) {
        return 1;
    }
    const char *port_env = getenv("JOULEDB_PORT");
    if (port_env && port_env[0] != '\0') {
        unsigned short port = (unsigned short)atoi(port_env);
        char buf[65536];
        int rc = jouledb_http_post_local(port, "/api/v1/query", "{\"sql\":\"SELECT 100 AS v\"}", buf, sizeof(buf));
        if (rc != 0 || strstr(buf, "100") == NULL) {
            return 2;
        }
    }
    return 0;
}
