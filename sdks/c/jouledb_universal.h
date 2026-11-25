/*
 * jouledb_universal.h — Official Single-Header C99 / C++20 SDK for JouleDB v1.0.0
 * Dual-licensed under Apache-2.0 OR MIT. Zero external dependencies.
 */
#ifndef JOULEDB_UNIVERSAL_H
#define JOULEDB_UNIVERSAL_H

#include <stddef.h>
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
  #include <winsock2.h>
  #include <ws2tcpip.h>
#else
  #include <arpa/inet.h>
  #include <netinet/in.h>
  #include <sys/socket.h>
  #include <unistd.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define JOULEDB_SDK_VERSION "1.0.0"

/* Sends an HTTP/1.1 POST JSON request to 127.0.0.1:`port` and writes response body into `out_buf`. */
static inline int jouledb_http_post_local(
    unsigned short port,
    const char *path,
    const char *json_body,
    char *out_buf,
    size_t out_cap
) {
    if (!path || !json_body || !out_buf || out_cap < 16) {
        return -1;
    }
    out_buf[0] = '\0';
#ifdef _WIN32
    SOCKET fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (fd == INVALID_SOCKET) return -2;
#else
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -2;
#endif
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    addr.sin_addr.s_addr = htonl(0x7f000001u); /* 127.0.0.1 */

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) != 0) {
#ifdef _WIN32
        closesocket(fd);
#else
        close(fd);
#endif
        return -3;
    }

    char req_hdr[1024];
    size_t body_len = strlen(json_body);
    int hdr_len = snprintf(
        req_hdr,
        sizeof(req_hdr),
        "POST %s HTTP/1.1\r\nHost: 127.0.0.1:%u\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: %zu\r\n\r\n",
        path,
        (unsigned int)port,
        body_len
    );
    if (hdr_len <= 0 || (size_t)hdr_len >= sizeof(req_hdr)) {
#ifdef _WIN32
        closesocket(fd);
#else
        close(fd);
#endif
        return -4;
    }

    (void)send(fd, req_hdr, (size_t)hdr_len, 0);
    if (body_len > 0) {
        (void)send(fd, json_body, body_len, 0);
    }

    size_t used = 0;
    while (used + 1 < out_cap) {
        int n = (int)recv(fd, out_buf + used, out_cap - 1 - used, 0);
        if (n <= 0) break;
        used += (size_t)n;
    }
    out_buf[used] = '\0';
#ifdef _WIN32
    closesocket(fd);
#else
    close(fd);
#endif
    const char *sep = strstr(out_buf, "\r\n\r\n");
    if (sep) {
        size_t body_off = (size_t)(sep + 4 - out_buf);
        memmove(out_buf, out_buf + body_off, used - body_off + 1);
    }
    return 0;
}

#ifdef __cplusplus
}
#endif

#endif /* JOULEDB_UNIVERSAL_H */
