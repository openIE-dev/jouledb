# jouledb server image. Build: docker build -t jouledb .
# Run:  docker run -e JOULEDB_JWT_SECRET=change-me -p 8080-8083:8080-8083 -v jdb:/data jouledb
# Ports: 8080 HTTP, 8081 TCP, 8082 Postgres wire (password = JOULEDB_JWT_SECRET), 8083 JWP.
# The container binds 0.0.0.0, so auth is always on; without JOULEDB_JWT_SECRET
# every client is refused.
FROM rust:1.94-bookworm AS build
WORKDIR /src
COPY . .
RUN cargo build --release --locked -p joule-db-server --bin jouledb \
 && cp target/release/jouledb /jouledb

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/* && useradd -r -u 10001 -d /data jouledb && mkdir /data && chown jouledb /data
COPY --from=build /jouledb /usr/local/bin/jouledb
USER jouledb
ENV JOULEDB_BIND_HOST=0.0.0.0
VOLUME /data
EXPOSE 8080 8081 8082 8083
HEALTHCHECK CMD curl -fsS http://127.0.0.1:8080/health || exit 1
ENTRYPOINT ["jouledb", "-p", "8080", "-d", "/data"]
