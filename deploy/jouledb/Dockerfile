# Multi-stage, multi-arch (`linux/amd64` + `linux/arm64`) production image for JouleDB.
#
# Supports:
#   - Bare CPU / SIMD (`POPCNT` RaBitQ + 1.58-bit Ternary + HDC)
#   - Heterogeneous GPU offload via Vulkan (`--device /dev/dri` or NVIDIA Container Toolkit)
#   - Multi-node Heterogeneous Energy Fabric (`JOULEDB_FABRIC_NODE_CLASS`, `JOULEDB_GRID_FACTOR`)
#
# Build:
#   docker build -f deploy/jouledb/Dockerfile -t ghcr.io/joulesperbit/jouledb:0.1.0 .
# Multi-arch buildx:
#   docker buildx build --platform linux/amd64,linux/arm64 -f deploy/jouledb/Dockerfile \
#     -t ghcr.io/joulesperbit/jouledb:0.1.0 .

FROM rust:1.94-bookworm AS builder
WORKDIR /src

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    cmake \
    clang \
    libvulkan1 \
    && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock ./
COPY crates/ ./crates/

RUN cargo build --release \
    -p joule-db-server \
    --bin jouledb

# ── Stage 2: Build-time smoke gate ───────────────────────────────────
FROM builder AS smoke-test
RUN /src/target/release/jouledb --version

# ── Stage 3: Minimal hardened runtime ────────────────────────────────
FROM debian:bookworm-slim AS runtime

LABEL org.opencontainers.image.title="jouledb"
LABEL org.opencontainers.image.description="JouleDB — Energy-First Heterogeneous Fabric Database for Agentic, Embodied, HMI & Cloud Computing"
LABEL org.opencontainers.image.licenses="Apache-2.0 OR MIT"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl3 \
    libvulkan1 \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 jouledb \
    && useradd --system --uid 10001 --gid jouledb --create-home jouledb \
    && mkdir -p /var/lib/jouledb/data /var/lib/jouledb/ledger \
    && chown -R jouledb:jouledb /var/lib/jouledb

COPY --from=smoke-test /src/target/release/jouledb /usr/local/bin/jouledb

USER jouledb
WORKDIR /var/lib/jouledb

ENV JOULEDB_BIND_HOST=0.0.0.0 \
    JOULEDB_LEDGER_ENABLED=1 \
    JOULEDB_LEDGER_DIR=/var/lib/jouledb/ledger \
    JOULEDB_FABRIC_NODE_CLASS=cloud_cxl \
    JOULEDB_GRID_FACTOR=120.0

# 8080: HTTP + MCP 2026 + Flight SQL + Fabric + Embodied/Living/SigQL
# 8081: Binary KV TCP
# 8082: PostgreSQL Wire (psql)
# 8083: JWP (Joule Wire Protocol)
# 9001: Raft Consensus
EXPOSE 8080 8081 8082 8083 9001

VOLUME ["/var/lib/jouledb/data", "/var/lib/jouledb/ledger"]

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/health/ready || exit 1

ENTRYPOINT ["/usr/local/bin/jouledb"]
CMD ["--port", "8080", "--data", "/var/lib/jouledb/data"]
