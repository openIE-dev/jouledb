"""JouleDB Python SDK.

    import jouledb
    db = jouledb.connect("http://localhost:8080")
    r = db.query("SELECT region, SUM(v) FROM t GROUP BY region")
    r.arrow      # pyarrow.Table
    r.to_pandas()
    r.to_polars()  # zero-copy from Arrow
    r.receipt    # {'energy_joules': ..., 'device_target': 'cpu', ...}

Energy is a receipt, never a refusal: every query runs and reports what
it cost.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pyarrow as pa

from ._native import __version__ as _native_version
from ._native import decode as _decode

__all__ = ["connect", "Connection", "Result", "JouleDBError", "__version__"]
__version__ = "0.2.2"


class JouleDBError(RuntimeError):
    """Server-side or transport error."""


def _column(values: List[Any]) -> pa.Array:
    try:
        return pa.array(values)
    except (pa.ArrowInvalid, pa.ArrowTypeError):
        # mixed ints and floats -> float64; anything else -> string
        try:
            return pa.array(values, type=pa.float64())
        except (pa.ArrowInvalid, pa.ArrowTypeError):
            return pa.array([None if v is None else str(v) for v in values], type=pa.string())


@dataclass
class Result:
    """One query result: an Arrow table plus the energy receipt."""

    arrow: pa.Table
    receipt: Dict[str, Any] = field(default_factory=dict)
    affected_rows: Optional[int] = None

    def to_pandas(self):
        return self.arrow.to_pandas()

    def to_polars(self):
        import polars as pl

        return pl.from_arrow(self.arrow)

    def to_pylist(self) -> List[Dict[str, Any]]:
        return self.arrow.to_pylist()

    @property
    def num_rows(self) -> int:
        return self.arrow.num_rows

    def __len__(self) -> int:
        return self.arrow.num_rows


class Connection:
    """HTTP connection to a JouleDB server (`POST /api/v1/query`)."""

    def __init__(self, url: str = "http://localhost:8080", token: Optional[str] = None, timeout: float = 60.0):
        self.url = url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def query(self, sql: str) -> Result:
        body = json.dumps({"sql": sql}).encode()
        req = urllib.request.Request(self.url + "/api/v1/query", data=body, method="POST")
        req.add_header("content-type", "application/json")
        if self.token:
            req.add_header("authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                msg = json.loads(raw).get("error") or raw.decode(errors="replace")
            except ValueError:
                msg = raw.decode(errors="replace")
            raise JouleDBError(f"HTTP {e.code}: {msg}") from None
        except urllib.error.URLError as e:
            raise JouleDBError(f"cannot reach {self.url}: {e.reason}") from None
        try:
            cols, data, receipt, affected = _decode(raw)
        except RuntimeError as e:
            raise JouleDBError(str(e)) from None
        table = pa.table({c: _column(v) for c, v in zip(cols, data)}) if cols else pa.table({})
        return Result(arrow=table, receipt=receipt, affected_rows=affected)

    execute = query

    def _json_request(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        body = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(self.url + path, data=body, method=method)
        if body is not None:
            req.add_header("content-type", "application/json")
        if self.token:
            req.add_header("authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                msg = json.loads(raw).get("error") or raw.decode(errors="replace")
            except ValueError:
                msg = raw.decode(errors="replace")
            raise JouleDBError(f"HTTP {e.code}: {msg}") from None
        except urllib.error.URLError as e:
            raise JouleDBError(f"cannot reach {self.url}: {e.reason}") from None

    def synapse_status(self) -> Dict[str, Any]:
        """Inspect JouleSynapse living tensors, GPU kernel, Titans, Engram, and loaded vocabs."""
        return self._json_request("GET", "/api/v1/synapse/status")

    def ward_ledger(self) -> Dict[str, Any]:
        """Inspect the 6-rail nanojoule JouleWard fabric energy ledger and live hardware telemetry."""
        return self._json_request("GET", "/api/v1/ward/ledger")

    def ward_mint_token(
        self,
        root_token_id: str,
        subject: str,
        caveats: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Mint an HMAC-SHA256 ThermoCap capability macaroon."""
        return self._json_request(
            "POST",
            "/api/v1/ward/tokens/mint",
            {
                "root_token_id": root_token_id,
                "subject": subject,
                "caveats": caveats or [],
            },
        )

    def ward_exec(self, exec_request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a confined command under JouleWard with Merkle memoization & fabric routing."""
        return self._json_request("POST", "/api/v1/ward/exec", exec_request)

    def embed(self, input_text: Any, model: str = "jouledb-cascade-v1") -> Dict[str, Any]:
        """Generate living embeddings via OpenAI-compatible POST /v1/embeddings."""
        return self._json_request("POST", "/v1/embeddings", {"model": model, "input": input_text})

    def chat(self, prompt: str, model: str = "jouledb-cascade-v1") -> Dict[str, Any]:
        """Run 4-tier thermodynamic chat completion via POST /v1/chat/completions."""
        return self._json_request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": prompt}]},
        )

    def messages(self, prompt: str, model: str = "jouledb-cascade-v1", max_tokens: int = 256) -> Dict[str, Any]:
        """Run Anthropic-compatible POST /v1/messages with inline 45 nJ JoulePulse guardrails."""
        return self._json_request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            },
        )

    def flow_execute(self, flow_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a crash-idempotent durable JouleFlow DAG via POST /api/v1/flow/execute."""
        return self._json_request("POST", "/api/v1/flow/execute", flow_spec)

    def decoder_generate(
        self,
        model_id: str,
        prompt: str,
        max_new_tokens: int = 24,
        temperature: float = 0.0,
        speculative_draft_k: int = 0,
    ) -> Dict[str, Any]:
        """Run autoregressive causal token generation over sub-1-bit LittleBit-2 bitplanes."""
        return self._json_request(
            "POST",
            "/api/v1/synapse/decoder/generate",
            {
                "model_id": model_id,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "speculative_draft_k": speculative_draft_k,
            },
        )

    def chat_structured(
        self,
        prompt: str,
        json_schema: Dict[str, Any],
        model: str = "jouledb-cascade-v1",
    ) -> Dict[str, Any]:
        """Run schema-constrained JSON chat completion via response_format."""
        return self._json_request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_schema", "json_schema": json_schema},
            },
        )

    def chat_tools(
        self,
        prompt: str,
        tools: List[Dict[str, Any]],
        model: str = "jouledb-cascade-v1",
    ) -> Dict[str, Any]:
        """Run tool-constrained chat completion returning schema-validated tool_calls."""
        return self._json_request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "tools": tools,
            },
        )

    def chat_stream(self, prompt: str, model: str = "jouledb-cascade-v1") -> List[Dict[str, Any]]:
        """Stream SSE chat.completion.chunk events with per-token x_jouledb_token_nj."""
        body = json.dumps(
            {
                "model": model,
                "stream": True,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode()
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=body, method="POST")
        req.add_header("content-type", "application/json")
        if self.token:
            req.add_header("authorization", f"Bearer {self.token}")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw_text = resp.read().decode("utf-8", errors="replace")
        events: List[Dict[str, Any]] = []
        for line in raw_text.splitlines():
            if line.startswith("data: "):
                data_str = line[6:].strip()
                if data_str and data_str != "[DONE]":
                    events.append(json.loads(data_str))
        return events

    def mcp_register_server(
        self,
        server_id: str,
        endpoint_url: str,
        discover_live: bool = True,
    ) -> Dict[str, Any]:
        """Register an external Streamable-HTTP MCP server on JouleDB's native MCP Host."""
        return self._json_request(
            "POST",
            "/api/v1/mcp/host/servers",
            {
                "server_id": server_id,
                "endpoint_url": endpoint_url,
                "discover_live": discover_live,
            },
        )

    def mcp_list_tools(self) -> Dict[str, Any]:
        """List all native and remote MCP tools registered on JouleDB's MCP Host."""
        return self._json_request("GET", "/api/v1/mcp/host/tools")

    def mcp_call_tool(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Invoke a native or remote MCP tool under JoulePulse + JouleWard."""
        return self._json_request(
            "POST",
            "/api/v1/mcp/host/call",
            {"tool_name": tool_name, "arguments": arguments or {}},
        )

    def agent_run(
        self,
        goal: str,
        max_turns: int = 2,
        commit_to_titans: bool = True,
    ) -> Dict[str, Any]:
        """Execute a server-side autonomous multi-turn SQL/MCP/Titans agent loop."""
        return self._json_request(
            "POST",
            "/api/v1/agent/run",
            {
                "goal": goal,
                "max_turns": max_turns,
                "commit_to_titans": commit_to_titans,
            },
        )

    def vision_encode(
        self,
        image_base64: str,
        model_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Encode an image (base64 PPM/BMP/image bytes) via sub-1-bit SigLIP/ViT bitplanes."""
        return self._json_request(
            "POST",
            "/api/v1/synapse/vision/encode",
            {"model_id": model_id, "image_base64": image_base64},
        )

    def audio_transcribe(
        self,
        audio_base64: str,
        model: str = "whisper-1",
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Transcribe base64 WAV audio via OpenAI-compatible POST /v1/audio/transcriptions."""
        return self._json_request(
            "POST",
            "/v1/audio/transcriptions",
            {"model": model, "audio_base64": audio_base64, "prompt": prompt},
        )

    def vision_zero_shot(
        self,
        model_id: str,
        image_base64: str,
        candidate_labels: List[str],
    ) -> Dict[str, Any]:
        """Score zero-shot SigLIP/CLIP vision-text candidate labels over sub-1-bit bitplanes."""
        return self._json_request(
            "POST",
            "/api/v1/synapse/vision/zero_shot",
            {
                "model_id": model_id,
                "image_base64": image_base64,
                "candidate_labels": candidate_labels,
            },
        )

    def decoder_generate_batch(
        self,
        model_id: str,
        prompts: List[str],
        max_new_tokens: int = 24,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Coalesce multiple prompts into Continuous In-Flight Multi-Request Batched Generation."""
        return self._json_request(
            "POST",
            "/api/v1/synapse/decoder/generate_batch",
            {
                "model_id": model_id,
                "prompts": prompts,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            },
        )

    def hf_export(
        self,
        model_id: str,
        output_dir: str,
        base_model_repo: Optional[str] = None,
        hf_repo_id: Optional[str] = None,
        license: Optional[str] = None,
        license_notice: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Export a loaded model as a CRC32-verified .jsyn Hugging Face Hub bundle directory."""
        return self._json_request(
            "POST",
            "/api/v1/synapse/hf/export",
            {
                "model_id": model_id,
                "output_dir": output_dir,
                "base_model_repo": base_model_repo,
                "hf_repo_id": hf_repo_id,
                "license": license,
                "license_notice": license_notice,
            },
        )

    def hf_pull(
        self,
        model_id: str,
        repo_id: Optional[str] = None,
        local_dir: Optional[str] = None,
        filename: Optional[str] = None,
        revision: str = "main",
        target_rank: int = 16,
        distill_svf: bool = True,
    ) -> Dict[str, Any]:
        """Pull a pre-factorized .jsyn, .safetensors (including MLX 1-bit & sharded), or .gguf checkpoint."""
        return self._json_request(
            "POST",
            "/api/v1/synapse/hf/pull",
            {
                "model_id": model_id,
                "repo_id": repo_id,
                "local_dir": local_dir,
                "filename": filename,
                "revision": revision,
                "target_rank": target_rank,
                "distill_svf": distill_svf,
            },
        )

    def hf_catalog(self) -> Dict[str, Any]:
        """List JouleDB's curated <30 GB permissive open-weight frontier model catalog."""
        return self._json_request("GET", "/api/v1/synapse/hf/catalog")

    def ingest_gguf(
        self,
        model_id: str,
        file_path: str,
        target_rank: int = 16,
        distill_svf: bool = True,
    ) -> Dict[str, Any]:
        """Compile a local .gguf checkpoint directly into sub-1-bit LittleBit-2 living bitplanes."""
        return self._json_request(
            "POST",
            "/api/v1/synapse/ingest/gguf",
            {
                "model_id": model_id,
                "file_path": file_path,
                "target_rank": target_rank,
                "distill_svf": distill_svf,
            },
        )

    def evict_model(self, model_id: str) -> Dict[str, Any]:
        """Evict a loaded JouleSynapse living-weights model from RAM and GPU bitplane cache."""
        return self._json_request(
            "POST",
            "/api/v1/synapse/models/evict",
            {"model_id": model_id},
        )

    def kube_apply(
        self,
        manifest: str,
        default_namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Apply single or multi-document (---) Kubernetes YAML/JSON manifests into JouleKube."""
        return self._json_request(
            "POST",
            "/api/v1/kube/apply",
            {"manifest": manifest, "default_namespace": default_namespace},
        )

    def kube_status(self) -> Dict[str, Any]:
        """Inspect JouleKube cluster status, resource counts, standby watts, and ThermoCap bindings."""
        return self._json_request("GET", "/api/v1/kube/status")

    def kube_route_inference(
        self,
        namespace: str,
        pool_name: str,
        prompt: str,
        max_new_tokens: int = 32,
        speculative_draft_k: int = 0,
    ) -> Dict[str, Any]:
        """Route a prompt through a Gateway API Inference Extension InferencePool in JouleKube."""
        return self._json_request(
            "POST",
            f"/api/v1/kube/inference-pools/{namespace}/{pool_name}/route",
            {
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "speculative_draft_k": speculative_draft_k,
            },
        )

    def kube_invoke_agent(
        self,
        namespace: str,
        agent_name: str,
        goal: str,
        max_turns: int = 3,
    ) -> Dict[str, Any]:
        """Invoke a declarative kagent.dev/v1alpha1 Agent under its SandboxClaim ThermoCap macaroon."""
        return self._json_request(
            "POST",
            f"/api/v1/kube/agents/{namespace}/{agent_name}/invoke",
            {"goal": goal, "max_turns": max_turns},
        )

    def close(self) -> None:  # stateless HTTP; kept for DB-API-style use
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def connect(url: str = "http://localhost:8080", token: Optional[str] = None, timeout: float = 60.0) -> Connection:
    return Connection(url, token=token, timeout=timeout)
