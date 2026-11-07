"""Anyone Can Dock Streamable HTTP client; no additional dependencies.

Transport follows MCP 2025-03-26. Submission is never automatically retried:
losing a response does not mean the remote job was not created.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import math
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DEFAULT_ENDPOINT = "https://anyone-can-dock-mcp.anyonecandock.workers.dev/mcp"
PROTOCOL_VERSION = "2025-03-26"
TERMINAL_STATUSES = {"completed", "failed", "cancelled", "canceled"}
MAX_RESPONSE_BYTES = 16 * 1024 * 1024


class MCPError(RuntimeError):
    """A remote or transport error, without including submitted structures/tokens."""


class SubmissionUncertain(MCPError):
    """The submission may have reached the server; do not blindly retry."""


def _connect_ipv4_first(address, timeout, source_address=None):
    """Avoid stale IPv6 routes delaying every RPC; retain an IPv6 fallback."""
    host, port = address
    deadline = time.monotonic() + timeout
    addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    addresses.sort(key=lambda item: item[0] != socket.AF_INET)
    last_error = None
    for family, socktype, proto, _, sockaddr in addresses:
        connection = socket.socket(family, socktype, proto)
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Connection deadline exceeded")
            connection.settimeout(min(3.0, remaining))
            if source_address:
                connection.bind(source_address)
            connection.connect(sockaddr)
            connection.settimeout(max(0.1, deadline - time.monotonic()))
            return connection
        except OSError as exc:
            last_error = exc
            connection.close()
    raise last_error or OSError("No addresses resolved for the docking service")


class _HTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._create_connection = _connect_ipv4_first


class _HTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, request):
        return self.do_open(_HTTPSConnection, request, context=self._context)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward the deployment's authentication to an unexpected endpoint.
        return None


def decode_message(body: str, request_id: int) -> dict:
    """Accept JSON or SSE and ignore unrelated notifications/messages."""
    try:
        messages = json.loads(body)
        messages = messages if isinstance(messages, list) else [messages]
    except json.JSONDecodeError:
        messages = []
        for event in body.replace("\r\n", "\n").split("\n\n"):
            data = "\n".join(line[5:].lstrip(" ") for line in event.splitlines()
                             if line.startswith("data:"))
            if data:
                try:
                    value = json.loads(data)
                except json.JSONDecodeError as exc:
                    raise MCPError("The MCP service returned malformed event data.") from exc
                messages.extend(value if isinstance(value, list) else [value])
    for message in messages:
        if isinstance(message, dict) and message.get("id") == request_id:
            if "error" in message:
                error = message["error"]
                raise MCPError(f"MCP request failed ({error.get('code')}): "
                               f"{error.get('message', 'unknown error')}")
            if isinstance(message.get("result"), dict):
                return message["result"]
    raise MCPError("The MCP response did not contain this request's result.")


def tool_data(result: dict) -> dict:
    if result.get("isError"):
        detail = " ".join(c.get("text", "") for c in result.get("content", [])
                          if c.get("type") == "text")
        raise MCPError(detail or "The docking service reported a tool error.")
    data = result.get("structuredContent")
    if isinstance(data, dict):
        return data
    for block in result.get("content", []):
        if block.get("type") == "text":
            try:
                data = json.loads(block["text"])
            except (json.JSONDecodeError, KeyError):
                continue
            if isinstance(data, dict):
                return data
    raise MCPError("The docking service did not return structured JSON data.")


class ACDMCPClient:
    def __init__(self, endpoint=None, token=None, timeout=20):
        self.endpoint = endpoint or os.environ.get("ACD_MCP_URL", DEFAULT_ENDPOINT)
        url = urllib.parse.urlsplit(self.endpoint)
        if url.scheme != "https" or not url.hostname or url.username or url.password:
            raise ValueError("ACD_MCP_URL must be an HTTPS endpoint without embedded credentials.")
        self.token = token if token is not None else os.environ.get("ACD_MCP_TOKEN", "")
        self.timeout = timeout
        self.session_id = None
        self.server_info = {}
        self.protocol_version = PROTOCOL_VERSION
        self._request_id = 0
        self._ready = False
        self._tools = None
        self._opener = urllib.request.build_opener(_NoRedirect, _HTTPSHandler)

    def _post(self, method, params=None, notify=False):
        self._request_id += 1
        rid = self._request_id
        message = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        if not notify:
            message["id"] = rid
        headers = {"User-Agent": "AnalogBuilder/0.1 (MCP docking client)",
                   "Content-Type": "application/json",
                   "Accept": "application/json, text/event-stream",
                   "MCP-Protocol-Version": self.protocol_version}
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(self.endpoint, data=json.dumps(message).encode(),
                                         headers=headers, method="POST")
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                session = response.headers.get("Mcp-Session-Id")
                if session:
                    self.session_id = session
                if notify:
                    return {}
                # SSE may remain open after its response. Stop once our result arrives.
                body = bytearray()
                deadline = time.monotonic() + self.timeout
                is_sse = "text/event-stream" in response.headers.get("Content-Type", "")
                if is_sse:
                    while time.monotonic() < deadline:
                        line = response.readline(MAX_RESPONSE_BYTES + 1)
                        if not line:
                            break
                        body.extend(line)
                        if len(body) > MAX_RESPONSE_BYTES:
                            raise MCPError("MCP response exceeds the size limit.")
                        if line.strip() == b"":
                            try:
                                return decode_message(body.decode("utf-8"), rid)
                            except MCPError as exc:
                                if "did not contain" not in str(exc):
                                    raise
                    else:
                        raise MCPError("The MCP response exceeded its time limit.")
                else:
                    body.extend(response.read(MAX_RESPONSE_BYTES + 1))
                if len(body) > MAX_RESPONSE_BYTES:
                    raise MCPError("MCP response exceeds the size limit.")
                return decode_message(body.decode("utf-8"), rid)
        except urllib.error.HTTPError as exc:
            raise MCPError(f"Docking service HTTP {exc.code}; check service availability "
                           "and deployment authentication.") from exc
        except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
            raise MCPError("Could not reach the docking service within the request timeout.") from exc

    def initialize(self):
        if self._ready:
            return
        data = self._post("initialize", {"protocolVersion": PROTOCOL_VERSION,
                          "capabilities": {}, "clientInfo": {"name": "AnalogBuilder", "version": "0.1"}})
        self.protocol_version = data.get("protocolVersion", PROTOCOL_VERSION)
        self.server_info = data.get("serverInfo", {})
        self._post("notifications/initialized", notify=True)
        self._ready = True

    def tools(self):
        self.initialize()
        if self._tools is None:
            found = {}
            params = {}
            while True:
                result = self._post("tools/list", params)
                found.update({t["name"]: t for t in result.get("tools", [])})
                cursor = result.get("nextCursor")
                if not cursor:
                    break
                params = {"cursor": cursor}
            self._tools = found
        return self._tools

    def call(self, name, arguments):
        if name not in self.tools():
            raise MCPError(f"The connected service does not expose {name}.")
        return tool_data(self._post("tools/call", {"name": name, "arguments": arguments}))

    def submit(self, payload):
        # Discovery failures happen before submission and are safe to correct/retry.
        if "submitDockingJob" not in self.tools():
            raise MCPError("The connected service does not expose submitDockingJob.")
        try:
            result = tool_data(self._post("tools/call", {
                "name": "submitDockingJob", "arguments": payload}))
            if not result.get("job_id"):
                raise MCPError("Submission returned no job ID.")
            return result
        except MCPError as exc:
            raise SubmissionUncertain(
                f"Submission was not confirmed: {exc} It may have created a job. "
                "Do not resubmit automatically; recover the job ID from the service."
            ) from exc

    def job(self, job_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,120}", job_id):
            raise ValueError("Invalid docking job ID.")
        return self.call("getDockingJob", {"job_id": job_id})


def safe_token(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip(".") or "ligand"


def build_payload(receptor_text, ligands, center, box, ph=7.4,
                  exhaustiveness=8, num_modes=10, protonation_mode="pkanet",
                  use_pubchem=True, receptor_name="receptor"):
    from rdkit import Chem
    if not any(line.startswith("ATOM  ") for line in receptor_text.splitlines()):
        raise ValueError("Receptor must be a PDB containing ATOM records.")
    if center is None or len(center) != 3 or not all(math.isfinite(float(v)) for v in center):
        raise ValueError("Resolve a docking centre or set its coordinates manually.")
    if len(box) != 3 or not all(math.isfinite(float(v)) and 0 < float(v) <= 40 for v in box):
        raise ValueError("Each box dimension must be greater than 0 and at most 40 Å.")
    if not 0 <= ph <= 14 or not 1 <= exhaustiveness <= 128 or not 1 <= num_modes <= 50:
        raise ValueError("Invalid pH, exhaustiveness, or pose count.")
    if protonation_mode not in {"pkanet", "neutral", "dimorphite"}:
        raise ValueError("Unsupported protonation mode.")
    if not ligands:
        raise ValueError("Select at least one ligand.")
    names = [str(row["name"]) for row in ligands]
    tokens = [safe_token(name) for name in names]
    if any(not name.strip() for name in names) or len(set(tokens)) != len(tokens):
        raise ValueError("Ligand names must be nonempty and unique after filename sanitisation.")
    for row in ligands:
        if Chem.MolFromSmiles(row["smiles"]) is None:
            raise ValueError(f"Invalid SMILES for {row['name']}.")
    # MCP accepts numeric dimensions directly; retain the user's fractional box.
    return {"receptor_pdb_text": receptor_text, "receptor_name": receptor_name,
            "ligands": [{"name": str(r["name"]), "smiles": r["smiles"]} for r in ligands],
            "center_mode": "manual", "grid": dict(zip(
                ("center_x", "center_y", "center_z", "size_x", "size_y", "size_z"),
                map(float, tuple(center) + tuple(box)))),
            "ph": float(ph), "exhaustiveness": int(exhaustiveness), "num_modes": int(num_modes),
            "protonation_mode": protonation_mode, "use_pubchem": bool(use_pubchem)}


def resolve_center(receptor_text, reference_text=None, mode="Protein centre", manual=None):
    if mode == "Manual":
        if manual is None or len(manual) != 3:
            raise ValueError("Enter all three manual centre coordinates.")
        return tuple(float(v) for v in manual)
    text = reference_text if mode == "Reference ligand" else receptor_text
    if not text:
        raise ValueError("No reference ligand is available; select a manual or protein centre.")
    points = []
    for line in text.splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            try:
                points.append(tuple(float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54))))
            except ValueError:
                continue
    if not points:
        raise ValueError("No usable coordinates are available to resolve the centre.")
    return tuple(sum(p[i] for p in points) / len(points) for i in range(3))


def payload_fingerprint(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_receipt(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2))
    os.replace(temporary, path)


def load_receipt(path):
    try:
        value = json.loads(Path(path).read_text())
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        raise MCPError("The saved docking receipt cannot be read; recover using its job ID.") from exc
    if not isinstance(value, dict):
        raise MCPError("The saved docking receipt has an invalid format.")
    return value


def result_rows(job):
    """Normalise per-ligand results; never promote a failed ligand's score."""
    rows = []
    for item in job.get("results", []):
        ok = item.get("status") == "ok"
        rows.append({"compound": item.get("name", ""), "status": item.get("status", "unknown"),
                     "top_BE_kcal_mol": item.get("top_score") if ok else None,
                     "n_poses": item.get("num_poses", 0) if ok else 0,
                     "selected_pose_rank": item.get("selected_pose_rank") if ok else None,
                     "selected_pose_RMSD_A": item.get("selected_pose_rmsd") if ok else None,
                     "error": item.get("error", "")})
    return rows


def public_url(value):
    """Only render server-provided HTTPS links, never arbitrary URL schemes."""
    try:
        url = urllib.parse.urlsplit(value or "")
        return value if url.scheme == "https" and url.hostname and not url.username else None
    except ValueError:
        return None
