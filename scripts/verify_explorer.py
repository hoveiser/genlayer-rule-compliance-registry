"""Independently verify the studionet evidence against the public explorer API.

Nothing here reads the local RPC or any cached state: every fact is fetched from
https://explorer-studio.genlayer.com, which indexes the chain separately from the
nodes that ran the consensus. It exists so the README's claims are checkable by a
third party, and so a failure is reported rather than assumed away. The explorer's
HTML page is an empty client-rendered shell, so this uses the JSON API with a real
User-Agent header, never the HTML.

What it checks
--------------
* every transaction hash from the run resolves on the explorer and reports
  ``status == FINALIZED``;
* the contract address resolves as ``type == CONTRACT``;
* the on-chain source really begins with the pinned-runner dependency header, and
  the ref is a real content hash (not :test / :latest / unversioned), i.e. the pin
  is what the chain stored, not just what the local file says;
* the deployed source matches this repo's contract file byte-for-byte (modulo line
  endings), so the README describes the contract that actually ran.

Usage::

    .venv/bin/python scripts/verify_explorer.py --contract 0x...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://explorer-studio.genlayer.com"
EXPLORER_TX = f"{BASE}/tx"
EXPLORER_ADDRESS = f"{BASE}/address"
PIN_PREFIX = '# { "Depends": "py-genlayer:'
BANNED_REFS = {"test", "latest", "dev", "main"}

# A 200 HTML shell is what the SPA serves for *unknown* routes, so the JSON API is
# the only honest way to confirm a record exists.
HEADERS = {
    "User-Agent": "genlayer-rule-compliance-registry-evidence-check",
    "Accept": "application/json",
}


def deployed_pin_ref(source: str) -> str:
    """The runner ref the chain actually stored, or '' when there is none.

    Returns '' for :test/:latest too, because a tag is not a pin.
    """
    first_line = source.splitlines()[0] if source else ""
    if not first_line.startswith(PIN_PREFIX):
        return ""
    ref = first_line.split("py-genlayer:", 1)[1].strip().rstrip('"} ').strip()
    if ref in BANNED_REFS or len(ref) < 40 or not ref.isalnum():
        return ""
    return ref


def get_json(url: str, timeout: int = 90, attempts: int = 4):
    """GET an explorer JSON record, retrying a transient 5xx.

    The explorer sits behind Cloudflare and can answer 502/503 for records that
    are indexed a second later, so a single failure is retried before it is
    treated as a problem. A genuinely absent record still fails with 404.
    """
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code < 500:
                raise
        except Exception as exc:  # noqa: BLE001 - connection resets etc.
            last_error = exc
        time.sleep(2.0 * (attempt + 1))
    raise last_error


def check_transaction(tx_hash: str) -> dict:
    payload = get_json(f"{BASE}/api/transactions/{tx_hash}")
    tx = payload.get("transaction") or {}
    record = {
        "tx_hash": tx_hash,
        "explorer_url": f"{EXPLORER_TX}/{tx_hash}",
        "found": bool(tx.get("hash")),
        "status": tx.get("status"),
        "from_address": tx.get("from_address"),
        "to_address": tx.get("to_address"),
        "value": int(tx.get("value") or 0),
        "created_at": tx.get("created_at"),
        "num_of_initial_validators": tx.get("num_of_initial_validators"),
        "result": (
            tx.get("consensus_data", {}).get("result_name")
            if isinstance(tx.get("consensus_data"), dict)
            else None
        ),
    }
    if not tx:
        record["explorer_payload_keys"] = (
            sorted(payload.keys()) if isinstance(payload, dict) else str(type(payload))
        )
    return record


def check_address(address: str, local_source: Path | None = None) -> dict:
    payload = get_json(f"{BASE}/api/address/{address}")
    source = payload.get("contract_code") or ""
    first_line = source.splitlines()[0] if source else ""
    pin = deployed_pin_ref(source)
    result = {
        "address": address,
        "explorer_url": f"{EXPLORER_ADDRESS}/{address}",
        "explorer_type": payload.get("type"),
        "tx_count": payload.get("tx_count"),
        "deployed_source_first_line": first_line[:120],
        "deployed_runner_pin": pin,
        "deployed_source_pin_is_content_hash": bool(pin),
        "pin_header_present_on_chain": bool(pin),
        "creator": (
            (payload.get("creator_info") or {}).get("address")
            if isinstance(payload.get("creator_info"), dict)
            else None
        ),
    }
    if local_source is not None and local_source.is_file() and source:
        deployed_bytes = source.encode("utf-8")
        local_bytes = local_source.read_bytes()
        result["local_source_file"] = str(local_source)
        result["deployed_source_equals_local_bytes"] = local_bytes == deployed_bytes
        result["deployed_source_equals_local_modulo_line_endings"] = (
            local_bytes.replace(b"\r\n", b"\n")
            == deployed_bytes.replace(b"\r\n", b"\n")
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", default=str(ROOT / "evidence" / "studionet-e2e.json"))
    parser.add_argument("--contract", required=True)
    parser.add_argument(
        "--out",
        default=str(ROOT / "evidence" / "explorer-verification.json"),
        help="where to write the report; a second file keeps a second run separate",
    )
    parser.add_argument(
        "--local-source",
        default=str(ROOT / "contracts" / "contract.py"),
        help="the repo file that claims to be the deployed contract's source",
    )
    args = parser.parse_args()

    evidence_path = Path(args.evidence)
    evidence = (
        json.loads(evidence_path.read_text(encoding="utf-8"))
        if evidence_path.is_file()
        else {"transactions": []}
    )

    report: dict = {"explorer_base": BASE, "transactions": [], "addresses": [], "problems": []}

    hashes = [e["tx_hash"] for e in evidence.get("transactions", []) if e.get("tx_hash")]
    print(f"[explorer] verifying {len(hashes)} transaction hashes from {evidence_path.name}")
    for tx_hash in hashes:
        try:
            record = check_transaction(tx_hash)
        except urllib.error.HTTPError as exc:
            record = {"tx_hash": tx_hash, "found": False, "http_error": exc.code}
            report["problems"].append(f"{tx_hash}: explorer returned HTTP {exc.code}")
        except Exception as exc:  # noqa: BLE001 - report, never hide
            record = {"tx_hash": tx_hash, "found": False, "error": str(exc)}
            report["problems"].append(f"{tx_hash}: {exc}")
        report["transactions"].append(record)
        step = next(
            (e.get("step") for e in evidence.get("transactions", []) if e.get("tx_hash") == tx_hash),
            "?",
        )
        status = record.get("status") or "NOT-FOUND"
        print(f"  {step:<24} {status:<12} value={record.get('value', 0)} {tx_hash}")
        if record.get("status") != "FINALIZED":
            report["problems"].append(
                f"{step} {tx_hash}: explorer status is {record.get('status')!r}, not FINALIZED"
            )

    local_source = Path(args.local_source)
    print(f"[explorer] verifying contract address {args.contract}")
    try:
        record = check_address(args.contract, local_source)
    except Exception as exc:  # noqa: BLE001
        record = {"address": args.contract, "error": str(exc)}
        report["problems"].append(f"contract {args.contract}: {exc}")
    report["addresses"].append({"role": "contract", **record})
    print(
        f"  contract   {record.get('explorer_type')} tx_count={record.get('tx_count')} "
        f"pin={str(record.get('deployed_runner_pin'))[:16]}.. {args.contract}"
    )
    if "deployed_source_equals_local_bytes" in record:
        exact = record["deployed_source_equals_local_bytes"]
        loose = record["deployed_source_equals_local_modulo_line_endings"]
        print(
            f"  {'':<10} deployed source vs {Path(record['local_source_file']).name}: "
            f"bytes={exact}, modulo line endings={loose}"
        )
        if not exact and not loose:
            report["problems"].append(
                f"contract {args.contract}: the chain's source is not this repo's "
                f"{record['local_source_file']}; the README would describe a contract "
                "that is not in the tree"
            )
    if record.get("explorer_type") != "CONTRACT":
        report["problems"].append(
            f"contract {args.contract}: explorer type is {record.get('explorer_type')!r}"
        )
    if not record.get("deployed_source_pin_is_content_hash"):
        report["problems"].append(
            f"contract {args.contract}: on-chain source has no pinned content-hash runner header"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    if report["problems"]:
        print("\n[explorer] PROBLEMS FOUND:")
        for problem in report["problems"]:
            print(f"  - {problem}")
        print(f"\n[explorer] report written to {out}")
        return 1
    print(f"\n[explorer] all {len(hashes)} transactions FINALIZED, contract verified as pinned.")
    print(f"[explorer] report written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
