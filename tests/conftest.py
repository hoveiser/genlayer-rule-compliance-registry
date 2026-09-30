# Direct-mode test harness for the rule-compliance registry (gltest 0.29.2).
#
# Facts verified against the ACTUALLY installed direct-mode API, not docs:
#   * direct_vm.sender drives gl.message.sender_address (an Address with .as_hex).
#   * The contract clock reads gl.message_raw["datetime"]; warp() does NOT refresh
#     that dict for the contract, so we set it directly (set_time), like sibling repos.
#   * mock_llm(pattern, json_string): _handle_llm_request auto-parses a JSON string
#     into a dict, so the contract's response parser must accept dict OR str.
#   * run_nondet_unsafe in direct mode runs ONLY leader_fn and captures the
#     validator; the validator is exercised explicitly via direct_vm.run_validator(),
#     and the runner bundle is resolved from the local cache (no network).
#   * spawn_sandbox is NOT isolated in direct mode; the contract falls back to the
#     identical inline deterministic evaluation, so verdicts are still ground truth.
import json
import re

import pytest

from gltest.direct.loader import create_address


TRANSLATE_PATTERN = "compliance-rule compiler"
SUBJECTIVE_PATTERN = "community moderator"

GEN_BASE_TS = "2026-01-01T00:00:00.000000Z"


def addr(seed):
    return create_address(seed)


def _advance_iso(base, seconds):
    """Return a fixed-width ISO stamp `seconds` after the base prefix."""
    import datetime

    dt = datetime.datetime.strptime(base[:19], "%Y-%m-%dT%H:%M:%S")
    dt = dt + datetime.timedelta(seconds=seconds)
    return dt.isoformat() + "Z"


class Controller:
    def __init__(self, vm, gl, contract):
        self.vm = vm
        self.gl = gl
        self.c = contract

    # ---- time -------------------------------------------------------------
    def set_time(self, ts):
        self.gl.message_raw["datetime"] = ts

    def advance(self, seconds):
        cur = self.gl.message_raw.get("datetime", GEN_BASE_TS)
        self.gl.message_raw["datetime"] = _advance_iso(cur, seconds)

    # ---- LLM mocks --------------------------------------------------------
    def mock_pipeline(self, checks=None, judgments=None, translate_garbage=False,
                      subjective_garbage=False):
        """Register LLM mocks in place so translate and subjective can coexist.

        gltest returns the FIRST regex-matching mock, so both distinct patterns
        must be present at once for a two-pass (translate + subjective) call.
        """
        self.vm._llm_mocks.clear()
        if translate_garbage:
            self.vm.mock_llm(TRANSLATE_PATTERN, "this is not json at all >>>")
        elif checks is not None:
            self.vm.mock_llm(TRANSLATE_PATTERN, json.dumps({"checks": checks}))
        if subjective_garbage:
            self.vm.mock_llm(SUBJECTIVE_PATTERN, "judgments?? not json")
        elif judgments is not None:
            self.vm.mock_llm(SUBJECTIVE_PATTERN, json.dumps({"judgments": judgments}))

    # ---- convenience rule builders ---------------------------------------
    @staticmethod
    def obj_expr(index, expression):
        return {"index": index, "kind": "objective", "expression": expression}

    @staticmethod
    def subj(index):
        return {"index": index, "kind": "subjective", "expression": ""}

    # ---- actions ----------------------------------------------------------
    def create_community(self, owner, name, rules, validity=3600):
        self.vm.sender = owner
        return self.c.create_community(name, rules, validity)

    def update_rules(self, sender, cid, rules):
        self.vm.sender = sender
        return self.c.update_rules(cid, rules)

    def set_moderator(self, sender, cid, moderator):
        self.vm.sender = sender
        return self.c.set_moderator(cid, moderator)

    def set_result_validity(self, sender, cid, validity):
        self.vm.sender = sender
        return self.c.set_result_validity(cid, validity)

    def submit(self, sender, cid, text):
        self.vm.sender = sender
        return self.c.submit_post(cid, text)

    def resolve(self, sender, cid, check_id):
        self.vm.sender = sender
        return self.c.resolve_check(cid, check_id)

    # ---- views ------------------------------------------------------------
    def check(self, cid, check_id):
        return self.c.get_check(cid, check_id)

    def community(self, cid):
        return self.c.get_community(cid)

    def rule_set(self, cid, version):
        return self.c.get_rule_set(cid, version)

    def communities(self):
        return self.c.list_community_ids()

    def checks_for(self, cid):
        return self.c.list_check_ids(cid)


@pytest.fixture
def registry(direct_vm, direct_deploy):
    """Deploy the compliance registry with a fresh, time-pinned VM context."""
    c = direct_deploy("contracts/contract.py")
    import genlayer.gl as gl

    direct_vm._llm_mocks.clear()
    ctrl = Controller(direct_vm, gl, c)
    ctrl.set_time(GEN_BASE_TS)
    return ctrl
