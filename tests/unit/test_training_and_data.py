"""Adapter registry, dataset writing, MBPP parsing, EvalPlus parsing, configs."""

import json
from pathlib import Path

import pytest

from src.config import load_config
from src.eval.scoring import pass_at_1
from src.models.core import Message, PreferencePair
from src.tasks.code import entry_point, signature, split, to_problem
from src.training.registry import AdapterRegistry, AdapterVersion
from src.training.trainer import write_dataset

CONFIGS = Path(__file__).resolve().parents[2] / "config"


class TestRegistry:
    def test_push_current_rollback(self, tmp_path):
        reg = AdapterRegistry(tmp_path / "registry.json")
        assert reg.current("a", "logical") is None
        reg.push("a", "logical", AdapterVersion(cycle=1, served_name="v1", path="p1"))
        reg.push("a", "logical", AdapterVersion(cycle=2, served_name="v2", path="p2"))
        assert reg.rollback("a", "logical", cycle=2).served_name == "v1"
        assert (
            reg.rollback("a", "logical", cycle=2).served_name == "v1"
        )  # nothing from cycle 2 left
        assert (
            AdapterRegistry(tmp_path / "registry.json").current("a", "logical").cycle
            == 1
        )

    def test_rollback_to_base(self, tmp_path):
        reg = AdapterRegistry(tmp_path / "registry.json")
        reg.push(
            "a", "coordinator", AdapterVersion(cycle=1, served_name="v1", path="p1")
        )
        assert reg.rollback("a", "coordinator", cycle=1) is None


def test_write_dataset_conversational_format(tmp_path):
    pair = PreferencePair(
        task_id="p",
        domain="code",
        agent="a",
        role="logical",
        prompt=[Message(role="system", content="s"), Message(role="user", content="u")],
        chosen="good",
        rejected="bad",
        source="self",
    )
    write_dataset([pair], tmp_path / "d.jsonl")
    row = json.loads((tmp_path / "d.jsonl").read_text())
    assert row["prompt"][1] == {"role": "user", "content": "u"}
    assert row["chosen"] == [{"role": "assistant", "content": "good"}]
    assert row["rejected"] == [{"role": "assistant", "content": "bad"}]


class TestMBPP:
    ROW = {
        "task_id": 601,
        "text": "Write a function to find the longest chain.",
        "code": "class Pair:\n    pass\ndef max_chain_length(arr, n):\n    return n",
        "test_list": ["assert max_chain_length([1], 4) == 3"],
        "test_setup_code": "",
    }

    def test_to_problem_hides_tests(self):
        p = to_problem(self.ROW)
        assert p.task_id == "mbpp/601" and p.entry_point == "max_chain_length"
        assert "def max_chain_length(arr, n):" in p.prompt
        assert "assert" not in p.prompt and p.tests == self.ROW["test_list"]

    def test_entry_point_prefers_tested_function(self):
        code = "def helper(x):\n    return x\ndef main(x):\n    return helper(x)"
        assert entry_point(code, ["assert main(1) == 1"]) == "main"

    def test_space_before_parenthesis(self):
        code = "import math \r\ndef max_Prime_Factors (n): \r\n    return n"
        assert (
            entry_point(code, ["assert max_Prime_Factors(15) == 5"])
            == "max_Prime_Factors"
        )
        assert signature(code, "max_Prime_Factors") == "def max_Prime_Factors (n):"

    def test_signature_multiline(self):
        code = "def f(a,\n      b):\n    return a"
        assert signature(code, "f") == "def f(a,\n      b):"

    def test_split_is_deterministic_and_disjoint(self):
        problems = [to_problem({**self.ROW, "task_id": i}) for i in range(20)]
        train, val = split(problems, 5, seed=0)
        assert len(val) == 5 and len(train) == 15
        assert not {p.task_id for p in train} & {p.task_id for p in val}
        assert split(problems, 5, seed=0) == (train, val)


def test_pass_at_1(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(
        json.dumps(
            {
                "eval": {
                    "t1": [{"base_status": "pass", "plus_status": "pass"}],
                    "t2": [{"base_status": "pass", "plus_status": "fail"}],
                    "t3": [{"base_status": "fail", "plus_status": "fail"}],
                    "t4": [{"base_status": "fail", "plus_status": "pass"}],
                }
            }
        )
    )
    assert pass_at_1(path) == {"base": 0.5, "plus": 0.25}


@pytest.mark.parametrize("name", ["general.yaml", "general_solo.yaml", "smoke.yaml"])
def test_configs_load(name):
    cfg = load_config(CONFIGS / name)
    assert cfg.environment.feedback.value == "binary"
    assert set(cfg.agents.seeds) == set(cfg.agents.names)


def test_pair_and_solo_differ_only_in_partner():
    pair, solo = load_config(CONFIGS / "general.yaml"), load_config(
        CONFIGS / "general_solo.yaml"
    )
    assert pair.agents.names == ["a", "b"] and solo.agents.names == ["a"]
    assert pair.agents.seeds["a"] == solo.agents.seeds["a"]
    assert pair.training == solo.training and pair.memory == solo.memory
    assert pair.environment == solo.environment and pair.data == solo.data
