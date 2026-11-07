"""Harness executed in a separate, isolated Python process (`python -I`).

Reads a JSON payload from stdin and writes a JSON result to `result_path`.

Modes:
  "tests"  — run the solution, then each assert; report pass/fail and error type.
  "probes" — run the solution, then evaluate each expression; report repr(output)
             or the exception type. Used by an agent to compare two solutions on
             inputs it wrote itself.

Must stay dependency-free.
"""

import json
import sys


def _load(payload: dict) -> tuple[dict, str | None]:
    namespace: dict = {"__name__": "solution"}
    exec(compile(payload["setup_code"], "<setup>", "exec"), namespace)
    try:
        exec(compile(payload["solution"], "<solution>", "exec"), namespace)
    except BaseException as exc:  # noqa: BLE001 — any failure is a failed attempt
        return namespace, type(exc).__name__
    return namespace, None


def _run_tests(payload: dict, namespace: dict) -> list[dict]:
    results: list[dict] = []
    for test in payload["tests"]:
        try:
            exec(compile(test, "<test>", "exec"), namespace)
            results.append({"passed": True})
        except BaseException as exc:  # noqa: BLE001
            results.append({"passed": False, "type": type(exc).__name__})
    return results


def _run_probes(payload: dict, namespace: dict) -> list[str]:
    outputs = []
    for expression in payload["probes"]:
        try:
            outputs.append(
                repr(eval(compile(expression, "<probe>", "eval"), namespace))
            )
        except BaseException as exc:  # noqa: BLE001
            outputs.append("!" + type(exc).__name__)
    return outputs


def main() -> None:
    payload = json.load(sys.stdin)
    namespace, load_error = _load(payload)
    result: dict = {"load_error": load_error}
    if load_error is None:
        if payload["mode"] == "tests":
            result["tests"] = _run_tests(payload, namespace)
        else:
            result["outputs"] = _run_probes(payload, namespace)
    with open(payload["result_path"], "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
