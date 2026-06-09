"""Robustness of JSON extraction — weak local models emit slightly malformed JSON."""

from __future__ import annotations

from call_agent.llm import _extract_json_object


def test_clean_json():
    d = _extract_json_object('{"reflection":"ok","should_end":false}')
    assert d["reflection"] == "ok"


def test_unquoted_keys_recovered():
    # Observed on the server: vLLM without guided decoding emitted bareword keys.
    s = '{"reflection":"Владимир, очень приятно. вы кто — вопрос к оператору.",answer:"",branch_signal:"none","should_end":false}'
    d = _extract_json_object(s)
    assert d.get("reflection", "").startswith("Владимир")
    assert d.get("branch_signal") == "none"


def test_trailing_comma_recovered():
    d = _extract_json_object('{"reflection":"ok","facts_update":{"a":"b"},}')
    assert d.get("facts_update") == {"a": "b"}


def test_json_with_prose_wrapper():
    d = _extract_json_object('Вот ответ: {"reflection":"ok"} спасибо')
    assert d.get("reflection") == "ok"
