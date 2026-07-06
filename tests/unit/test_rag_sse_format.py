from __future__ import annotations

import json

from app.api.routes.rag import _sse


def test_sse_format():
    raw = _sse("stage", {"stage": "planning"})

    assert raw.startswith("event: stage\n")
    assert "data: " in raw
    assert raw.endswith("\n\n")

    data_line = [line for line in raw.splitlines() if line.startswith("data: ")][0]
    payload = json.loads(data_line.removeprefix("data: "))

    assert payload == {"stage": "planning"}