from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SSEEvent:
    event: str
    data: dict[str, Any]


async def parse_sse_lines(lines: AsyncIterator[str]) -> AsyncIterator[SSEEvent]:
    event_name = "message"
    data_lines: list[str] = []

    async for raw_line in lines:
        line = raw_line.strip("\r\n")

        if not line:
            if data_lines:
                raw_data = "\n".join(data_lines)
                try:
                    data = json.loads(raw_data)
                except json.JSONDecodeError:
                    data = {"raw": raw_data}

                yield SSEEvent(event=event_name, data=data)

            event_name = "message"
            data_lines = []
            continue

        if line.startswith("event:"):
            event_name = line.removeprefix("event:").strip()
            continue

        if line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").strip())
            continue

    if data_lines:
        raw_data = "\n".join(data_lines)
        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError:
            data = {"raw": raw_data}

        yield SSEEvent(event=event_name, data=data)