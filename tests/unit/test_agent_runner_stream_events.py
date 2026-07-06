from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.services.agent.runner import AgentRunner
from app.services.agent.tools import BaseAgentTool


class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []

    def model_dump(self, exclude_none=True):
        data = {}
        if self.content is not None:
            data["role"] = "assistant"
            data["content"] = self.content
        if self.tool_calls:
            data["role"] = "assistant"
            data["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in self.tool_calls
            ]
        return data


class FakeToolCall:
    def __init__(self, name, arguments):
        self.id = "call_1"
        self.function = SimpleNamespace(
            name=name,
            arguments=json.dumps(arguments),
        )


class FakeCompletions:
    def __init__(self):
        self.calls = 0

    async def create(self, **kwargs):
        # JSON helper completions for decomposition/self-eval
        if kwargs.get("response_format"):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=FakeMessage(
                            content=json.dumps(
                                {
                                    "sub_questions": [],
                                    "search_plan": [],
                                }
                            )
                        )
                    )
                ]
            )

        self.calls += 1

        if self.calls == 1:
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=FakeMessage(
                            tool_calls=[
                                FakeToolCall(
                                    "search_text",
                                    {"query": "announcement"},
                                )
                            ]
                        )
                    )
                ]
            )

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=FakeMessage(
                        content="The company announced a shareholder vote result [C1]."
                    )
                )
            ]
        )


class FakeOpenAIClient:
    def __init__(self):
        self.chat = SimpleNamespace(
            completions=FakeCompletions()
        )


class FakeSearchTextTool(BaseAgentTool):
    name = "search_text"

    def get_schema(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "fake",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }

    async def execute(self, **kwargs):
        return "[C1 | source=fake] Evidence text"


@pytest.mark.asyncio
async def test_agent_runner_emits_stage_and_tool_events():
    events = []

    async def sink(event, data):
        events.append((event, data))

    runner = AgentRunner(
        openai_client=FakeOpenAIClient(),
        tools_registry={"search_text": FakeSearchTextTool()},
        model_name="fake-model",
        event_sink=sink,
    )

    result = await runner.generate_answer(question="What did the company announce?")

    assert "shareholder vote" in result.answer
    assert result.tool_calls[0].name == "search_text"

    event_names = [event for event, _data in events]
    assert "stage" in event_names
    assert "tool_call" in event_names
    assert "tool_result" in event_names