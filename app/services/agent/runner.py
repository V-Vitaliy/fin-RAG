from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any
from collections.abc import Awaitable, Callable

from app.services.agent.metric_contracts import (
    compact_contract,
    detect_metric,
    get_metric_contract,
    render_contract_for_prompt,
    validate_metric_answer,
)
from app.services.agent.prompts import (
    BASE_SYSTEM_PROMPT,
    build_additional_evidence_prompt,
    build_auto_describe_table_block,
    build_citation_validation_retry_prompt,
    build_decomposition_prompt,
    build_metric_validation_retry_prompt,
    build_narrative_grounding_retry_prompt,
    build_narrative_system_prompt,
    build_self_eval_prompt,
    build_self_eval_retry_prompt,
    build_three_errors_recovery_suffix,
    build_user_question_prompt,
)
from app.services.agent.schemas import AgentAnswer, AgentToolTrace
from app.services.agent.tools import BaseAgentTool

logger = logging.getLogger(__name__)


class AgentRunner:
    MAX_EVAL_RETRIES = 1
    MAX_NARRATIVE_GROUNDING_RETRIES = 1
    MAX_FORMULA_VALIDATION_RETRIES = 1
    MAX_CITATION_RETRIES = 1

    def __init__(
        self,
        *,
        openai_client,
        tools_registry: dict[str, BaseAgentTool],
        model_name: str,
        max_tool_turns: int = 15,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        trace_path: str | None = None,
        max_tool_output_chars: int = 12000,
        event_sink: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
    ):
        self.client = openai_client
        self.tools_registry = tools_registry
        self.model_name = str(model_name)
        self.max_tool_turns = int(max_tool_turns)
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.trace_path = trace_path
        self.max_tool_output_chars = int(max_tool_output_chars)
        self.event_sink = event_sink

        self.tools_schemas = (
            [tool.get_schema() for tool in tools_registry.values()]
            if tools_registry
            else None
        )


    async def _emit(self, event: str, data: dict[str, Any] | None = None) -> None:
        if not self.event_sink:
            return

        try:
            await self.event_sink(event, data or {})
        except Exception as exc:
            logger.warning("Agent event sink failed: %s", exc)

    def _build_create_params(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if tools:
            params["tools"] = tools

        return params

    async def _json_completion(self, prompt: str) -> dict[str, Any]:
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or "{}"
            return json.loads(raw)
        except Exception as exc:
            logger.warning("Agent JSON helper completion failed: %s", exc)
            return {}

    async def _decompose_question(self, question: str) -> dict[str, Any]:
        prompt = build_decomposition_prompt(question)

        result = await self._json_completion(prompt)

        if not isinstance(result, dict):
            return {"sub_questions": [], "search_plan": []}

        result.setdefault("sub_questions", [])
        result.setdefault("search_plan", [])
        return result

    async def _self_evaluate(
        self,
        *,
        question: str,
        messages: list[dict[str, Any]],
    ) -> tuple[bool, str, list[dict[str, Any]], str]:
        history_text = ""

        for message in messages:
            role = message.get("role")

            if role == "tool":
                content = str(message.get("content", ""))
                history_text += (
                    f"Tool {message.get('name')} returned: "
                    f"{content[:3000] if len(content) > 3000 else content}\n"
                )

            elif (
                role == "user"
                and isinstance(message.get("content"), str)
                and "Additional evidence" in str(message.get("content"))
            ):
                content = str(message.get("content", ""))
                history_text += (
                    "User-provided gathered evidence: "
                    f"{content[:3000] if len(content) > 3000 else content}\n"
                )

            elif role == "assistant" and message.get("content"):
                history_text += f"Assistant draft: {message['content']}\n"

        prompt = build_self_eval_prompt(question, history_text)

        result = await self._json_completion(prompt)

        is_ready = bool(result.get("is_ready", True))
        reason = result.get("missing_information", "") or ""
        suggested = result.get("suggested_searches", []) or []
        reason_type = (result.get("reason_type") or "missing_evidence").strip().lower()

        if reason_type == "needs_external":
            is_ready = True

        if not isinstance(suggested, list):
            suggested = []

        if not is_ready:
            logger.warning("Agent self-eval failed [%s]: %s", reason_type, reason)

        return is_ready, reason, suggested, reason_type

    @staticmethod
    def _is_driver_question(question: str) -> bool:
        q = (question or "").lower()
        return any(
            phrase in q
            for phrase in [
                "what drove",
                "driven by",
                "drivers",
                "why did",
                "what caused",
                "reasons for",
                "primarily due to",
                "offset by",
                "margin change",
                "revenue change",
                "sales change",
                "operating margin change",
                "gross margin change",
                "increase was due",
                "decrease was due",
                "increase was driven",
                "decrease was driven",
            ]
        )

    @classmethod
    def _is_narrative_question(cls, question: str) -> bool:
        q = (question or "").lower()
        return cls._is_driver_question(q) or any(
            phrase in q
            for phrase in [
                "major acquisitions",
                "acquisition activities",
                "business combinations",
                "products and services",
                "major products",
                "customer concentration",
                "legal proceedings",
                "legal battles",
                "high growth",
                "growth company",
                "excluding m&a",
                "exclude m&a",
                "organic",
                "organically",
                "constant currency",
            ]
        )

    @classmethod
    def _build_narrative_search_guidance(cls, question: str) -> str:
        q = (question or "").lower()
        guidance: list[str] = []

        if cls._is_driver_question(q):
            guidance.append(
                "For driver/change questions, search MD&A / Results of Operations narrative first, "
                "using terms like: management discussion results of operations due to driven by primarily offset by "
                "currency inflation supply chain acquisition product mix volume price demand."
            )

        if "gross margin" in q:
            guidance.append(
                "For gross margin driver questions, search for: gross profit margin decrease increase cost of products sold "
                "currency commodity inflation supply chain product mix."
            )

        if "operating margin" in q or "operating income" in q:
            guidance.append(
                "For operating margin/income drivers, search for: operating income decrease increase amortization "
                "intangible assets acquisition restructuring operating expenses."
            )

        if "revenue" in q or "sales" in q or "high growth" in q or "growth company" in q:
            guidance.append(
                "For revenue/sales growth questions, search for: net sales increase decrease sales growth results of operations "
                "volume price currency acquisitions organic growth."
            )

        if "acquisition" in q or "business combination" in q:
            guidance.append(
                "For acquisition questions asking what companies were acquired, search narrative notes with: business combinations "
                "acquired outstanding shares purchase price acquired company closed acquisition. Do not answer only from cash-flow line items."
            )

        if not guidance:
            guidance.append(
                "Prefer filing text/narrative evidence before tables when the question asks for explanations, named items, or qualitative facts."
            )

        return " ".join(guidance)

    @classmethod
    def _narrative_grounding_issue(cls, question: str, answer: str) -> str | None:
        if not answer or not cls._is_driver_question(question):
            return None

        answer_lower = answer.lower()
        has_text_citation = bool(re.search(r"\[C\d+\]", answer))
        has_only_table_sources = ("table=tbl_" in answer_lower or "[t" in answer_lower) and not has_text_citation

        generic_phrases = [
            "cost management",
            "operational efficiency",
            "pricing strategy",
            "market conditions",
            "product mix",
            "revenue growth",
            "cost control",
            "supply chain challenges",
        ]
        generic_count = sum(1 for phrase in generic_phrases if phrase in answer_lower)

        if has_only_table_sources and generic_count >= 2:
            return (
                "Narrative driver answer appears to rely on table numbers plus generic explanations. "
                "Search MD&A / Results of Operations text for explicit drivers stated in the filing and revise with [C#] text citations."
            )

        return None

    @staticmethod
    def _answer_has_financial_number(answer: str) -> bool:
        if not answer:
            return False

        return bool(
            re.search(
                r"(?:\$|€|£)?\(?-?\d[\d,]*(?:\.\d+)?\)?\s*(?:%|x|million|billion|thousand|mn|bn)?",
                answer,
                re.IGNORECASE,
            )
        )

    @staticmethod
    def _has_citation_marker(answer: str) -> bool:
        if not answer:
            return False

        return bool(
            re.search(r"\[(?:C|T)\d+\]", answer)
            or re.search(r"\[[^\]]+\bp\.?\s*\d+[^\]]*\]", answer, re.IGNORECASE)
            or re.search(r"\[[^\]]+table=tbl_[^\]]+\]", answer, re.IGNORECASE)
        )

    def _validate_citations(
        self,
        *,
        question: str,
        answer: str,
        narrative_required: bool,
        metric_contract: dict[str, Any] | None,
    ) -> tuple[bool, str]:
        if not answer or answer.strip().lower().startswith("i could not produce"):
            return True, ""

        needs_citation = (
            narrative_required
            or bool(metric_contract)
            or self._answer_has_financial_number(answer)
        )

        if needs_citation and not self._has_citation_marker(answer):
            return False, (
                "The answer contains a financial fact, calculation, or filing-specific narrative but has no citation. "
                "Cite retrieved evidence using [C#]/[T#] markers from search results or the source citation returned by execute_sql."
            )

        return True, ""

    @staticmethod
    def _extract_sql_table_name(query: str | None) -> str | None:
        if not query:
            return None

        match = re.search(
            r'FROM\s+["\'`]?([a-zA-Z_][a-zA-Z0-9_]*)["\'`]?',
            str(query),
            re.IGNORECASE,
        )
        return match.group(1) if match else None

    @staticmethod
    def _is_row_filtered_sql(query: str | None) -> bool:
        q = (query or "").lower()
        return " where " in q and (" ilike " in q or " like " in q or " = " in q)

    def _truncate_tool_result(self, result: str) -> str:
        result = str(result)

        if len(result) <= self.max_tool_output_chars:
            return result

        return (
            result[: self.max_tool_output_chars]
            + f"\n\n[Tool output truncated to {self.max_tool_output_chars} characters]"
        )

    async def _auto_describe_table(
        self,
        *,
        table_name: str,
        reason: str,
        executed_calls: set[tuple[str, str]],
        sample_rows: int = 80,
    ) -> str:
        describe_tool = self.tools_registry.get("describe_table")
        if not describe_tool or not table_name:
            return ""

        kwargs = {
            "table_name": table_name,
            "include_sample": True,
            "sample_rows": sample_rows,
        }
        signature = ("describe_table", json.dumps(kwargs, sort_keys=True, default=str))

        if signature in executed_calls:
            return ""

        executed_calls.add(signature)

        try:
            snapshot = await describe_tool.execute(**kwargs)
        except Exception as exc:
            snapshot = f"Error running automatic describe_table for {table_name}: {exc}"

        return build_auto_describe_table_block(
            table_name=table_name,
            reason=reason,
            snapshot=str(snapshot),
        )

    async def _run_suggested_searches(
        self,
        *,
        suggested: list[dict[str, Any]],
        executed_calls: set[tuple[str, str]],
    ) -> list[str]:
        evidence_blocks: list[str] = []

        for suggestion in suggested[:2]:
            tool_name = suggestion.get("tool")
            query = suggestion.get("query")

            if not tool_name or not query:
                continue

            if tool_name == "search_text":
                kwargs = {"query": query}
            elif tool_name == "search_tables":
                kwargs = {"concept": query}
            elif tool_name == "execute_sql":
                kwargs = {"query": query}
            elif tool_name == "describe_table":
                kwargs = {
                    "table_name": query,
                    "include_sample": True,
                    "sample_rows": 80,
                }
            else:
                continue

            signature = (tool_name, json.dumps(kwargs, sort_keys=True, default=str))
            if signature in executed_calls:
                continue

            tool = self.tools_registry.get(tool_name)
            if not tool:
                continue

            executed_calls.add(signature)

            try:
                result = await tool.execute(**kwargs)
            except Exception as exc:
                result = f"Error executing suggested tool {tool_name}: {exc}"

            evidence_blocks.append(
                f"Suggested tool executed: {tool_name}({json.dumps(kwargs, ensure_ascii=False)})\n"
                f"Result:\n{self._truncate_tool_result(str(result))}"
            )

        return evidence_blocks

    async def _write_trace(self, record: dict[str, Any]) -> None:
        if not self.trace_path:
            return

        path = Path(self.trace_path)

        def write_sync() -> None:
            path.parent.mkdir(parents=True, exist_ok=True) if path.parent != Path(".") else None
            with path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

        try:
            await asyncio.to_thread(write_sync)
        except Exception as exc:
            logger.warning("Failed to write agent trace: %s", exc)

    async def generate_answer(self, *, question: str) -> AgentAnswer:
        metric_name = detect_metric(question)
        metric_contract = get_metric_contract(metric_name) if metric_name else None

        metric_validation_retries = 0
        citation_validation_retries = 0
        narrative_grounding_retries = 0
        self_eval_retries = 0
        last_metric_validation = None

        narrative_required = self._is_narrative_question(question)

        trace: list[AgentToolTrace] = []

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content":BASE_SYSTEM_PROMPT,
            }
        ]

        if metric_contract:
            logger.info("Agent metric contract detected: %s", metric_contract.get("metric_id"))
            messages.append(
                {
                    "role": "system",
                    "content": render_contract_for_prompt(metric_contract),
                }
            )

        if narrative_required:
            messages.append(
                {
                    "role": "system",
                    "content": build_narrative_system_prompt(
                                question,
                                self._build_narrative_search_guidance(question),
                            ),
                }
            )

        messages.append(
            {
                "role": "user",
                "content": build_user_question_prompt(question),
            }
        )

        await self._emit("stage", {"stage": "planning", "message": "Planning search strategy"})
        plan = await self._decompose_question(question)
        if plan.get("search_plan"):
            messages.append(
                {
                    "role": "assistant",
                    "content": f"Decomposition plan: {json.dumps(plan, ensure_ascii=False)}",
                }
            )

        executed_calls: set[tuple[str, str]] = set()
        consecutive_errors = 0
        row_filtered_table_counts: dict[str, int] = {}

        for _turn in range(self.max_tool_turns):
            try:
                await self._emit("stage", {"stage": "thinking", "message": "Analyzing evidence and deciding next step"})
                response = await self.client.chat.completions.create(
                    **self._build_create_params(messages, self.tools_schemas)
                )
            except Exception as exc:
                logger.exception("Agent OpenAI call failed")
                await self._emit(
                    "error",
                    {
                        "message": "Agent OpenAI call failed.",
                        "detail": str(exc),
                    },
                )
                return AgentAnswer(
                    answer=f"API Error: {exc}",
                    tool_calls=trace,
                )

            msg = response.choices[0].message
            msg_dict = msg.model_dump(exclude_none=True)
            messages.append(msg_dict)

            tool_calls = getattr(msg, "tool_calls", None) or []

            if not tool_calls:
                final_answer = msg.content or "Done"

                should_self_eval = (
                    narrative_required
                    and not metric_contract
                    and not final_answer.strip().lower().startswith("i could not produce")
                )

                if should_self_eval:
                    is_ready, reason, suggested, reason_type = await self._self_evaluate(
                        question=question,
                        messages=messages,
                    )

                    if (
                        not is_ready
                        and reason_type in {"missing_evidence", "overbroad"}
                        and suggested
                        and self_eval_retries < self.MAX_EVAL_RETRIES
                    ):
                        self_eval_retries += 1

                        messages.append(
                            {
                                "role": "user",
                                "content":  build_self_eval_retry_prompt(
                                                reason_type=reason_type,
                                                reason=reason,
                                                suggested=suggested,
                                            ),
                            }
                        )

                        forced_evidence_blocks = await self._run_suggested_searches(
                            suggested=suggested,
                            executed_calls=executed_calls,
                        )

                        if forced_evidence_blocks:
                            messages.append(
                                {
                                    "role": "user",
                                    "content": build_additional_evidence_prompt(forced_evidence_blocks)
                                }
                            )
                            continue

                    elif not is_ready:
                        logger.warning(
                            "Agent self-eval did not trigger more retrieval "
                            "(reason_type=%s, retries=%s/%s). Finalizing best supported draft.",
                            reason_type,
                            self_eval_retries,
                            self.MAX_EVAL_RETRIES,
                        )

                narrative_issue = self._narrative_grounding_issue(question, final_answer)
                if narrative_issue and narrative_grounding_retries < self.MAX_NARRATIVE_GROUNDING_RETRIES:
                    narrative_grounding_retries += 1
                    logger.warning("Agent narrative grounding failed: %s", narrative_issue)
                    messages.append(
                        {
                            "role": "user",
                            "content": build_narrative_grounding_retry_prompt(narrative_issue)
                        }
                    )
                    continue

                if metric_contract:
                    last_metric_validation = validate_metric_answer(
                        metric_contract,
                        messages,
                        final_answer,
                        question=question,
                    )

                    if (
                        not last_metric_validation.ok
                        and metric_validation_retries < self.MAX_FORMULA_VALIDATION_RETRIES
                    ):
                        metric_validation_retries += 1
                        logger.warning(
                            "Agent metric validation failed for %s: %s",
                            metric_contract.get("metric_id"),
                            last_metric_validation.reason,
                        )
                        messages.append(
                            {
                                "role": "user",
                                "content": build_metric_validation_retry_prompt(
                                                reason=last_metric_validation.reason,
                                                repair_instruction=last_metric_validation.repair_instruction,
                                            )
                            }
                        )
                        continue

                citation_ok, citation_reason = self._validate_citations(
                    question=question,
                    answer=final_answer,
                    narrative_required=narrative_required,
                    metric_contract=metric_contract,
                )

                if not citation_ok and citation_validation_retries < self.MAX_CITATION_RETRIES:
                    citation_validation_retries += 1
                    logger.warning("Agent citation validation failed: %s", citation_reason)
                    messages.append(
                        {
                            "role": "user",
                            "content": build_citation_validation_retry_prompt(citation_reason),
                        }
                    )
                    continue

                await self._write_trace(
                    {
                        "question": question,
                        "metric_contract": compact_contract(metric_contract) if metric_contract else None,
                        "metric_validation": (
                            {
                                "ok": last_metric_validation.ok,
                                "reason": last_metric_validation.reason,
                                "failure_stage": last_metric_validation.failure_stage,
                            }
                            if last_metric_validation
                            else None
                        ),
                        "citation_validation": {
                            "required": narrative_required
                            or bool(metric_contract)
                            or self._answer_has_financial_number(final_answer),
                            "has_citation": self._has_citation_marker(final_answer),
                        },
                        "narrative_policy_applied": narrative_required,
                        "messages": messages,
                    }
                )
                await self._emit("stage", {"stage": "finalizing", "message": "Preparing final answer"})

                return AgentAnswer(answer=final_answer, tool_calls=trace)

            for tool_call in tool_calls:
                func_name = tool_call.function.name
                raw_args = tool_call.function.arguments or "{}"

                try:
                    args = json.loads(raw_args)
                    if not isinstance(args, dict):
                        raise ValueError("Tool arguments must be a JSON object")

                    signature = (
                        func_name,
                        json.dumps(args, sort_keys=True, default=str),
                    )

                    if signature in executed_calls:
                        tool_result = "Error: You already ran this exact tool call."
                        consecutive_errors += 1
                        ok = False
                    else:
                        executed_calls.add(signature)

                        tool_instance = self.tools_registry.get(func_name)
                        if not tool_instance:
                            tool_result = f"Error: Tool {func_name} not found."
                            consecutive_errors += 1
                            ok = False
                        else:

                            await self._emit(
                                "tool_call",
                                {
                                    "name": func_name,
                                    "arguments": args,
                                },
                            )

                            tool_result = await tool_instance.execute(**args)


                            if func_name == "execute_sql":
                                sql_query = args.get("query", "")
                                table_name = self._extract_sql_table_name(sql_query)

                                if table_name and self._is_row_filtered_sql(sql_query):
                                    row_filtered_table_counts[table_name] = (
                                        row_filtered_table_counts.get(table_name, 0) + 1
                                    )

                                result_lower = str(tool_result).lower()
                                needs_schema_recovery = (
                                    "sql error" in result_lower
                                    or "binder error" in result_lower
                                    or "referenced column" in result_lower
                                    or "not found in from clause" in result_lower
                                )
                                repeated_row_probe = (
                                    bool(table_name)
                                    and row_filtered_table_counts.get(table_name, 0) >= 3
                                )

                                if table_name and needs_schema_recovery:
                                    tool_result = str(tool_result) + await self._auto_describe_table(
                                        table_name=table_name,
                                        reason="SQL column/schema error",
                                        executed_calls=executed_calls,
                                        sample_rows=80,
                                    )
                                elif table_name and repeated_row_probe:
                                    tool_result = str(tool_result) + await self._auto_describe_table(
                                        table_name=table_name,
                                        reason="three row-filter queries against the same table",
                                        executed_calls=executed_calls,
                                        sample_rows=80,
                                    )

                            consecutive_errors = 0
                            ok = True

                except Exception as exc:
                    logger.exception("Agent tool execution failed")
                    args = {"raw_arguments": raw_args}
                    tool_result = f"Error executing tool: {exc}"
                    consecutive_errors += 1
                    ok = False

                if consecutive_errors >= 3:
                    logger.warning("Agent had 3 consecutive tool errors after %s", func_name)
                    search_tool_names = {"search_text", "search_tables", "describe_table"}
                    executed_calls = {call for call in executed_calls if call[0] not in search_tool_names}
                    tool_result =  str(tool_result) + build_three_errors_recovery_suffix()

                truncated_result = self._truncate_tool_result(str(tool_result))

                await self._emit(
                    "tool_result",
                    {
                        "name": func_name,
                        "ok": ok,
                        "preview": truncated_result[:1000],
                    },
                )

                trace.append(
                    AgentToolTrace(
                        name=func_name,
                        arguments=args,
                        result_preview=truncated_result[:1000],
                        ok=ok,
                    )
                )

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": truncated_result,
                    }
                )

        await self._write_trace(
            {
                "question": question,
                "metric_contract": compact_contract(metric_contract) if metric_contract else None,
                "metric_validation": (
                    {
                        "ok": last_metric_validation.ok,
                        "reason": last_metric_validation.reason,
                        "failure_stage": last_metric_validation.failure_stage,
                    }
                    if last_metric_validation
                    else None
                ),
                "narrative_policy_applied": narrative_required,
                "error": "Max tool turns reached",
                "messages": messages,
            }
        )

        await self._emit(
            "error",
            {
                "message": "Max tool turns reached.",
            },
        )

        return AgentAnswer(
            answer="I could not produce an answer from the available evidence within the allowed steps.",
            tool_calls=trace,
        )