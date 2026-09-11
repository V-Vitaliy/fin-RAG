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
    build_forced_final_answer_prompt,
    build_unsupported_assumption_retry_prompt,
    build_no_tool_retrieval_retry_prompt,
)
from app.services.agent.schemas import AgentAnswer, AgentToolTrace
from app.services.agent.tools import BaseAgentTool

logger = logging.getLogger(__name__)


class AgentRunner:
    MAX_EVAL_RETRIES = 1
    MAX_NARRATIVE_GROUNDING_RETRIES = 1
    MAX_FORMULA_VALIDATION_RETRIES = 1
    MAX_CITATION_RETRIES = 1
    MAX_UNSUPPORTED_ASSUMPTION_RETRIES = 1
    MAX_NO_TOOL_RETRIEVAL_RETRIES = 1

    def __init__(
        self,
        *,
        openai_client,
        tools_registry: dict[str, BaseAgentTool],
        model_name: str,
        max_tool_turns: int = 15,
        max_tokens: int = 4096,
        reasoning_effort: str = "low",
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
        self.reasoning_effort = str(reasoning_effort or "low").strip().lower()
        if self.reasoning_effort not in {"low", "medium", "high"}:
            self.reasoning_effort = "low"

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

    def _is_reasoning_model(self) -> bool:
        model = self.model_name.lower()
        return (
                model.startswith("o")
                or model.startswith("gpt-5")
        )

    def _instruction_role(self) -> str:
        return "developer" if self._is_reasoning_model() else "system"

    def _build_create_params(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
        }

        if self._is_reasoning_model():
            params["max_completion_tokens"] = self.max_tokens
            params["reasoning_effort"] = self.reasoning_effort
        else:
            params["temperature"] = self.temperature
            params["max_tokens"] = self.max_tokens

        if tools:
            params["tools"] = tools

        return params

    async def _json_completion(self, prompt: str) -> dict[str, Any]:
        try:
            messages = [{"role": "user", "content": prompt}]

            params: dict[str, Any] = {
                "model": self.model_name,
                "messages": messages,
                "response_format": {"type": "json_object"},
            }

            if self._is_reasoning_model():
                params["max_completion_tokens"] = min(self.max_tokens, 1200)
                params["reasoning_effort"] = self.reasoning_effort
            else:
                params["temperature"] = 0.0
                params["max_tokens"] = min(self.max_tokens, 1200)

            response = await self.client.chat.completions.create(**params)
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
                "national securities exchange",
                "trading symbol",
                "title of each class",
                "registered to trade",
                "exchange on which registered",
                "adjusted eps",
                "adjusted earnings per share",
                "eps expected to accelerate",
                "votes against",
                "board member nominee",
                "nominees",
                "notional",
                "derivative",
                "cross currency",
                "cross-currency",
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

        if "adjusted eps" in q or "adjusted earnings per share" in q or ("eps" in q and "accelerat" in q):
            guidance.append(
                "For adjusted EPS acceleration/deceleration questions, compare the same metric across periods: "
                "explicit FY2022 adjusted EPS growth versus explicit FY2023 adjusted EPS guidance/growth. "
                "Do not substitute adjusted operational sales growth, reported EPS, or a different guidance midpoint/change metric when the filing gives adjusted EPS growth rates."
            )

        if "votes against" in q or "board member nominee" in q or "nominees" in q:
            guidance.append(
                "For board nominee voting questions, search the Form 8-K voting results / Proposal 1 table using terms: "
                "Proposal 1 elect directors nominees votes for votes against abstentions broker non-votes. "
                "Extract the nominee with the highest Votes Against and compare against the next-highest nominee."
            )

        if "derivative" in q or "notional" in q or "cross currency" in q or "cross-currency" in q:
            guidance.append(
                "For derivative notional questions, prefer tables explicitly about outstanding derivative instruments or notional amounts outstanding at year-end. "
                "Do not answer from tables about derivatives entered into during the year, fair values, gains/losses, or collateral unless the question asks for those."
            )

        if "national securities exchange" in q or "trading symbol" in q or "registered to trade" in q:
            guidance.append(
                "For securities registered on a national exchange, search the cover-page registration table using exact wording: "
                "title of each class trading symbol name of each exchange on which registered. "
                "Do not answer from marketable securities, investments, debt balances, or derivative notes."
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

        # Only registry markers can be resolved into citations/source_documents.
        # Raw strings such as "[ACME p.10, table=tbl_...]" look like citations
        # to the model, but AgentCitationRegistry cannot attach source links for them.
        return bool(re.search(r"\[(?:C|T)\d+\]", answer))

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
    def _answer_has_unsupported_numeric_assumption(answer: str) -> bool:
        if not answer:
            return False

        lowered = answer.lower()
        assumption_terms = [
            "assumed",
            "assumption",
            "not retrieved",
            "not available",
            "placeholder",
            "typically",
            "estimated",
            "estimate",
            "using 100,000",
            "100,000 million",
        ]

        if not any(term in lowered for term in assumption_terms):
            return False

        return AgentRunner._answer_has_financial_number(answer)

    @staticmethod
    def _is_generic_failure_answer(answer: str) -> bool:
        normalized = (answer or "").strip().lower()
        return normalized.startswith("i could not produce an answer") or normalized.startswith(
            "i could not find enough evidence"
        )

    @staticmethod
    def _answer_requests_user_provided_document(answer: str) -> bool:
        text = (answer or "").lower()
        if not text:
            return False

        request_terms = [
            "please provide",
            "provide one of the following",
            "upload the",
            "upload a",
            "provide a link",
            "link to the",
            "send me the",
            "give me the document",
            "i don’t yet have any retrieved documents",
            "i don't yet have any retrieved documents",
            "i can’t retrieve",
            "i can't retrieve",
            "once you provide",
            "confirm which company",
            "company name (or ticker)",
        ]
        return any(term in text for term in request_terms)

    def _retrieval_tools_available(self) -> bool:
        return bool(
            self.tools_registry.get("search_text")
            or self.tools_registry.get("search_tables")
        )

    def _requires_retrieval_retry_before_final(
        self,
        *,
        final_answer: str,
        trace: list[AgentToolTrace],
        narrative_required: bool,
        metric_contract: dict[str, Any] | None,
    ) -> bool:
        if not self._retrieval_tools_available():
            return False

        if self._has_successful_evidence(trace):
            return False

        if self._answer_requests_user_provided_document(final_answer):
            return True

        # In this RAG agent, filing-specific narrative/metric answers must be grounded.
        # If the model tries to answer from memory before any retrieval, force one retrieval attempt.
        if narrative_required or metric_contract:
            return True

        if self._answer_has_financial_number(final_answer) and not self._has_citation_marker(final_answer):
            return True

        return False


    @staticmethod
    def _tool_output_is_error(tool_name: str, output: str) -> bool:
        text = str(output or "").lstrip()
        lowered = text.lower()

        if lowered.startswith("sql error:"):
            return True
        if lowered.startswith("error:"):
            return True
        if lowered.startswith("calculation error:"):
            return True
        if lowered.startswith("error executing tool:"):
            return True
        if lowered.startswith("error describing table:"):
            return True
        if "binder error:" in lowered[:1200]:
            return True
        if "referenced column" in lowered[:1200] and "not found" in lowered[:1200]:
            return True

        return False

    @classmethod
    def _tool_output_has_evidence(cls, tool_name: str, output: str) -> bool:
        text = str(output or "")
        lowered = text.lower()

        if not text.strip() or cls._tool_output_is_error(tool_name, text):
            return False

        if "query returned 0 rows" in lowered:
            return False

        if tool_name in {"search_text", "search_tables", "describe_table", "execute_sql"}:
            return bool(
                re.search(r"\[(?:C|T)\d+\s*\|", text)
                or "Source: [" in text
                or "| source=" in text
                or "table=tbl_" in text
            )

        if tool_name == "calculate":
            return bool(re.fullmatch(r"\s*-?\d+(?:\.\d+)?\s*", text))

        if tool_name == "get_financial_formula":
            return "Formula:" in text or "required components" in lowered

        return False

    @classmethod
    def _has_successful_evidence(cls, trace: list[AgentToolTrace]) -> bool:
        return any(
            call.ok and cls._tool_output_has_evidence(call.name, call.result_preview)
            for call in trace
        )

    @classmethod
    def _successful_tool_history(cls, messages: list[dict[str, Any]]) -> str:
        blocks: list[str] = []

        for message in messages:
            if message.get("role") != "tool":
                continue

            tool_name = str(message.get("name") or "")
            content = str(message.get("content") or "")

            if not cls._tool_output_has_evidence(tool_name, content):
                continue

            if len(content) > 5000:
                content = content[:5000] + "\n[Successful tool output truncated for forced finalization]"

            blocks.append(f"Tool {tool_name} returned successful evidence:\n{content}")

        return "\n\n".join(blocks[-12:])

    async def _force_final_answer_from_evidence(
        self,
        *,
        question: str,
        messages: list[dict[str, Any]],
        metric_contract: dict[str, Any] | None,
        narrative_required: bool,
        reason: str,
    ) -> str | None:
        evidence_history = self._successful_tool_history(messages)
        if not evidence_history.strip():
            return None

        prompt = build_forced_final_answer_prompt(
            question=question,
            reason=reason,
            evidence_history=evidence_history,
            metric_contract=compact_contract(metric_contract) if metric_contract else None,
            narrative_required=narrative_required,
        )

        try:
            response = await self.client.chat.completions.create(
                **self._build_create_params(
                    [
                        {"role": self._instruction_role(), "content": BASE_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    tools=None,
                )
            )
        except Exception as exc:
            logger.warning("Agent forced finalization failed: %s", exc)
            return None

        answer = response.choices[0].message.content or ""
        answer = answer.strip()
        return answer or None

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
            path.parent.mkdir(parents=True, exist_ok=True) if path.parent != Path("../../../../../Downloads") else None
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
        unsupported_assumption_retries = 0
        no_tool_retrieval_retries = 0
        instruction_role = self._instruction_role()

        narrative_required = self._is_narrative_question(question)

        trace: list[AgentToolTrace] = []

        messages: list[dict[str, Any]] = [
            {
                "role": instruction_role,
                "content": BASE_SYSTEM_PROMPT,
            }
        ]

        if metric_contract:
            logger.info("Agent metric contract detected: %s", metric_contract.get("metric_id"))
            messages.append(
                {
                    "role": instruction_role,
                    "content": render_contract_for_prompt(metric_contract),
                }
            )

        if narrative_required:
            messages.append(
                {
                    "role": instruction_role,
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

                if (
                    no_tool_retrieval_retries < self.MAX_NO_TOOL_RETRIEVAL_RETRIES
                    and self._requires_retrieval_retry_before_final(
                        final_answer=final_answer,
                        trace=trace,
                        narrative_required=narrative_required,
                        metric_contract=metric_contract,
                    )
                ):
                    no_tool_retrieval_retries += 1
                    logger.warning(
                        "Agent attempted to finalize without retrieval evidence; forcing retrieval retry."
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": build_no_tool_retrieval_retry_prompt(question),
                        }
                    )
                    continue

                if self._is_generic_failure_answer(final_answer) and self._has_successful_evidence(trace):
                    repaired_answer = await self._force_final_answer_from_evidence(
                        question=question,
                        messages=messages,
                        metric_contract=metric_contract,
                        narrative_required=narrative_required,
                        reason="model returned generic failure despite successful tool evidence",
                    )
                    if repaired_answer:
                        logger.warning("Agent generic failure replaced by forced finalization from evidence.")
                        final_answer = repaired_answer
                        messages.append(
                            {
                                "role": "assistant",
                                "content": final_answer,
                            }
                        )

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

                if (
                    self._answer_has_unsupported_numeric_assumption(final_answer)
                    and unsupported_assumption_retries < self.MAX_UNSUPPORTED_ASSUMPTION_RETRIES
                ):
                    unsupported_assumption_retries += 1
                    logger.warning("Agent unsupported numeric assumption validation failed.")
                    messages.append(
                        {
                            "role": "user",
                            "content": build_unsupported_assumption_retry_prompt(),
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

                            if self._tool_output_is_error(func_name, str(tool_result)):
                                consecutive_errors += 1
                                ok = False
                            else:
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

        forced_answer = None
        if self._has_successful_evidence(trace):
            forced_answer = await self._force_final_answer_from_evidence(
                question=question,
                messages=messages,
                metric_contract=metric_contract,
                narrative_required=narrative_required,
                reason="max tool turns reached despite successful tool evidence",
            )

        return AgentAnswer(
            answer=forced_answer
            or "I could not produce an answer from the available evidence within the allowed steps.",
            tool_calls=trace,
        )