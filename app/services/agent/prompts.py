from __future__ import annotations

import json
from typing import Any


BASE_SYSTEM_PROMPT = (
    "You are a financial AI assistant. Use tools to find data and calculate answers. "
    "NEVER invent or combine column names with dots/newlines (e.g., do not create 'Year_2022' if the column is 'Year'). "
    "Use ONLY the exact quoted column names provided. If a year is in a data row, select the base column name. "
    "For arithmetic, call calculate(). For formula/ratio questions, follow any provided financial metric contract exactly. "
    "If the question explicitly defines a formula or source constraint, that user-defined formula/source constraint overrides generic formula memory. "
    "When comparing multiple rows, segments, years, or categories from one table, inspect the table with describe_table(include_sample=true) or query all relevant rows in one batched SELECT. "
    "Do not issue more than two separate row-filter SQL queries against the same table before inspecting a table sample/snapshot. "
    "Every factual financial answer must cite retrieved evidence. Use [C#]/[T#] markers from search results or the source citation returned by execute_sql. "
    "When using search_text or search_tables, you may provide 1-3 query_rewrites/concept_rewrites in the tool arguments. "
    "Those rewrites should be model-generated paraphrases of the same user question, using generic financial or MD&A language only. "
    "Do not put expected answers, exact row names, table IDs, page numbers, or unsupported facts into rewrites. "
    "For calculated metrics, cite the source table/page for each input component, not just the final calculation."
)


def build_user_question_prompt(question: str) -> str:
    return f"Question: {question}"


def build_decomposition_prompt(question: str) -> str:
    return (
        "You are an expert financial analyst. Break down the user's question into logical sub-questions "
        "and create a search plan. Output MUST be valid JSON.\n"
        "Format:\n"
        "{\n"
        '  "sub_questions": ["Q1", "Q2"],\n'
        '  "search_plan": [\n'
        '    {"query": "search term", "type": "text|table", "priority": 1}\n'
        "  ]\n"
        "}\n"
        f"Question: {question}"
    )


def build_self_eval_prompt(question: str, history_text: str) -> str:
    return (
        "Evaluate whether the provided tool history contains enough filing evidence to answer the user's question.\n"
        "Do NOT require external benchmarks, industry averages, analyst commentary, or market context unless the user explicitly asks for external comparison. "
        "For filing-only questions, if the filing has enough numbers or narrative facts to answer, mark is_ready=true.\n"
        "For narrative driver questions, require explicit filing wording about drivers/causes (for example: due to, driven by, primarily, offset by, currency, inflation, supply chain, acquisition, product mix). "
        "Do not accept a generic business explanation if the filing evidence does not say those drivers.\n"
        "Output MUST be valid JSON.\n"
        "Format:\n"
        "{\n"
        '  "is_ready": true|false,\n'
        '  "reason_type": "missing_evidence|bad_calculation|uncited|overbroad|needs_external",\n'
        '  "missing_information": "what is still needed if false",\n'
        '  "suggested_searches": [{"tool": "search_text", "query": "..."}]\n'
        "}\n"
        "Guidance for suggested_searches: suggest at most 2 searches. Use search_text for narrative/driver evidence and search_tables for table IDs. "
        "If the problem is calculation, citation, or external benchmarks, do not suggest more searches.\n"
        f"Question: {question}\n\nHistory:\n{history_text}"
    )


def build_narrative_system_prompt(question: str, narrative_guidance: str) -> str:
    return (
        "This is a filing-specific narrative question. Prefer MD&A, Results of Operations, "
        "segment discussion, acquisition/business-combination notes, legal/proceedings notes, or other source text that states concrete facts. "
        "Do not answer narrative driver questions from financial tables alone, and do not use generic business explanations unless the filing evidence supports those exact drivers. "
        "For organic/ex-M&A questions, prefer organic/constant-currency/acquisition-adjusted evidence over total reported growth. "
        "For acquisition questions asking which companies were acquired, do not answer only from cash-flow statement acquisition amounts; find the narrative acquisition note naming the acquired businesses. "
        "Cite the source evidence used. "
        "For the first search_text call, include 1-3 query_rewrites that paraphrase the same narrative intent with generic MD&A language. "
        + narrative_guidance
    )


def build_self_eval_retry_prompt(
    *,
    reason_type: str,
    reason: str,
    suggested: list[dict[str, Any]],
) -> str:
    return (
        f"Your answer may be incomplete. Reason type: {reason_type}. Reason: {reason}. "
        f"You may perform at most one more targeted retrieval using these suggestions: {json.dumps(suggested)}. "
        "Do not look for external benchmarks unless explicitly requested."
    )


def build_additional_evidence_prompt(evidence_blocks: list[str]) -> str:
    return (
        "Additional evidence gathered after self-evaluation. "
        "Use it to revise and complete the answer.\n\n"
        + "\n\n".join(evidence_blocks)
    )


def build_narrative_grounding_retry_prompt(issue: str) -> str:
    return (
        f"Narrative grounding validation failed. Reason: {issue}\n"
        "Perform one targeted search_text call for MD&A / Results of Operations narrative evidence, then revise. "
        "Do not answer a driver question from tables alone."
    )


def build_metric_validation_retry_prompt(
    *,
    reason: str,
    repair_instruction: str,
) -> str:
    return (
        "Financial metric contract validation failed. "
        f"Reason: {reason}\n"
        f"Repair instruction: {repair_instruction}\n"
        "Continue searching if needed, then provide a corrected answer. "
        "Do not repeat the invalid calculation."
    )


def build_citation_validation_retry_prompt(reason: str) -> str:
    return (
        "Citation validation failed. "
        f"Reason: {reason}\n"
        "Revise the answer using only previously retrieved evidence. Add citations next to the factual numbers/claims. "
        "Do not invent citations; use [C#]/[T#] or the source citation line returned by execute_sql."
    )


def build_three_errors_recovery_suffix() -> str:
    return (
        "\n\nThree errors in a row. Reset your approach entirely — "
        "your previous search paths are cleared so you can try new ones: "
        "1) search_tables with a DIFFERENT concept keyword to find the right table, "
        "2) describe_table(table_name, include_sample=true) to confirm exact column names and inspect a table snapshot, "
        "3) use one batched SELECT for all relevant rows instead of row-by-row SQL, "
        "4) calculate() with the values you find. "
        "If you already have the numbers, skip directly to step 4."
    )


def build_auto_describe_table_block(
    *,
    table_name: str,
    reason: str,
    snapshot: str,
) -> str:
    return (
        f"\n\n[Automatic table snapshot triggered: {reason}]\n"
        f"Instead of issuing more row-by-row SQL against {table_name}, use this schema/sample and batch the needed rows.\n"
        f"{snapshot}"
    )