from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ValidationResult:
    ok: bool
    reason: str = ""
    repair_instruction: str = ""
    failure_stage: str = "formula_validation"


METRIC_CONTRACTS: dict[str, dict[str, Any]] = {
    "quick_ratio": {
        "display_name": "Quick Ratio",
        "aliases": ["quick ratio", "acid-test ratio", "liquidity profile based on quick ratio"],
        "formula": "(total_current_assets - inventories) / total_current_liabilities",
        "required_components": [
            {
                "name": "total_current_assets",
                "aliases": ["total current assets"],
                "preferred_statement": "balance_sheet",
            },
            {
                "name": "inventories",
                "aliases": ["inventories", "inventory", "total inventories"],
                "preferred_statement": "balance_sheet",
            },
            {
                "name": "total_current_liabilities",
                "aliases": ["total current liabilities", "current liabilities"],
                "preferred_statement": "balance_sheet",
            },
        ],
        "calculation_steps": [
            "FinanceBench-style quick ratio = (total current assets - inventories) / total current liabilities unless the user explicitly defines another quick-assets formula.",
            "Do not treat Total current assets as Cash and cash equivalents.",
            "If the filing provides cash, marketable securities, and accounts receivable cleanly, the classic quick-assets formula may be used, but disclose the components used.",
        ],
        "suggested_searches": [
            "total current assets total inventories total current liabilities balance sheet",
            "inventories current assets current liabilities balance sheet",
            "quick ratio total current assets inventories current liabilities",
        ],
        "forbidden_or_risky_components": [
            {
                "term": "total current assets / total current liabilities",
                "reason": "This is current ratio, not quick ratio. Inventories must be subtracted or quick assets must be retrieved directly.",
                "hard_fail_if_used_directly": True,
            },
            {
                "term": "cash and cash equivalents = total current assets",
                "reason": "Total current assets is not cash. Do not relabel total current assets as cash or quick assets.",
                "hard_fail_if_used_directly": True,
            },
        ],
        "output_contract": {"format": "decimal_ratio", "unit": "x", "interpretation": "below 1.0 is usually weaker liquidity"},
    },
    "depreciation_and_amortization_margin": {
        "display_name": "Depreciation and Amortization Margin",
        "aliases": ["d&a margin", "da margin", "depreciation and amortization margin", "depreciation and amortization % margin"],
        "formula": "depreciation_and_amortization / revenue",
        "required_components": [
            {
                "name": "depreciation_and_amortization",
                "aliases": ["depreciation and amortization", "D&A", "depreciation & amortization"],
                "preferred_statement": "cash_flow_statement",
            },
            {
                "name": "revenue",
                "aliases": ["revenue", "net revenue", "net sales", "sales"],
                "preferred_statement": "income_statement",
            },
        ],
        "calculation_steps": [
            "Find depreciation and amortization from the cash flow statement.",
            "Find revenue/net revenue/net sales from the income statement.",
            "D&A margin = depreciation and amortization / revenue. Return percent if the question asks for % margin.",
        ],
        "suggested_searches": [
            "depreciation and amortization cash flow statement",
            "net revenue income statement",
            "D&A margin depreciation amortization revenue",
        ],
        "forbidden_or_risky_components": [
            {
                "term": "accumulated depreciation and amortization",
                "reason": "Accumulated depreciation is a balance-sheet contra-asset, not the period D&A expense from the cash flow statement.",
                "hard_fail_if_used_directly": True,
            }
        ],
        "output_contract": {"format": "percent", "unit": "%"},
    },
    "capital_intensive_assessment": {
        "display_name": "Capital-Intensive Business Assessment",
        "aliases": ["capital-intensive business", "capital intensive business", "is capital intensive", "is capital-intensive"],
        "formula": "assess capex/revenue, fixed_assets/total_assets, and roa together",
        "required_components": [
            {"name": "capital_expenditures", "aliases": ["capital expenditures", "purchases of property, plant and equipment", "purchases of PP&E"], "preferred_statement": "cash_flow_statement"},
            {"name": "revenue", "aliases": ["revenue", "net sales", "sales"], "preferred_statement": "income_statement"},
            {"name": "fixed_assets", "aliases": ["property, plant and equipment net", "PP&E net", "fixed assets"], "preferred_statement": "balance_sheet"},
            {"name": "total_assets", "aliases": ["total assets"], "preferred_statement": "balance_sheet"},
            {"name": "net_income", "aliases": ["net income", "net earnings", "net income attributable"], "preferred_statement": "income_statement"},
        ],
        "calculation_steps": [
            "Calculate CAPEX / revenue.",
            "Calculate fixed assets / total assets.",
            "Calculate ROA = net income / total assets or average total assets if the question specifies average assets.",
            "When revenue and total assets are available, also calculate total assets / revenue as a supporting capital intensity signal, especially for telecom, utilities, energy, transportation, and other infrastructure-heavy businesses.",
            "Use the metrics together for the capital-intensive business conclusion; do not rely only on total assets / revenue unless the user specifically asks for capital intensity ratio.",
            "Answer yes/no directly when the question asks whether the company is capital-intensive. Do not leave the final conclusion as only 'moderately capital-intensive'.",
            "General heuristic: low/moderate CAPEX/revenue, low/moderate fixed-assets-to-assets, and strong ROA usually support a 'No, not clearly capital-intensive' conclusion; high total-assets-to-revenue, large fixed/infrastructure asset base, sustained high CAPEX/revenue, or weak asset returns usually support 'Yes'. Treat these as analytical heuristics, not absolute rules.",
        ],
        "suggested_searches": [
            "capital expenditures purchases of property plant equipment cash flow statement",
            "net sales revenue income statement",
            "property plant equipment net total assets balance sheet",
            "net income income statement",
            "total assets revenue capital intensity balance sheet income statement",
        ],
        "forbidden_or_risky_components": [
            {
                "term": "total assets / revenue only",
                "reason": "This is the capital intensity ratio, but the business assessment should consider capex/revenue, fixed assets/total assets, and ROA when the question asks whether the business is capital-intensive.",
                "hard_fail_if_used_directly": False,
            }
        ],
        "output_contract": {
            "format": "direct_yes_no_then_metrics",
            "unit": "mixed",
            "must_include": "Direct Yes/No conclusion followed by CAPEX/revenue, fixed assets/total assets, ROA, and total assets/revenue when available.",
        },
    },
    "current_ratio": {
        "display_name": "Current Ratio",
        "aliases": ["current ratio", "working capital ratio"],
        "formula": "total_current_assets / total_current_liabilities",
        "required_components": [
            {"name": "total_current_assets", "aliases": ["total current assets"], "preferred_statement": "balance_sheet"},
            {"name": "total_current_liabilities", "aliases": ["total current liabilities"], "preferred_statement": "balance_sheet"},
        ],
        "suggested_searches": ["total current assets total current liabilities balance sheet"],
        "forbidden_or_risky_components": [],
        "output_contract": {"format": "decimal_ratio", "unit": "x"},
    },
    "working_capital_amount": {
        "display_name": "Working Capital Amount",
        "aliases": [
            "working capital",
            "net working capital",
            "positive working capital",
            "negative working capital",
            "operating working capital",
            "cash-excluded working capital",
        ],
        "formula": "standard_working_capital = total_current_assets - total_current_liabilities; operating_working_capital = total_current_assets - cash_and_cash_equivalents - short_term_investments - total_current_liabilities",
        "required_components": [
            {"name": "total_current_assets", "aliases": ["total current assets"], "preferred_statement": "balance_sheet"},
            {"name": "total_current_liabilities", "aliases": ["total current liabilities"], "preferred_statement": "balance_sheet"},
            {"name": "cash_and_cash_equivalents", "aliases": ["cash and cash equivalents", "cash equivalents", "cash"], "preferred_statement": "balance_sheet", "optional": True},
            {"name": "short_term_investments", "aliases": ["short-term investments", "short term investments", "marketable securities current", "marketable securities"], "preferred_statement": "balance_sheet", "optional": True},
        ],
        "calculation_steps": [
            "If the user explicitly defines working capital, follow that definition exactly.",
            "If the user asks for working capital ratio/current ratio, use total current assets / total current liabilities instead of this contract.",
            "For an unconstrained plain working-capital amount, calculate standard working capital = total current assets - total current liabilities.",
            "For questions asking whether working capital is positive/negative, useful, relevant, or analytically meaningful, also calculate cash-excluded operating working capital = total current assets - cash and cash equivalents - short-term/marketable investments - total current liabilities when those cash/investment lines are available.",
            "If standard working capital and cash-excluded operating working capital lead to different analytical conclusions, make the distinction explicit and base the yes/no operating-liquidity conclusion on the cash-excluded operating measure, while still reporting standard working capital as context.",
            "For payment platforms, financial platforms, brokers, insurers, banks, or businesses with customer/custodial funds, explain that total current assets and liabilities can be distorted by pass-through customer balances; do not rely on only one broad subtotal if detailed current-asset lines are available.",
        ],
        "suggested_searches": [
            "balance sheet total current assets total current liabilities cash and cash equivalents short-term investments",
            "working capital current assets current liabilities cash equivalents short-term investments",
            "customer accounts funds receivable funds payable current assets current liabilities",
        ],
        "forbidden_or_risky_components": [
            {
                "term": "total current assets - total current liabilities only",
                "reason": "Plain standard working capital is useful context, but it can be analytically incomplete when the question asks whether working capital is useful/relevant or when cash/investment balances dominate current assets.",
                "hard_fail_if_used_directly": False,
            }
        ],
        "output_contract": {"format": "currency_or_yes_no_with_amount", "unit": "same as filing"},
    },
    "inventory_turnover": {
        "display_name": "Inventory Turnover",
        "aliases": ["inventory turnover", "sold its inventory", "converted inventory"],
        "formula": "cogs / average_inventory",
        "required_components": [
            {
                "name": "cogs",
                "aliases": ["cost of goods sold", "cost of products sold", "cost of sales", "cost of revenue"],
                "preferred_statement": "income_statement",
            },
            {
                "name": "ending_inventory",
                "aliases": ["inventories", "inventory"],
                "preferred_statement": "balance_sheet",
            },
            {
                "name": "beginning_inventory",
                "aliases": ["prior year inventories", "prior year inventory", "inventories"],
                "preferred_statement": "balance_sheet",
            },
        ],
        "calculation_steps": [
            "Find COGS / cost of sales / cost of products sold from the income statement.",
            "Find current-year and prior-year inventory from the balance sheet.",
            "Average inventory = (prior-year inventory + current-year inventory) / 2.",
            "Inventory turnover = COGS / average inventory.",
        ],
        "suggested_searches": [
            "cost of products sold cost of sales income statement",
            "inventories inventory balance sheet",
            "inventory turnover cost of sales inventories",
        ],
        "forbidden_or_risky_components": [
            {
                "term": "revenue",
                "reason": "Revenue/sales is not COGS. Use it only if COGS is not separately stated and explicitly disclose the approximation.",
                "hard_fail_if_used_directly": False,
            },
            {
                "term": "ending inventory as proxy",
                "reason": "Do not use ending inventory as average inventory if prior-year inventory is available.",
                "hard_fail_if_used_directly": True,
            },
        ],
        "output_contract": {"format": "decimal_ratio", "unit": "times"},
    },
    "roa": {
        "display_name": "Return on Assets (ROA)",
        "aliases": ["return on assets", "roa"],
        "formula": "net_income / average_total_assets",
        "required_components": [
            {
                "name": "net_income",
                "aliases": ["net income", "net earnings", "net income attributable", "net loss"],
                "preferred_statement": "income_statement",
            },
            {
                "name": "ending_total_assets",
                "aliases": ["total assets"],
                "preferred_statement": "balance_sheet",
            },
            {
                "name": "beginning_total_assets",
                "aliases": ["prior year total assets", "total assets"],
                "preferred_statement": "balance_sheet",
            },
        ],
        "calculation_steps": [
            "Average total assets = (prior-year total assets + current-year total assets) / 2.",
            "ROA = signed net income / average total assets.",
            "Parentheses in filings mean negative values.",
            "Use consolidated total assets unless the user explicitly asks for parent-company-only or segment assets.",
        ],
        "suggested_searches": [
            "consolidated balance sheets total assets current assets noncurrent assets",
            "consolidated statements of operations net income loss attributable",
            "statement of financial position total assets consolidated",
            "return on assets net income average total assets",
        ],
        "forbidden_or_risky_components": [],
        "output_contract": {"format": "decimal_ratio_unless_question_requests_percent", "unit": "ratio"},
    },
    "gross_margin": {
        "display_name": "Gross Margin",
        "aliases": ["gross margin", "gross profit margin"],
        "formula": "gross_profit / revenue",
        "required_components": [
            {"name": "revenue", "aliases": ["revenue", "net sales", "sales"], "preferred_statement": "income_statement"},
            {"name": "cogs", "aliases": ["cost of goods sold", "cost of sales", "cost of products sold"], "preferred_statement": "income_statement", "optional": True},
            {"name": "gross_profit", "aliases": ["gross profit"], "preferred_statement": "income_statement", "optional": True},
        ],
        "suggested_searches": ["revenue cost of sales gross profit income statement"],
        "forbidden_or_risky_components": [],
        "output_contract": {"format": "percent", "unit": "%"},
    },
    "operating_margin": {
        "display_name": "Operating Margin",
        "aliases": ["operating margin", "operating income margin"],
        "formula": "operating_income / revenue",
        "required_components": [
            {"name": "operating_income", "aliases": ["operating income", "income from operations", "operating profit"], "preferred_statement": "income_statement"},
            {"name": "revenue", "aliases": ["revenue", "net sales", "sales"], "preferred_statement": "income_statement"},
        ],
        "suggested_searches": ["operating income net sales income statement", "operating margin percent of net sales"],
        "forbidden_or_risky_components": [],
        "output_contract": {"format": "percent", "unit": "%"},
    },
    "net_margin": {
        "display_name": "Net Margin",
        "aliases": ["net margin", "net profit margin"],
        "formula": "net_income / revenue",
        "required_components": [
            {"name": "net_income", "aliases": ["net income", "net earnings", "net loss"], "preferred_statement": "income_statement"},
            {"name": "revenue", "aliases": ["revenue", "net sales", "sales"], "preferred_statement": "income_statement"},
        ],
        "suggested_searches": ["net income revenue income statement"],
        "forbidden_or_risky_components": [],
        "output_contract": {"format": "percent", "unit": "%"},
    },
    "fcf": {
        "display_name": "Free Cash Flow",
        "aliases": ["free cash flow", "fcf"],
        "formula": "operating_cash_flow - capital_expenditures",
        "required_components": [
            {"name": "operating_cash_flow", "aliases": ["net cash provided by operating activities", "cash flow from operations"], "preferred_statement": "cash_flow_statement"},
            {"name": "capital_expenditures", "aliases": ["capital expenditures", "purchases of property, plant and equipment", "purchases of property and equipment"], "preferred_statement": "cash_flow_statement"},
        ],
        "suggested_searches": ["net cash provided by operating activities capital expenditures cash flow statement"],
        "forbidden_or_risky_components": [],
        "output_contract": {"format": "currency", "unit": "same as filing"},
    },
    "debt_to_equity": {
        "display_name": "Debt-to-Equity",
        "aliases": ["debt to equity", "debt-to-equity"],
        "formula": "total_debt / total_shareholders_equity",
        "required_components": [
            {"name": "total_debt", "aliases": ["total debt", "short-term borrowings", "current portion of long-term debt", "long-term debt"], "preferred_statement": "balance_sheet"},
            {"name": "total_shareholders_equity", "aliases": ["total shareholders' equity", "total stockholders' equity", "shareowners' equity"], "preferred_statement": "balance_sheet"},
        ],
        "suggested_searches": ["total debt total shareholders equity balance sheet"],
        "forbidden_or_risky_components": [],
        "output_contract": {"format": "decimal_ratio", "unit": "x"},
    },
    "capital_intensity": {
        "display_name": "Capital Intensity",
        "aliases": ["capital intensity", "capital intensive", "capital-intensive"],
        "formula": "total_assets / revenue",
        "required_components": [
            {"name": "total_assets", "aliases": ["total assets"], "preferred_statement": "balance_sheet"},
            {"name": "revenue", "aliases": ["revenue", "total operating revenues", "net sales", "sales"], "preferred_statement": "income_statement"},
        ],
        "suggested_searches": ["total assets balance sheet", "total operating revenues revenue income statement"],
        "forbidden_or_risky_components": [
            {
                "term": "capital expenditures / revenue",
                "reason": "CAPEX/revenue is a supporting capital-spending metric, but not the capital intensity ratio Total Assets / Revenue.",
                "hard_fail_if_used_directly": False,
            }
        ],
        "output_contract": {"format": "decimal_ratio", "unit": "x"},
    },
}


# High-confidence direct patterns.
_DETECTION_ORDER = [
    "quick_ratio",
    "depreciation_and_amortization_margin",
    "capital_intensive_assessment",
    "current_ratio",
    "working_capital_amount",
    "inventory_turnover",
    "roa",
    "debt_to_equity",
    "capital_intensity",
    "operating_margin",
    "gross_margin",
    "net_margin",
    "fcf",
]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _has_phrase(q: str, phrase: str) -> bool:
    return re.search(r"(?<![a-z0-9])" + re.escape(phrase.lower()) + r"(?![a-z0-9])", q) is not None


def detect_metric(question: str) -> str | None:
    """
    Detect whether the question requires a formula contract.
    Conservative by design: unknown or narrative-only questions return None.
    """
    q = _norm(question)

    if re.search(r"\bwhat\s+drove\b|\bdriven\s+by\b|\bdrivers?\b|\bwhy\s+did\b", q):
        # Exception: "what drove inventory turnover" is still not a standard formula request.
        return None

    # Direct high-confidence metric cues.
    if "quick ratio" in q or "acid-test" in q:
        return "quick_ratio"
    if (
        "depreciation and amortization" in q
        or "depreciation & amortization" in q
        or re.search(r"\bd\s*&\s*a\b", q)
        or re.search(r"\bd&a\b", q)
    ) and ("margin" in q or "%" in q):
        return "depreciation_and_amortization_margin"
    if "current ratio" in q or "working capital ratio" in q:
        return "current_ratio"
    if "working capital" in q:
        return "working_capital_amount"
    if "inventory turnover" in q or "sold its inventory" in q or "converted inventory" in q:
        return "inventory_turnover"
    if "return on assets" in q or re.search(r"\broa\b", q):
        return "roa"
    if "debt-to-equity" in q or "debt to equity" in q:
        return "debt_to_equity"
    if "capital intensive" in q or "capital-intensive" in q:
        if "ratio" in q and "business" not in q and not q.startswith("is "):
            return "capital_intensity"
        return "capital_intensive_assessment"
    if "capital intensity" in q:
        return "capital_intensity"
    if "free cash flow" in q or re.search(r"\bfcf\b", q):
        return "fcf"


    formula_intent = bool(re.search(r"\b(calculate|ratio|what is|what was|how much|historically consistent|consistent)\b", q))
    if formula_intent:
        if "operating margin" in q or "operating income margin" in q:
            return "operating_margin"
        if "gross margin" in q or "gross profit margin" in q:
            return "gross_margin"
        if "net margin" in q or "net profit margin" in q:
            return "net_margin"

    return None


def get_metric_contract(metric_name: str | None) -> dict[str, Any] | None:
    if not metric_name:
        return None
    key = metric_name.lower().strip().replace(" ", "_").replace("-", "_")
    aliases = {
        "quickratio": "quick_ratio",
        "currentratio": "current_ratio",
        "working_capital": "working_capital_amount",
        "net_working_capital": "working_capital_amount",
        "operating_working_capital": "working_capital_amount",
        "inventoryturnover": "inventory_turnover",
        "return_on_assets": "roa",
        "free_cash_flow": "fcf",
        "debt_to_equity_ratio": "debt_to_equity",
        "d&a_margin": "depreciation_and_amortization_margin",
        "da_margin": "depreciation_and_amortization_margin",
        "capital_intensive": "capital_intensive_assessment",
    }
    key = aliases.get(key, key)
    if key in METRIC_CONTRACTS:
        contract = dict(METRIC_CONTRACTS[key])
        contract["metric_id"] = key
        return contract

    # Best-effort alias matching for tool calls like get_financial_formula("Quick Ratio").
    q = _norm(metric_name)
    for metric_id in _DETECTION_ORDER:
        c = METRIC_CONTRACTS[metric_id]
        if any(alias in q for alias in c.get("aliases", [])):
            contract = dict(c)
            contract["metric_id"] = metric_id
            return contract
    return None


def compact_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Return only fields useful for the LLM prompt/trace."""
    return {
        "metric_id": contract.get("metric_id"),
        "display_name": contract.get("display_name"),
        "formula": contract.get("formula"),
        "required_components": contract.get("required_components", []),
        "calculation_steps": contract.get("calculation_steps", []),
        "suggested_searches": contract.get("suggested_searches", []),
        "forbidden_or_risky_components": contract.get("forbidden_or_risky_components", []),
        "output_contract": contract.get("output_contract", {}),
    }


def render_contract_for_prompt(contract: dict[str, Any]) -> str:
    compact = compact_contract(contract)
    return (
        "A financial metric contract applies to this question. Follow it exactly.\n"
        "Do not calculate until every required component is retrieved from the filing or explicitly unavailable.\n"
        "Use the suggested searches to find the right statements/tables.\n"
        "Do not use forbidden/risky substitutes unless the contract explicitly allows approximation.\n\n"
        + json.dumps(compact, ensure_ascii=False, indent=2)
    )


def _iter_assistant_tool_queries(messages: list[dict]) -> list[str]:
    queries: list[str] = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls", []) or []:
            func = (tc.get("function") or {})
            if func.get("name") != "execute_sql":
                continue
            raw_args = func.get("arguments") or "{}"
            try:
                args = json.loads(raw_args)
                q = args.get("query")
                if q:
                    queries.append(str(q))
            except Exception:
                queries.append(str(raw_args))
    return queries


def _negated_or_subtracted_context(text: str, term: str) -> bool:
    """Detect cases where a risky component is mentioned as excluded/subtracted, not used directly."""
    t = _norm(text)
    idx = t.find(term.lower())
    if idx < 0:
        return False
    window = t[max(0, idx - 80): idx + len(term) + 120]
    return any(x in window for x in ["do not use", "not use", "subtract", "less inventories", "excluding", "minus"])


def validate_metric_answer(contract: dict[str, Any], messages: list[dict], final_answer: str, question: str = "") -> ValidationResult:
    """
    Conservative validation: only hard-fail obvious repeated failure modes.
    This is not a judge; it is a guardrail against formula violations.
    """
    metric_id = contract.get("metric_id")
    answer_l = _norm(final_answer)
    question_l = _norm(question)
    sql_queries_l = "\n".join(_iter_assistant_tool_queries(messages)).lower()

    if metric_id == "quick_ratio":
        direct_formula_in_answer = bool(re.search(
            r"total current assets\s*/\s*total current liabilities|total current assets.+total current liabilities",
            answer_l,
        ))
        direct_sql_lookup = "total current assets" in sql_queries_l and not any(
            x in sql_queries_l for x in ["inventor", "prepaid", "cash", "receivable"]
        )
        used_classic_formula_without_inventory = (
            ("cash and cash equivalents" in answer_l or "cash equivalents" in answer_l)
            and ("accounts receivable" in answer_l or "receivables" in answer_l)
            and "total current liabilities" in answer_l
            and "total current assets" not in answer_l
        )
        inventory_assumed_zero = "inventories" in answer_l and any(
            phrase in answer_l for phrase in ["assumed to be zero", "assumed zero", "not explicitly listed, assumed"]
        )

        if inventory_assumed_zero or used_classic_formula_without_inventory:
            return ValidationResult(
                ok=False,
                reason="The answer used a classic cash+receivables quick-assets formula or assumed inventories were zero instead of following the filing/dataset quick-ratio contract.",
                repair_instruction=(
                    "Formula validation failed for QUICK RATIO. Use the filing balance sheet lines: "
                    "Total current assets, Total inventories, and Total current liabilities. "
                    "Calculate (Total current assets - Total inventories) / Total current liabilities. "
                    "Do not assume inventories are zero and do not switch to cash+receivables unless the user explicitly asks for that variant."
                ),
            )

        if (direct_formula_in_answer or direct_sql_lookup) and not _negated_or_subtracted_context(final_answer, "total current assets"):
            return ValidationResult(
                ok=False,
                reason="The answer appears to use Total Current Assets as the quick-ratio numerator, which is current ratio, not quick ratio.",
                repair_instruction=(
                    "Formula validation failed for QUICK RATIO. Do not use Total Current Assets directly. "
                    "Retrieve Total current assets, Total inventories, and Total current liabilities. "
                    "Then calculate quick ratio as (Total current assets - Total inventories) / Total current liabilities."
                ),
            )

    if metric_id == "capital_intensive_assessment":
        used_only_capital_intensity_ratio = (
            "total assets" in answer_l
            and "revenue" in answer_l
            and not any(term in answer_l for term in ["capex", "capital expenditure", "fixed assets", "property, plant", "roa", "return on assets"])
        )
        if used_only_capital_intensity_ratio:
            return ValidationResult(
                ok=False,
                reason="The answer appears to use only Total assets / Revenue for a capital-intensive business assessment.",
                repair_instruction=(
                    "Formula validation failed for CAPITAL-INTENSIVE BUSINESS ASSESSMENT. "
                    "Retrieve and discuss CAPEX/revenue, fixed assets/total assets, and ROA before concluding whether the business is capital-intensive."
                ),
            )

        asks_direct_yes_no = question_l.startswith("is ") or " is " in question_l[:80]
        ambiguous_capital_conclusion = (
            "moderately capital-intensive" in answer_l
            or "moderately capital intensive" in answer_l
            or "not extremely capital-intensive" in answer_l
            or "not extremely capital intensive" in answer_l
        )
        has_direct_yes_no = bool(re.search(r"\b(yes|no)\b", answer_l[:300]))
        if asks_direct_yes_no and ambiguous_capital_conclusion and not has_direct_yes_no:
            return ValidationResult(
                ok=False,
                reason="The answer gives a hedged capital-intensive conclusion instead of directly answering the yes/no question.",
                repair_instruction=(
                    "Formula validation failed for CAPITAL-INTENSIVE BUSINESS ASSESSMENT. Keep the metric discussion, but start with a direct Yes or No. "
                    "Use CAPEX/revenue, fixed assets/total assets, ROA, and total assets/revenue when available. "
                    "Do not make the final conclusion only 'moderately capital-intensive'; choose the better-supported side and explain any nuance after the direct answer."
                ),
            )

    if metric_id == "working_capital_amount":
        asks_analytical_wc = any(
            phrase in question_l
            for phrase in [
                "positive working capital",
                "negative working capital",
                "working capital positive",
                "working capital negative",
                "useful",
                "relevant",
                "liquidity",
                "healthy",
            ]
        )
        mentions_standard_wc = (
            "total current assets" in answer_l
            and "total current liabilities" in answer_l
            and ("working capital" in answer_l or "current assets - current liabilities" in answer_l)
        )
        mentions_cash_excluded_wc = any(
            phrase in answer_l
            for phrase in [
                "cash-excluded",
                "excluding cash",
                "exclude cash",
                "non-cash current assets",
                "operating working capital",
                "short-term investments",
                "marketable securities",
            ]
        )

        if asks_analytical_wc and mentions_standard_wc and not mentions_cash_excluded_wc:
            return ValidationResult(
                ok=False,
                reason="The answer relies only on standard working capital but the question asks for an analytical working-capital/liquidity conclusion.",
                repair_instruction=(
                    "Formula validation failed for WORKING CAPITAL AMOUNT. Keep standard working capital as context, but also retrieve cash and cash equivalents and short-term/marketable investments when available. "
                    "Compute cash-excluded operating working capital = total current assets - cash and cash equivalents - short-term/marketable investments - total current liabilities. "
                    "If the company has customer/custodial funds or payment-platform balances, explain that broad current-asset/current-liability subtotals can be analytically noisy. "
                    "Give a direct positive/negative conclusion using the operating measure and cite every input."
                ),
            )

    if metric_id == "depreciation_and_amortization_margin":
        used_accumulated_da = (
            "accumulated depreciation and amortization" in answer_l
            or "accumulated depreciation and amortization" in sql_queries_l
        )
        negative_da_margin = bool(re.search(r"-\s*\d+(?:\.\d+)?\s*%", final_answer))
        if used_accumulated_da or negative_da_margin:
            return ValidationResult(
                ok=False,
                reason="The answer appears to use accumulated depreciation/amortization or a negative contra-asset value instead of period D&A expense.",
                repair_instruction=(
                    "Formula validation failed for D&A MARGIN. Use the cash flow statement line exactly named "
                    "'Depreciation and amortization' for the period D&A expense, not 'Accumulated depreciation and amortization' from PP&E notes. "
                    "Then divide that positive D&A expense by revenue/net revenue from the income statement and return the percent margin."
                ),
            )

    if metric_id == "inventory_turnover":
        if "ending inventory" in answer_l and "proxy" in answer_l:
            return ValidationResult(
                ok=False,
                reason="The answer used ending inventory as a proxy for average inventory.",
                repair_instruction=(
                    "Formula validation failed for INVENTORY TURNOVER. Do not use ending inventory as a proxy if prior-year inventory can be retrieved. "
                    "Find current-year and prior-year inventory from the balance sheet, compute average inventory, then divide COGS/cost of sales by average inventory."
                ),
            )
        if re.search(r"(revenue|sales)\s+(?:as|for)\s+(?:cogs|cost of goods sold|cost of sales)", answer_l):
            return ValidationResult(
                ok=False,
                reason="The answer appears to use revenue/sales as COGS.",
                repair_instruction=(
                    "Formula validation failed for INVENTORY TURNOVER. Revenue/sales is not COGS. Search for cost of products sold, cost of sales, cost of goods sold, or cost of revenue. "
                    "Only use revenue as a proxy if no cost line exists and explicitly state the approximation."
                ),
            )

    if metric_id == "roa":
        if any(term in answer_l for term in ["parent company", "parent-only", "standalone parent"]):
            return ValidationResult(
                ok=False,
                reason="ROA appears to use parent-company-only or standalone parent balance sheet data instead of consolidated total assets.",
                repair_instruction=(
                    "Formula validation failed for ROA source selection. Use consolidated total assets from the consolidated balance sheet/statement of financial position, "
                    "not parent-company-only schedules or subsidiary schedules, unless the user explicitly asks for parent-only ROA."
                ),
            )

        # If the user explicitly asks to round to two decimals without saying percent,
        # prefer decimal ratio (e.g. -0.02), not only percent (e.g. -2%).
        asks_decimal = "round" in question_l and "percent" not in question_l and "%" not in question_l
        answer_has_percent = "%" in final_answer
        # If there is no decimal-looking ROA ratio near the final answer, ask for conversion.
        has_decimal_ratio = bool(re.search(r"[-+]?0\.\d{1,4}\b", answer_l))
        if asks_decimal and answer_has_percent and not has_decimal_ratio:
            return ValidationResult(
                ok=False,
                reason="ROA output appears to be in percent while the question asks for a rounded decimal ratio.",
                repair_instruction=(
                    "Formula validation failed for ROA output format. The question asks for a rounded answer, not a percent. "
                    "Calculate net income / average total assets and return the decimal ratio rounded as requested. You may mention the percent only after the decimal answer."
                ),
            )

    # Generic required component sanity check: do not hard-fail, but repair if the
    # final answer clearly contains no calculation and no key component names.
    if metric_id and "could not" not in answer_l:
        required_names = [c.get("name", "") for c in contract.get("required_components", []) if not c.get("optional")]
        if required_names and "formula" in answer_l and not any(n.replace("_", " ") in answer_l for n in required_names):
            return ValidationResult(
                ok=False,
                reason="The answer discusses the formula but does not show the required retrieved components.",
                repair_instruction=(
                    f"Formula validation failed for {contract.get('display_name')}. Show the required components from the filing before calculating: "
                    + ", ".join(required_names)
                ),
            )

    return ValidationResult(ok=True)
