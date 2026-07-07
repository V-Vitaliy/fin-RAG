from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import re
import statistics
import time
import uuid
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.infrastructure.agent import build_agent_use_case
from app.infrastructure.duckdb import DuckDBManager
from app.infrastructure.openai import build_openai_client
from app.infrastructure.postgres import build_postgres_engine, build_sessionmaker
from app.infrastructure.qdrant import build_qdrant_client
from app.infrastructure.rag import (
    build_dense_embedder,
    build_reranker,
    build_sparse_embedder,
)
from app.infrastructure.s3storage import s3session
from app.repositories.duckdb_repo import DuckDBTableRepository
from app.repositories.qdrant_repo import VectorIndexRepository
from app.repositories.s3storage_repository import S3StorageRepository
from app.repositories.uow import SqlAlchemyUnitOfWork
from app.services.retrieval.hybrid import AgentHybridRetriever
from app.use_cases.retrieval import RetrieveDocumentsUseCase


MARKER_RE = re.compile(r"\[(?:C|T)\d+\]")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")
        file.flush()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def to_jsonable(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, uuid.UUID):
        return str(value)

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, Path):
        return str(value)

    if hasattr(value, "model_dump"):
        return to_jsonable(value.model_dump())

    if hasattr(value, "dict"):
        return to_jsonable(value.dict())

    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]

    if hasattr(value, "__dict__"):
        return {
            key: to_jsonable(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }

    return str(value)


def load_dataset(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(raw, list):
        raise ValueError(f"Dataset must be a JSON list: {path}")

    return raw


def load_document_id_map(path: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(raw, dict):
        raise ValueError(f"Document id map must be a JSON object: {path}")

    return raw


def extract_case_id(case: dict[str, Any], fallback_index: int) -> str:
    return str(
        case.get("financebench_id")
        or case.get("id")
        or case.get("case_id")
        or f"case_{fallback_index:04d}"
    )


def extract_expected_pages(case: dict[str, Any]) -> list[int]:
    pages: list[int] = []

    def add_page(value: Any) -> None:
        if value is None:
            return

        if isinstance(value, bool):
            return

        if isinstance(value, int):
            pages.append(value)
            return

        if isinstance(value, float) and value.is_integer():
            pages.append(int(value))
            return

        if isinstance(value, str):
            for match in re.findall(r"\d+", value):
                pages.append(int(match))

    evidence = case.get("evidence")

    if isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, dict):
                for key in [
                    "evidence_page_num",
                    "page",
                    "page_num",
                    "page_number",
                    "doc_page",
                ]:
                    add_page(item.get(key))
            else:
                add_page(item)
    elif isinstance(evidence, dict):
        for key in [
            "evidence_page_num",
            "page",
            "page_num",
            "page_number",
            "doc_page",
        ]:
            add_page(evidence.get(key))

    for key in [
        "evidence_page_num",
        "page",
        "page_num",
        "page_number",
        "doc_page",
    ]:
        add_page(case.get(key))

    return sorted(set(pages))


def extract_evidence_texts(case: dict[str, Any]) -> list[str]:
    texts: list[str] = []

    evidence = case.get("evidence")

    def add_text(value: Any) -> None:
        if isinstance(value, str) and value.strip():
            texts.append(value.strip())

    if isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, dict):
                for key in ["evidence_text", "text", "content", "excerpt"]:
                    add_text(item.get(key))
            else:
                add_text(item)
    elif isinstance(evidence, dict):
        for key in ["evidence_text", "text", "content", "excerpt"]:
            add_text(evidence.get(key))

    justification = case.get("justification")
    if isinstance(justification, str) and justification.strip():
        texts.append(justification.strip())

    return texts


def result_get(result: Any, key: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(key, default)

    return getattr(result, key, default)


def source_blob(citations: Any, source_documents: Any) -> str:
    return json.dumps(
        to_jsonable(
            {
                "citations": citations,
                "source_documents": source_documents,
            }
        ),
        ensure_ascii=False,
    ).lower()


def compute_deterministic_metrics(
    *,
    answer: str,
    expected_doc_name: str,
    expected_document_id: str | None,
    expected_pages: list[int],
    citations: Any,
    source_documents: Any,
) -> dict[str, Any]:
    blob = source_blob(citations, source_documents)

    has_citation_marker = bool(MARKER_RE.search(answer or ""))
    citations_count = len(citations) if isinstance(citations, list) else 0
    source_documents_count = (
        len(source_documents) if isinstance(source_documents, list) else 0
    )

    expected_doc_norm = expected_doc_name.lower()
    correct_document_cited = expected_doc_norm in blob

    if expected_document_id:
        correct_document_cited = correct_document_cited or expected_document_id.lower() in blob

    expected_page_hit = False
    expected_page_hit_tolerance_1 = False

    for page in expected_pages:
        page_tokens = {
            f'"page": {page}',
            f'"page": "{page}"',
            f"p.{page}",
            f"page {page}",
            f"page_number\": {page}",
            f"page_number\": \"{page}\"",
        }

        if any(token.lower() in blob for token in page_tokens):
            expected_page_hit = True

        tolerance_pages = {page - 1, page, page + 1}
        for tol_page in tolerance_pages:
            if tol_page <= 0:
                continue

            tolerance_tokens = {
                f'"page": {tol_page}',
                f'"page": "{tol_page}"',
                f"p.{tol_page}",
                f"page {tol_page}",
                f"page_number\": {tol_page}",
                f"page_number\": \"{tol_page}\"",
            }

            if any(token.lower() in blob for token in tolerance_tokens):
                expected_page_hit_tolerance_1 = True

    return {
        "has_answer": bool((answer or "").strip()),
        "has_citation_marker": has_citation_marker,
        "citations_count": citations_count,
        "source_documents_count": source_documents_count,
        "correct_document_cited": correct_document_cited,
        "expected_pages": expected_pages,
        "expected_page_hit": expected_page_hit,
        "expected_page_hit_tolerance_1": expected_page_hit_tolerance_1,
    }


def summarize_answer_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    ok_rows = [row for row in rows if row.get("status") == "OK"]

    def rate(predicate) -> float:
        if not total:
            return 0.0
        return sum(1 for row in rows if predicate(row)) / total

    latencies = [
        row["latency_seconds"]
        for row in ok_rows
        if isinstance(row.get("latency_seconds"), (int, float))
    ]

    judge_rows = [
        row.get("judge") or {}
        for row in rows
        if isinstance(row.get("judge"), dict)
    ]

    def avg_judge_score(key: str) -> float | None:
        values = [
            item.get(key)
            for item in judge_rows
            if isinstance(item.get(key), (int, float))
        ]
        if not values:
            return None
        return round(sum(values) / len(values), 4)

    return {
        "cases_total": total,
        "runtime_success_count": len(ok_rows),
        "runtime_success_rate": round(len(ok_rows) / total, 4) if total else 0.0,
        "avg_latency_seconds": round(sum(latencies) / len(latencies), 4)
        if latencies
        else None,
        "median_latency_seconds": round(statistics.median(latencies), 4)
        if latencies
        else None,
        "citation_marker_rate": round(
            rate(lambda row: row.get("deterministic_metrics", {}).get("has_citation_marker")),
            4,
        ),
        "correct_document_citation_rate": round(
            rate(lambda row: row.get("deterministic_metrics", {}).get("correct_document_cited")),
            4,
        ),
        "expected_page_hit_rate": round(
            rate(lambda row: row.get("deterministic_metrics", {}).get("expected_page_hit")),
            4,
        ),
        "expected_page_hit_tolerance_1_rate": round(
            rate(
                lambda row: row.get("deterministic_metrics", {}).get(
                    "expected_page_hit_tolerance_1"
                )
            ),
            4,
        ),
        "avg_answer_correctness_score": avg_judge_score("answer_correctness_score"),
        "avg_faithfulness_score": avg_judge_score("faithfulness_score"),
        "avg_citation_support_score": avg_judge_score("citation_support_score"),
        "avg_calculation_quality_score": avg_judge_score("calculation_quality_score"),
        "avg_completeness_score": avg_judge_score("completeness_score"),
        "overall_pass_rate": round(
            rate(lambda row: row.get("judge", {}).get("overall_pass") is True),
            4,
        ),
    }


async def call_agent(
    *,
    agent_use_case: Any,
    question: str,
    workspace_id: uuid.UUID,
    document_ids: list[uuid.UUID],
) -> Any:
    method_candidates = [
        "ask",
        "run",
        "execute",
        "answer",
        "generate_answer",
    ]

    last_error: Exception | None = None

    for method_name in method_candidates:
        method = getattr(agent_use_case, method_name, None)

        if method is None:
            continue

        signature = inspect.signature(method)
        kwargs: dict[str, Any] = {}

        for param_name in signature.parameters:
            if param_name in {"self", "args", "kwargs"}:
                continue

            if param_name in {"question", "query", "user_question"}:
                kwargs[param_name] = question
            elif param_name == "workspace_id":
                kwargs[param_name] = workspace_id
            elif param_name == "document_ids":
                kwargs[param_name] = document_ids
            elif param_name == "requested_document_ids":
                kwargs[param_name] = document_ids

        try:
            result = method(**kwargs)

            if inspect.isawaitable(result):
                result = await result

            return result

        except TypeError as exc:
            last_error = exc
            continue

    raise RuntimeError(
        "Could not call agent use case. Tried methods: "
        f"{method_candidates}. Last error: {last_error!r}"
    )


async def call_judge(
    *,
    openai_client: Any,
    model: str,
    case: dict[str, Any],
    actual_answer: str,
    citations: Any,
    source_documents: Any,
    tool_calls: Any,
) -> dict[str, Any]:
    expected_answer = case.get("answer")
    justification = case.get("justification")
    evidence = case.get("evidence")

    system_prompt = (
        "You are a strict financial QA evaluator. "
        "Evaluate whether the actual answer is correct, faithful to the evidence, "
        "and properly supported by citations. Return only valid JSON."
    )

    user_payload = {
        "question": case.get("question"),
        "expected_answer": expected_answer,
        "expected_justification": justification,
        "expected_evidence": evidence,
        "actual_answer": actual_answer,
        "citations": to_jsonable(citations),
        "source_documents": to_jsonable(source_documents),
        "tool_calls": to_jsonable(tool_calls),
        "scoring_schema": {
            "answer_correctness_score": "integer 1-5",
            "faithfulness_score": "integer 1-5",
            "citation_support_score": "integer 1-5",
            "calculation_quality_score": "integer 1-5 or null if no calculation",
            "completeness_score": "integer 1-5",
            "overall_pass": "boolean",
            "failure_reason": "string or null",
            "missing_key_facts": "list of strings",
            "unsupported_claims": "list of strings",
        },
    }

    response = await openai_client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False),
            },
        ],
    )

    content = response.choices[0].message.content or "{}"

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = {
            "answer_correctness_score": None,
            "faithfulness_score": None,
            "citation_support_score": None,
            "calculation_quality_score": None,
            "completeness_score": None,
            "overall_pass": False,
            "failure_reason": "Judge returned invalid JSON.",
            "raw_response": content,
        }

    usage = getattr(response, "usage", None)

    parsed["_judge_metadata"] = {
        "model": model,
        "usage": to_jsonable(usage),
    }

    return parsed


async def run_answer_eval(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.dataset)
    document_id_map = load_document_id_map(args.document_map)

    workspace_id = uuid.UUID(str(args.workspace_id or settings.RAG_GLOBAL_WORKSPACE_ID))

    run_dir = args.eval_root / args.run_id
    answer_dir = run_dir / "answer_eval"
    traces_dir = run_dir / "traces"

    answer_results_path = answer_dir / "answer_results.jsonl"
    judge_results_path = answer_dir / "judge_results.jsonl"
    answer_summary_path = answer_dir / "answer_summary.json"

    run_dir.mkdir(parents=True, exist_ok=True)
    answer_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        answer_dir / "answer_eval_config.json",
        {
            "run_id": args.run_id,
            "created_at": utc_now_iso(),
            "dataset": str(args.dataset),
            "document_map": str(args.document_map),
            "workspace_id": str(workspace_id),
            "judge_enabled": not args.no_judge,
            "judge_model": args.judge_model,
            "cases_total": len(dataset),
        },
    )

    postgres_engine = build_postgres_engine()
    db_sessionmaker = build_sessionmaker(postgres_engine)

    rows: list[dict[str, Any]] = []

    async with AsyncExitStack() as stack:
        stack.push_async_callback(postgres_engine.dispose)

        uow_factory = lambda: SqlAlchemyUnitOfWork(db_sessionmaker)

        openai_client = build_openai_client()
        stack.push_async_callback(openai_client.close)

        s3_client = await stack.enter_async_context(
            s3session.client(
                service_name="s3",
                endpoint_url=settings.S3_ENDPOINT_URL,
                aws_access_key_id=settings.S3_ACCESS_KEY,
                aws_secret_access_key=settings.S3_SECRET_KEY,
            )
        )

        s3_repo = S3StorageRepository(
            client=s3_client,
            bucket_name=settings.S3_BUCKET_NAME,
        )

        qdrant_client = build_qdrant_client()
        stack.push_async_callback(qdrant_client.close)

        vector_repo = VectorIndexRepository(
            client=qdrant_client,
            collection_name=settings.RAG_COLLECTION_NAME,
        )

        duckdb_manager = DuckDBManager(path=settings.RAG_DUCKDB_PATH)
        duckdb_repo = DuckDBTableRepository(duckdb_manager)

        dense_embedder = build_dense_embedder()
        sparse_embedder = build_sparse_embedder()
        reranker = build_reranker()

        retriever = AgentHybridRetriever(
            vector_repo=vector_repo,
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            reranker=reranker,
            prefetch_min=settings.RAG_PREFETCH_MIN,
            fusion_min=settings.RAG_FUSION_MIN,
            first_stage_multiplier=settings.RAG_FIRST_STAGE_MULTIPLIER,
            fusion_multiplier=settings.RAG_FUSION_MULTIPLIER,
        )

        retrieval_use_case = RetrieveDocumentsUseCase(
            uow_factory=uow_factory,
            retriever=retriever,
            global_workspace_id=settings.RAG_GLOBAL_WORKSPACE_ID,
        )

        agent_use_case = build_agent_use_case(
            openai_client=openai_client,
            retrieval_use_case=retrieval_use_case,
            uow_factory=uow_factory,
            s3_repository=s3_repo,
        )

        for index, case in enumerate(dataset, start=1):
            case_id = extract_case_id(case, index)
            doc_name = str(case.get("doc_name") or "").strip()
            question = str(case.get("question") or "").strip()
            expected_pages = extract_expected_pages(case)
            expected_evidence_texts = extract_evidence_texts(case)

            trace_path = traces_dir / f"{case_id}.jsonl"

            base = {
                "run_id": args.run_id,
                "case_index": index,
                "cases_total": len(dataset),
                "case_id": case_id,
                "doc_name": doc_name,
                "question": question,
                "question_type": case.get("question_type"),
                "question_reasoning": case.get("question_reasoning"),
                "doc_type": case.get("doc_type"),
                "expected_answer": case.get("answer"),
                "expected_justification": case.get("justification"),
                "expected_pages": expected_pages,
                "expected_evidence_texts": expected_evidence_texts,
                "started_at": utc_now_iso(),
            }

            print(f"[{index}/{len(dataset)}] {case_id} | {doc_name}")

            doc_map_item = document_id_map.get(doc_name)

            if not doc_map_item:
                row = {
                    **base,
                    "status": "FAILED",
                    "error": f"doc_name not found in document map: {doc_name}",
                }
                rows.append(row)
                append_jsonl(answer_results_path, row)
                append_jsonl(trace_path, {"type": "error", "error": row["error"]})
                continue

            document_id = uuid.UUID(str(doc_map_item["document_id"]))

            append_jsonl(
                trace_path,
                {
                    "type": "case_started",
                    "case_id": case_id,
                    "doc_name": doc_name,
                    "document_id": str(document_id),
                    "question": question,
                    "timestamp": utc_now_iso(),
                },
            )

            started = time.perf_counter()

            try:
                result = await call_agent(
                    agent_use_case=agent_use_case,
                    question=question,
                    workspace_id=workspace_id,
                    document_ids=[document_id],
                )

                latency = round(time.perf_counter() - started, 4)

                answer = result_get(result, "answer", "") or ""
                tool_calls = result_get(result, "tool_calls", []) or []
                citations = result_get(result, "citations", []) or []
                source_documents = result_get(result, "source_documents", []) or []

                deterministic_metrics = compute_deterministic_metrics(
                    answer=answer,
                    expected_doc_name=doc_name,
                    expected_document_id=str(document_id),
                    expected_pages=expected_pages,
                    citations=citations,
                    source_documents=source_documents,
                )

                append_jsonl(
                    trace_path,
                    {
                        "type": "agent_result",
                        "case_id": case_id,
                        "answer": answer,
                        "tool_calls": tool_calls,
                        "citations": citations,
                        "source_documents": source_documents,
                        "latency_seconds": latency,
                        "timestamp": utc_now_iso(),
                    },
                )

                judge_result: dict[str, Any] | None = None

                if not args.no_judge:
                    judge_started = time.perf_counter()

                    judge_result = await call_judge(
                        openai_client=openai_client,
                        model=args.judge_model,
                        case=case,
                        actual_answer=answer,
                        citations=citations,
                        source_documents=source_documents,
                        tool_calls=tool_calls,
                    )

                    judge_result["_judge_metadata"]["latency_seconds"] = round(
                        time.perf_counter() - judge_started,
                        4,
                    )

                    append_jsonl(
                        judge_results_path,
                        {
                            "run_id": args.run_id,
                            "case_id": case_id,
                            "doc_name": doc_name,
                            "judge": judge_result,
                        },
                    )

                    append_jsonl(
                        trace_path,
                        {
                            "type": "judge_result",
                            "case_id": case_id,
                            "judge": judge_result,
                            "timestamp": utc_now_iso(),
                        },
                    )

                row = {
                    **base,
                    "status": "OK",
                    "document_id": str(document_id),
                    "latency_seconds": latency,
                    "actual_answer": answer,
                    "tool_calls": tool_calls,
                    "citations": citations,
                    "source_documents": source_documents,
                    "deterministic_metrics": deterministic_metrics,
                    "judge": judge_result,
                    "error": None,
                    "finished_at": utc_now_iso(),
                }

                rows.append(row)
                append_jsonl(answer_results_path, row)

            except Exception as exc:
                latency = round(time.perf_counter() - started, 4)

                row = {
                    **base,
                    "status": "FAILED",
                    "document_id": str(document_id),
                    "latency_seconds": latency,
                    "error": repr(exc),
                    "finished_at": utc_now_iso(),
                }

                rows.append(row)
                append_jsonl(answer_results_path, row)
                append_jsonl(
                    trace_path,
                    {
                        "type": "error",
                        "case_id": case_id,
                        "error": repr(exc),
                        "timestamp": utc_now_iso(),
                    },
                )

                print(f"FAILED {case_id}: {exc!r}")

                if args.stop_on_error:
                    break

    summary = summarize_answer_results(rows)
    summary["run_id"] = args.run_id
    summary["finished_at"] = utc_now_iso()
    summary["answer_results_path"] = str(answer_results_path)
    summary["judge_results_path"] = str(judge_results_path)
    summary["traces_dir"] = str(traces_dir)

    write_json(answer_summary_path, summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    return 1 if any(row.get("status") != "OK" for row in rows) else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run answer evaluation for FinanceBench-style dataset."
    )

    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--document-map", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--eval-root", type=Path, default=Path("/data/evals"))
    parser.add_argument("--workspace-id", default=None)
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return asyncio.run(run_answer_eval(args))


if __name__ == "__main__":
    raise SystemExit(main())