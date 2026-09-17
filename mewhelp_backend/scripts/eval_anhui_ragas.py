from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.anhui_main import ChatRequest, harness  # noqa: E402


DATASET_PATH = ROOT / "tests" / "data" / "anhui_ragas_eval.jsonl"
OUTPUT_DIR = ROOT / "reports"
ANSWERS_PATH = OUTPUT_DIR / "anhui_ragas_answers.jsonl"
REPORT_PATH = OUTPUT_DIR / "anhui_ragas_report.csv"


def _load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def _collect_answers(cases: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    original_key = os.environ.get("MODEL_API_KEY")
    if os.getenv("RAGAS_COLLECT_WITHOUT_LLM", "1") == "1":
        os.environ["MODEL_API_KEY"] = ""
    try:
        for case in cases:
            result = harness.run(
                ChatRequest(
                    query=case["question"],
                    top_k=top_k,
                    session_id=f"ragas_{case['id']}",
                    user_id="ragas-evaluator",
                )
            )
            contexts = []
            for item in result.get("results", []):
                parts = [
                    item.get("name", ""),
                    item.get("track", ""),
                    item.get("technology", ""),
                    item.get("stage", ""),
                    item.get("summary", ""),
                    item.get("core_technology", ""),
                    item.get("reason", ""),
                ]
                contexts.append("\n".join(str(part) for part in parts if part))
            rows.append(
                {
                    "id": case["id"],
                    "question": case["question"],
                    "answer": result.get("answer", ""),
                    "contexts": contexts,
                    "ground_truth": case.get("reference", ""),
                    "reference": case.get("reference", ""),
                    "retrieval_mode": result.get("retrieval_mode"),
                    "answer_mode": result.get("answer_mode"),
                    "result_count": len(result.get("results", [])),
                }
            )
    finally:
        if original_key is None:
            os.environ.pop("MODEL_API_KEY", None)
        else:
            os.environ["MODEL_API_KEY"] = original_key
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run_ragas(rows: list[dict[str, Any]]) -> None:
    judge_api_key = os.getenv("RAGAS_JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("MODEL_API_KEY")
    judge_base_url = os.getenv("RAGAS_JUDGE_BASE_URL") or os.getenv("OPENAI_BASE_URL") or os.getenv("MODEL_BASE_URL")
    judge_model = os.getenv("RAGAS_JUDGE_MODEL", os.getenv("CHAT_MODEL", "qwen-plus"))
    embedding_model = os.getenv("RAGAS_EMBEDDING_MODEL", os.getenv("EMBEDDING_MODEL", "text-embedding-v4"))

    if not judge_api_key:
        raise RuntimeError("RAGAS needs RAGAS_JUDGE_API_KEY, OPENAI_API_KEY, or MODEL_API_KEY.")

    from datasets import Dataset
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    dataset_rows = [
        {
            "question": row["question"],
            "answer": row["answer"],
            "contexts": row["contexts"],
            "ground_truth": row["ground_truth"],
        }
        for row in rows
        if row.get("contexts")
    ]
    dataset = Dataset.from_list(dataset_rows)
    judge_llm = ChatOpenAI(
        model=judge_model,
        temperature=0,
        api_key=judge_api_key,
        base_url=judge_base_url,
    )
    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        api_key=judge_api_key,
        base_url=judge_base_url,
    )
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=judge_llm,
        embeddings=embeddings,
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_pandas().to_csv(REPORT_PATH, index=False, encoding="utf-8-sig")
    print(result)
    print(f"RAGAS report saved: {REPORT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Anhui project RAG recommendations with RAGAS.")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()

    cases = _load_cases(args.dataset)
    rows = _collect_answers(cases, args.top_k)
    _write_jsonl(ANSWERS_PATH, rows)
    print(f"Collected {len(rows)} answers: {ANSWERS_PATH}")
    if not args.collect_only:
        _run_ragas(rows)


if __name__ == "__main__":
    main()

