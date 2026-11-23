import asyncio
import json
import re
from typing import List
from app.agent import orchestrator
from app.llm_client import llm_client
from app.schemas import EvalCase, EvalResult, ResearchResponse


BENCHMARK_DATASET: List[EvalCase] = [
    EvalCase(
        id="EVAL-01",
        query="Compare PostgreSQL with pgvector vs dedicated vector databases like Qdrant for a 5M-vector RAG workload, and recommend the best option for a small team.",
        expected_keywords=["pgvector", "Qdrant", "latency", "operational"],
        min_citations=2,
        reference_answer="PostgreSQL with pgvector simplifies stack operations for <10M vectors with strong ACID guarantees, while Qdrant offers higher throughput and lower p99 latency at scale. For a small team at 5M vectors, pgvector is usually recommended unless ultra-low latency filtering is critical.",
    ),
    EvalCase(
        id="EVAL-02",
        query="What are the core architectural differences between Model Context Protocol (MCP) and traditional OpenAI function calling?",
        expected_keywords=["MCP", "client", "server", "standard"],
        min_citations=2,
        reference_answer="MCP standardizes tool, resource, and prompt discovery over a client-server protocol (stdio/HTTP) decoupled from any single LLM vendor, whereas traditional function calling binds tool schemas directly inside model-specific chat completion payloads.",
    ),
]


def compute_citation_correctness(resp: ResearchResponse) -> float:
    """
    Checks:
    1. All inline [C#] tags reference valid retrieved citation IDs.
    2. At least one valid citation is used in final_answer or recommendation.
    """
    combined_text = f"{resp.final_answer}\n{resp.decision_recommendation}"
    used_ids = set(re.findall(r"\[(C\d+)\]", combined_text))
    valid_ids = {c.id for c in resp.citations}

    if not used_ids:
        return 0.0
    hallucinated = used_ids - valid_ids
    precision = (len(used_ids) - len(hallucinated)) / len(used_ids)
    return round(precision, 3)


def compute_evidence_coverage(resp: ResearchResponse, expected_keywords: List[str]) -> float:
    combined = f"{resp.final_answer} {resp.decision_recommendation}".lower()
    if not expected_keywords:
        return 1.0
    hits = sum(1 for kw in expected_keywords if kw.lower() in combined)
    return round(hits / len(expected_keywords), 3)


def compute_reasoning_efficiency(resp: ResearchResponse, max_steps: int = 5) -> float:
    steps = resp.metrics.get("total_steps", max_steps)
    if steps <= 0:
        return 0.0
    # Penalize hitting max steps without early convergence
    efficiency = max(0.2, 1.0 - ((steps - 1) / (max_steps * 1.5)))
    return round(min(1.0, efficiency), 3)


async def llm_judge_task_completion(case: EvalCase, resp: ResearchResponse) -> float:
    judge_prompt = f"""Evaluate how accurately and completely the candidate response answers the query compared to the reference benchmark.
Query: {case.query}
Reference Key Points: {case.reference_answer}
Candidate Answer: {resp.final_answer}
Candidate Recommendation: {resp.decision_recommendation}

Score from 0.0 to 1.0 in JSON: {{"score": 0.85, "reason": "brief explanation"}}
"""
    res, _ = await llm_client.generate_json([{"role": "user", "content": judge_prompt}])
    return round(float(res.get("score", 0.0)), 3)


async def run_evaluation_suite(cases: List[EvalCase] = BENCHMARK_DATASET) -> List[EvalResult]:
    results: List[EvalResult] = []
    for case in cases:
        resp = await orchestrator.run(query=case.query, session_id=f"eval_{case.id}", max_steps=4)
        completion = await llm_judge_task_completion(case, resp)
        cit_score = compute_citation_correctness(resp)
        cov_score = compute_evidence_coverage(resp, case.expected_keywords)
        eff_score = compute_reasoning_efficiency(resp, max_steps=4)

        passed = (
            completion >= 0.7
            and cit_score >= 0.8
            and len(resp.citations) >= case.min_citations
        )
        results.append(
            EvalResult(
                case_id=case.id,
                query=case.query,
                task_completion_score=completion,
                citation_correctness_score=cit_score,
                evidence_coverage_score=cov_score,
                reasoning_efficiency_score=eff_score,
                total_latency_ms=resp.metrics["total_latency_ms"],
                total_tokens=resp.metrics["total_tokens"],
                estimated_cost_usd=resp.metrics["estimated_cost_usd"],
                passed=passed,
            )
        )
    return results


if __name__ == "__main__":
    eval_results = asyncio.run(run_evaluation_suite())
    report = [r.model_dump() for r in eval_results]
    print(json.dumps(report, indent=2))
    with open("eval_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
