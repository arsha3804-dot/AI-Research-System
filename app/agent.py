
import json
import time
from typing import Any, Dict, List
from app.llm_client import llm_client
from app.rag_engine import rag_engine
from app.mcp_client import mcp_manager
from app.schemas import (
    AgentStepLog,
    Citation,
    ResearchPlan,
    ResearchResponse,
    SubTask,
)


class AgenticResearchOrchestrator:
    """
    Multi-step Autonomous Research & Decision Agent:
    1. Task Decomposition & Planning
    2. Dynamic Tool Selection (Local RAG + MCP Tools: Web, arXiv, Wikipedia, MCDA)
    3. Iterative ReAct Reasoning Loop with Evidence Accumulation
    4. Grounded Synthesis with Verified Inline Citations [C1], [C2]...
    """

    def __init__(self):
        # Simple in-memory conversation memory keyed by session_id
        self.sessions: Dict[str, List[Dict[str, str]]] = {}

    async def _decompose_task(
        self, query: str, available_tools: List[Dict[str, Any]], history: List[Dict[str, str]]
    ) -> tuple[ResearchPlan, Dict[str, Any]]:
        tool_summary = "\n".join(
            f"- {t['name']}: {t['description']}" for t in available_tools
        )
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content'][:300]}" for m in history[-4:]
        )
        prompt = f"""You are a senior research planner. Decompose the user's query into 2 to 4 concrete, sequential subtasks.
Available tools:
- local_rag_search: Search internal ingested knowledge base documents.
{tool_summary}

Recent conversation context:
{history_text or "None"}

User Query: "{query}"

Respond strictly in JSON matching this schema:
{{
  "refined_goal": "clear statement of decision/research objective",
  "subtasks": [
    {{"id": "T1", "objective": "what to find or compute", "recommended_tool": "tool_name"}}
  ]
}}
"""
        data, meta = await llm_client.generate_json(
            [{"role": "user", "content": prompt}]
        )
        subtasks = [
            SubTask(
                id=st.get("id", f"T{i+1}"),
                objective=st["objective"],
                recommended_tool=st.get("recommended_tool", "web_search"),
            )
            for i, st in enumerate(data.get("subtasks", []))
        ]
        plan = ResearchPlan(
            original_query=query,
            refined_goal=data.get("refined_goal", query),
            subtasks=subtasks,
        )
        return plan, meta

    def _register_citations(
        self,
        citations: List[Citation],
        source_type: str,
        raw_items: List[Dict[str, Any]],
    ) -> str:
        formatted_lines = []
        for item in raw_items:
            if "error" in item:
                continue
            cid = f"C{len(citations) + 1}"
            cit = Citation(
                id=cid,
                source_type=source_type,
                title=item.get("title", "Untitled Source"),
                url_or_path=item.get("url") or item.get("source", "unknown"),
                snippet=item.get("snippet", "")[:450],
            )
            citations.append(cit)
            formatted_lines.append(
                f"[{cid}] {cit.title} ({cit.url_or_path}): {cit.snippet}"
            )
        return "\n".join(formatted_lines) if formatted_lines else "No usable results."

    async def run(
        self,
        query: str,
        session_id: str = "default",
        max_steps: int = 5,
        use_rag: bool = True,
    ) -> ResearchResponse:
        overall_start = time.perf_counter()
        total_tokens = 0
        total_cost = 0.0
        trace: List[AgentStepLog] = []
        citations: List[Citation] = []
        evidence_bank: List[str] = []

        history = self.sessions.setdefault(session_id, [])

        # Discover MCP tools dynamically
        mcp_tools = await mcp_manager.list_tools()

        # Step 0: Task Decomposition
        plan, plan_meta = await self._decompose_task(query, mcp_tools, history)
        total_tokens += plan_meta["total_tokens"]
        total_cost += plan_meta["cost_usd"]

        # Optional initial RAG grounding pass
        if use_rag:
            t0 = time.perf_counter()
            rag_hits = rag_engine.retrieve(query, top_k=3)
            if rag_hits:
                obs = self._register_citations(citations, "rag_doc", rag_hits)
                evidence_bank.append(f"--- Local RAG Evidence ---\n{obs}")
                trace.append(
                    AgentStepLog(
                        step_number=0,
                        thought="Checking internal RAG knowledge base before external search.",
                        action_tool="local_rag_search",
                        action_input={"query": query, "top_k": 3},
                        observation_preview=obs[:300],
                        tokens_used=0,
                        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                    )
                )

        # Iterative Autonomous Execution Loop
        tool_descriptions = "\n".join(
            f"- {t['name']}: {t['description']} | Schema: {json.dumps(t['input_schema'])}"
            for t in mcp_tools
        )
        tool_descriptions += '\n- local_rag_search: Search internal documents | Schema: {"query": "string"}'

        for step in range(1, max_steps + 1):
            plan_status = "\n".join(
                f"{st.id} [{st.status}]: {st.objective} (suggested: {st.recommended_tool})"
                for st in plan.subtasks
            )
            collected_evidence = "\n\n".join(evidence_bank[-6:]) or "None yet."

            controller_prompt = f"""You are an autonomous research and decision agent.
Goal: {plan.refined_goal}
Subtasks:
{plan_status}

Collected Evidence so far:
{collected_evidence}

Available Tools:
{tool_descriptions}

Decide the next action. If you have gathered sufficient verified evidence to answer the query and make a clear recommendation, set "action": "FINISH".
Otherwise, choose the single best tool and arguments to gather missing facts or run decision scoring.

Respond strictly in JSON:
{{
  "thought": "concise reasoning about what is known and what is needed next",
  "completed_subtask_id": "T1 or null",
  "action": "tool_name OR FINISH",
  "action_input": {{}}
}}
"""
            decision, step_meta = await llm_client.generate_json(
                [{"role": "user", "content": controller_prompt}]
            )
            total_tokens += step_meta["total_tokens"]
            total_cost += step_meta["cost_usd"]

            done_id = decision.get("completed_subtask_id")
            if done_id:
                for st in plan.subtasks:
                    if st.id == done_id:
                        st.status = "completed"

            action = decision.get("action", "FINISH")
            action_input = decision.get("action_input") or {}
            thought = decision.get("thought", "")

            if action == "FINISH":
                trace.append(
                    AgentStepLog(
                        step_number=step,
                        thought=thought,
                        action_tool="FINISH",
                        action_input={},
                        observation_preview="Sufficient evidence gathered. Proceeding to synthesis.",
                        tokens_used=step_meta["total_tokens"],
                        latency_ms=step_meta["latency_ms"],
                    )
                )
                break

            # Execute chosen tool
            t_tool = time.perf_counter()
            try:
                if action == "local_rag_search":
                    q = action_input.get("query", query)
                    hits = rag_engine.retrieve(q, top_k=3)
                    obs_formatted = self._register_citations(citations, "rag_doc", hits)
                else:
                    raw_out = await mcp_manager.call_tool(action, action_input)
                    parsed = json.loads(raw_out)
                    if isinstance(parsed, list):
                        obs_formatted = self._register_citations(citations, action, parsed)
                    elif isinstance(parsed, dict) and "snippet" in parsed:
                        obs_formatted = self._register_citations(citations, action, [parsed])
                    else:
                        obs_formatted = f"Tool ({action}) output: {raw_out}"
            except Exception as exc:
                obs_formatted = f"Tool execution error on {action}: {exc}"

            tool_ms = round((time.perf_counter() - t_tool) * 1000, 2)
            evidence_bank.append(f"--- Step {step} ({action}) ---\n{obs_formatted}")

            trace.append(
                AgentStepLog(
                    step_number=step,
                    thought=thought,
                    action_tool=action,
                    action_input=action_input,
                    observation_preview=obs_formatted[:350],
                    tokens_used=step_meta["total_tokens"],
                    latency_ms=round(step_meta["latency_ms"] + tool_ms, 2),
                )
            )

        # Final Grounded Synthesis with Citations
        all_evidence = "\n\n".join(evidence_bank) or "No external evidence retrieved."
        synthesis_prompt = f"""You are an executive research analyst. Synthesize a rigorous, well-structured answer and an actionable decision recommendation based ONLY on the verified evidence below.

Rules:
1. Cite specific sources inline using [C1], [C2], etc. whenever stating facts from the evidence.
2. Do NOT fabricate citation IDs that do not exist in the evidence list.
3. Highlight trade-offs, risks, and quantitative comparisons where available.

User Query: "{query}"

Verified Evidence:
{all_evidence}

Respond strictly in JSON:
{{
  "final_answer": "comprehensive markdown analysis with inline [C#] citations",
  "decision_recommendation": "clear, decisive recommendation with rationale and [C#] citations"
}}
"""
        synth, synth_meta = await llm_client.generate_json(
            [{"role": "user", "content": synthesis_prompt}], temperature=0.2
        )
        total_tokens += synth_meta["total_tokens"]
        total_cost += synth_meta["cost_usd"]

        total_latency_ms = round((time.perf_counter() - overall_start) * 1000, 2)

        final_answer = synth.get("final_answer", "")
        recommendation = synth.get("decision_recommendation", "")

        # Update conversation memory
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": final_answer})

        return ResearchResponse(
            session_id=session_id,
            query=query,
            plan=plan,
            final_answer=final_answer,
            decision_recommendation=recommendation,
            citations=citations,
            execution_trace=trace,
            metrics={
                "model_used": synth_meta["model"],
                "total_steps": len(trace),
                "total_citations": len(citations),
                "total_tokens": total_tokens,
                "total_latency_ms": total_latency_ms,
                "estimated_cost_usd": round(total_cost, 6),
            },
        )


orchestrator = AgenticResearchOrchestrator()
