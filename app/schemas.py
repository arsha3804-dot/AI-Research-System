from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SubTask(BaseModel):
    id: str
    objective: str
    recommended_tool: str
    status: str = "pending"
    result_summary: Optional[str] = None


class ResearchPlan(BaseModel):
    original_query: str
    refined_goal: str
    subtasks: List[SubTask]


class Citation(BaseModel):
    id: str  # e.g., "C1"
    source_type: str  # "rag_doc" | "web_search" | "arxiv" | "wikipedia"
    title: str
    url_or_path: str
    snippet: str


class AgentStepLog(BaseModel):
    step_number: int
    thought: str
    action_tool: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation_preview: Optional[str] = None
    tokens_used: int = 0
    latency_ms: float = 0.0


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Complex research or decision question")
    session_id: Optional[str] = Field(default="default", description="Conversation context ID")
    max_steps: int = Field(default=5, ge=1, le=10)
    use_rag: bool = True


class ResearchResponse(BaseModel):
    session_id: str
    query: str
    plan: ResearchPlan
    final_answer: str
    decision_recommendation: str
    citations: List[Citation]
    execution_trace: List[AgentStepLog]
    metrics: Dict[str, Any]


class EvalCase(BaseModel):
    id: str
    query: str
    expected_keywords: List[str]
    min_citations: int = 1
    reference_answer: str


class EvalResult(BaseModel):
    case_id: str
    query: str
    task_completion_score: float
    citation_correctness_score: float
    evidence_coverage_score: float
    reasoning_efficiency_score: float
    total_latency_ms: float
    total_tokens: int
    estimated_cost_usd: float
    passed: bool
