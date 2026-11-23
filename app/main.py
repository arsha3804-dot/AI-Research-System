from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.agent import orchestrator
from app.rag_engine import rag_engine
from app.mcp_client import mcp_manager
from app.schemas import ResearchRequest, ResearchResponse
import asyncio
import sys

if sys.platform == 'wni32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Auto-ingest local documents in ./data/docs at startup
    rag_engine.ingest_directory()
    yield


app = FastAPI(
    title="Agentic AI Research & Decision System",
    description="FastAPI orchestration layer for autonomous planning, MCP tool calling, local RAG, and cited synthesis using free LLMs.",
    version="1.0.0",
    lifespan=lifespan,
)


class IngestTextRequest(BaseModel):
    doc_id: str
    title: str
    source: str = "api_upload"
    content: str


@app.get("/health")
async def health():
    tools = await mcp_manager.list_tools()
    return {
        "status": "ok",
        "rag_indexed_chunks": rag_engine.collection.count(),
        "mcp_tools_available": [t["name"] for t in tools],
    }


@app.post("/rag/ingest")
async def ingest_document(req: IngestTextRequest):
    chunks = rag_engine.ingest_text(req.doc_id, req.title, req.source, req.content)
    return {"status": "indexed", "doc_id": req.doc_id, "chunks_added": chunks}


@app.post("/rag/reindex-folder")
async def reindex_folder():
    return rag_engine.ingest_directory()


@app.post("/research", response_model=ResearchResponse)
async def run_research(req: ResearchRequest):
    try:
        return await orchestrator.run(
            query=req.query,
            session_id=req.session_id or "default",
            max_steps=req.max_steps,
            use_rag=req.use_rag,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    orchestrator.sessions.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}
