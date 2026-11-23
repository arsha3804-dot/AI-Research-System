# Internal Engineering Architecture Notes (2026)

## Vector Database Benchmark Summary
- For datasets under 10 million embeddings (768d or 384d), PostgreSQL 17 with `pgvector` (HNSW index) achieves 12ms p95 query latency while keeping transactional and vector data in one database.
- Dedicated vector engines like Qdrant or Milvus outperform `pgvector` above 25M vectors or when heavy metadata pre-filtering and distributed sharding are required.

## Agentic System Design Standards
- All external tool integrations should expose typed JSON schemas via Model Context Protocol (MCP) servers to prevent vendor lock-in.
- Every factual claim in executive decision memos must map to a verifiable source citation ID.
