# Customer Support RAG Assistant

An ecommerce support assistant built with FastAPI, LangGraph, FAISS, OpenAI embeddings, and SQLite. The project started as a small RAG demo and was upgraded into a resume-ready backend that demonstrates routing, retrieval, ticketing, and operational visibility.

## What it does

- Routes customer questions into self-service or human escalation.
- Retrieves answers from a persistent vector store backed by a curated knowledge base.
- Creates support tickets for billing disputes, fraud, account takeover, and other high-risk cases.
- Logs every interaction in SQLite for analytics and follow-up.
- Exposes support metrics and ticket history through API endpoints.

## Why it is stronger as a resume project

- Shows end-to-end LLM application design, not just a single prompt.
- Uses a real data layer instead of hard-coded answers.
- Demonstrates intent routing with structured outputs.
- Adds persistent observability through query logs and tickets.
- Leaves room for production extensions like auth, feedback, and multi-tenant support.

## Architecture

1. `main.py` exposes the FastAPI app and operational endpoints.
2. `graph.py` contains the LangGraph workflow.
3. `db.py` manages the knowledge base, FAISS retrieval, and SQLite persistence.
4. `schema.py` defines request, response, routing, and storage schemas.

## API

### `GET /health`
Returns basic service health and current counts.

### `POST /support/query`
Accepts:

```json
{
  "user_query": "Can I cancel my order?",
  "session_id": "session-123",
  "customer_id": "cust-001"
}
```

Returns a routed response with the assistant answer, confidence, sources, and any ticket ID.

### `POST /customer/support`
Compatibility alias for the main query endpoint.

### `GET /support/stats`
Returns query volume, escalation rate, open tickets, and top topics.

### `GET /support/tickets`
Lists recent human escalations.

### `GET /knowledge-base`
Lists the seeded support articles.

## Local setup

1. Create a `.env` file with your OpenAI credentials.
2. Install dependencies.
3. Start the server with:

```bash
uvicorn main:app --reload
```

## Suggested resume bullets

- Built a production-shaped ecommerce support assistant using FastAPI, LangGraph, FAISS, OpenAI, and SQLite.
- Implemented structured intent routing to separate self-service questions from human escalation.
- Added persistent query logging, ticket creation, and support analytics for operational visibility.
- Designed a reusable knowledge-base pipeline with cached vector retrieval and citation-ready responses.

## Next upgrades

- Add auth and role-based access for support agents.
- Add a feedback endpoint to capture resolution quality.
- Replace the local SQLite store with Postgres for multi-user scaling.
- Add ingestion from CSV, PDFs, or a CMS-backed knowledge base.
