# Customer Support RAG Assistant

An ecommerce customer support assistant built with FastAPI, LangGraph, FAISS, OpenAI models, and SQLite. The system classifies incoming questions, retrieves policy answers from a knowledge base, escalates sensitive cases to human review, and logs every interaction for analytics.

This project was designed to be more than a demo. It shows end-to-end LLM application design: routing, retrieval, persistence, observability, and operational APIs.

## What it does

- Classifies customer questions into self-service or human escalation.
- Answers policy-based questions using retrieval augmented generation.
- Escalates billing disputes, account security issues, and other risky cases.
- Creates support tickets for escalated requests.
- Persists query logs and tickets in SQLite.
- Stores the knowledge base in a local FAISS index for fast retrieval.
- Exposes support metrics and ticket history through FastAPI endpoints.

## Why this is a strong resume project

- Uses a real orchestration layer with LangGraph instead of a single prompt call.
- Separates routing, retrieval, generation, and escalation into distinct components.
- Adds persistence for both the knowledge base and support workflow state.
- Includes basic observability through query logs, support tickets, and stats endpoints.
- Demonstrates a realistic product pattern for a support automation system.

## Architecture

1. `main.py` exposes the FastAPI app and HTTP endpoints.
2. `graph.py` defines the LangGraph workflow.
3. `db.py` manages the knowledge base, FAISS vector store, and SQLite persistence.
4. `schema.py` defines request, response, ticket, and routing schemas.
5. `config.py` centralizes runtime configuration.

## Workflow

1. A customer submits a support question.
2. The classifier determines the topic and whether the request can be self-served.
3. If the request is simple, the assistant retrieves relevant knowledge base entries and generates a response.
4. If the request is risky or ambiguous, the assistant creates a support ticket and escalates it.
5. The result is logged in SQLite for later review and analytics.

## API Endpoints

### `GET /health`
Returns basic service health and current counts.

Example response:

```json
{
  "status": "ok",
  "version": "2.0.0",
  "knowledge_base_documents": 7,
  "total_queries": 12,
  "open_tickets": 3
}
```

### `POST /support/query`
Primary endpoint for customer support requests.

Request:

```json
{
  "user_query": "Can I cancel my order?",
  "session_id": "session-123",
  "customer_id": "cust-001"
}
```

Response:

```json
{
  "query_type": "orders",
  "route": "self_service",
  "confidence": 0.91,
  "bot_reply": "You can cancel your order within 2 hours of purchase ...",
  "sources": [
    {
      "title": "Order Cancellation Policy",
      "category": "orders",
      "content": "Users can cancel an order within 2 hours of purchase ...",
      "score": 0.83
    }
  ],
  "ticket_id": null,
  "session_id": "session-123"
}
```

### `POST /customer/support`
Compatibility alias for `/support/query`.

### `GET /support/stats`
Returns operational metrics:

- total queries
- self-service queries
- escalated queries
- open tickets
- knowledge base size
- self-service rate
- top topics

### `GET /support/tickets`
Lists recent support tickets.

Query parameters:

- `limit` defaults to `20`
- minimum `1`
- maximum `100`

### `GET /knowledge-base`
Returns the seeded knowledge base articles.

## Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0
ESCALATION_CONFIDENCE_THRESHOLD=0.62
```

## Local Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the API

```bash
uvicorn main:app --reload
```

### 3. Try a request

```bash
curl -X POST "http://127.0.0.1:8000/support/query" \
  -H "Content-Type: application/json" \
  -d '{
    "user_query": "What is your refund policy?",
    "session_id": "demo-session-1",
    "customer_id": "cust-100"
  }'
```

## Data Storage

The app creates local persistent data under `data/`:

- `data/knowledge_base.json` stores the support articles.
- `data/vectorstore/` stores the FAISS index.
- `data/support.db` stores query logs and support tickets.

These files are generated locally and should not be committed.

## Project Structure

```text
.
├── config.py
├── db.py
├── graph.py
├── main.py
├── schema.py
├── data/
│   └── knowledge_base.json
├── requirements.txt
└── README.md
```

## Suggested Resume Bullets

- Built a production-shaped ecommerce support assistant using FastAPI, LangGraph, FAISS, OpenAI, and SQLite.
- Implemented structured routing to separate self-service requests from human escalation cases.
- Added persistent query logging, support ticketing, and analytics for operational visibility.
- Designed a reusable knowledge base pipeline with cached vector retrieval and citation-style source output.

## Next Improvements

- Add authentication and role-based access for support agents.
- Add a feedback endpoint to track answer quality and resolution success.
- Replace SQLite with Postgres for multi-user scale and deployment.
- Add ingestion pipelines for CSV, PDFs, or CMS-backed knowledge sources.
- Add a lightweight dashboard for support staff and analytics.
