from __future__ import annotations

from fastapi import FastAPI, Query

from config import APP_NAME, APP_VERSION
from db import (
    get_support_stats,
    get_vector_store,
    init_support_db,
    list_knowledge_base_items,
    list_recent_tickets,
    load_knowledge_base,
)
from graph import workflow
from schema import SupportQueryRequest, SupportQueryResponse, SupportStats, SupportTicketRecord

app = FastAPI(title=APP_NAME, version=APP_VERSION)


@app.on_event("startup")
def startup() -> None:
    init_support_db()
    load_knowledge_base()
    get_vector_store()


@app.get("/")
def check_api_health():
    stats = get_support_stats()
    return {
        "status": "ok",
        "version": APP_VERSION,
        "knowledge_base_documents": stats["knowledge_base_documents"],
        "total_queries": stats["total_queries"],
        "open_tickets": stats["open_tickets"],
    }


@app.post("/support/query", response_model=SupportQueryResponse)
def support_query(payload: SupportQueryRequest):
    result = workflow.invoke(payload.model_dump())
    return result


@app.post("/customer/support", response_model=SupportQueryResponse)
def customer_support(payload: SupportQueryRequest):
    return support_query(payload)


@app.get("/support/stats", response_model=SupportStats)
def support_stats():
    return get_support_stats()


@app.get("/support/tickets", response_model=list[SupportTicketRecord])
def support_tickets(limit: int = Query(default=20, ge=1, le=100)):
    return list_recent_tickets(limit=limit)


@app.get("/knowledge-base")
def knowledge_base():
    return {"documents": list_knowledge_base_items()}
