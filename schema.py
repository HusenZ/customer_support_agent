from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


SupportRoute = Literal["self_service", "human_escalation"]
SupportTopic = Literal[
    "orders",
    "shipping",
    "refunds",
    "damaged_items",
    "billing",
    "account",
    "subscription",
    "other",
]


class SupportQueryRequest(BaseModel):
    user_query: str = Field(..., min_length=1, description="Customer support question")
    session_id: str | None = Field(
        default=None,
        description="Optional conversation identifier for logging and follow-up context",
    )
    customer_id: str | None = Field(
        default=None,
        description="Optional customer identifier for future personalization",
    )


class QueryClassification(BaseModel):
    topic: SupportTopic = Field(description="Primary support topic")
    route: SupportRoute = Field(description="Whether the assistant can answer or must escalate")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for the selected route and topic",
    )
    reason: str = Field(description="Short explanation for the routing decision")


class KnowledgeSnippet(BaseModel):
    title: str
    category: str
    content: str
    score: float | None = None


class SupportQueryResponse(BaseModel):
    query_type: str
    route: SupportRoute
    confidence: float
    bot_reply: str
    sources: list[KnowledgeSnippet] = Field(default_factory=list)
    ticket_id: str | None = None
    session_id: str | None = None


class SupportTicketRecord(BaseModel):
    ticket_id: str
    session_id: str
    user_query: str
    topic: str
    reason: str
    status: str
    created_at: str
    updated_at: str


class KnowledgeBaseItem(BaseModel):
    id: str
    title: str
    category: str
    content: str
    keywords: list[str] = Field(default_factory=list)


class SupportStats(BaseModel):
    total_queries: int
    self_service_queries: int
    escalated_queries: int
    open_tickets: int
    knowledge_base_documents: int
    self_service_rate: float
    top_topics: list[dict[str, Any]] = Field(default_factory=list)


class CustomerSupportState(TypedDict, total=False):
    user_query: str
    session_id: str
    customer_id: str
    query_type: str
    route: SupportRoute
    confidence: float
    reason: str
    retrieved_docs: list[dict[str, Any]]
    bot_reply: str
    ticket_id: str
