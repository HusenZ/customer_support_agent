from __future__ import annotations

from uuid import uuid4

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from config import ESCALATION_CONFIDENCE_THRESHOLD, OPENAI_MODEL, OPENAI_TEMPERATURE
from db import create_ticket, get_session_history, record_query, search_knowledge_base
from schema import CustomerSupportState, KnowledgeSnippet, QueryClassification

load_dotenv()

model = ChatOpenAI(model=OPENAI_MODEL, temperature=OPENAI_TEMPERATURE)
structured_model = model.with_structured_output(QueryClassification)


def _format_history(history: list[dict[str, object]]) -> str:
    if not history:
        return "No prior conversation history."

    lines: list[str] = []
    for turn in history:
        lines.append(f"User: {turn.get('user_query', '')}")
        lines.append(f"Assistant: {turn.get('bot_reply', '')}")
    return "\n".join(lines)


def _format_context(snippets: list[dict[str, object]]) -> str:
    if not snippets:
        return "No supporting knowledge base articles were retrieved."

    blocks: list[str] = []
    for snippet in snippets:
        blocks.append(
            "\n".join(
                [
                    f"Title: {snippet.get('title', 'Knowledge Base')}",
                    f"Category: {snippet.get('category', 'general')}",
                    f"Content: {snippet.get('content', '')}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _normalize_session_id(session_id: str | None) -> str:
    return session_id or f"session-{uuid4().hex[:12]}"


def classify_query(state: CustomerSupportState):
    user_query = state["user_query"]
    session_id = _normalize_session_id(state.get("session_id"))
    history = _format_history(get_session_history(session_id))

    prompt = f"""
You are routing an ecommerce support request.

Conversation history:
{history}

Customer query:
{user_query}

Routing rules:
- Return route="self_service" when the issue can be answered from standard support policies or normal troubleshooting.
- Return route="human_escalation" for billing disputes, chargebacks, account takeover, fraud, legal threats, safety issues, policy edge cases, or vague requests with low confidence.
- Choose the most relevant topic from: orders, shipping, refunds, damaged_items, billing, account, subscription, other.
- Confidence must be a number between 0 and 1 and reflect your certainty in the routing decision.
"""

    output = structured_model.invoke(prompt)
    route = output.route
    confidence = output.confidence

    if confidence < ESCALATION_CONFIDENCE_THRESHOLD and route == "self_service":
        route = "human_escalation"

    return {
        "session_id": session_id,
        "query_type": output.topic,
        "route": route,
        "confidence": confidence,
        "reason": output.reason,
    }


def rag_node(state: CustomerSupportState):
    user_query = state["user_query"]
    session_id = state["session_id"]
    topic = state.get("query_type", "other")
    confidence = state.get("confidence", 0.0)
    history = _format_history(get_session_history(session_id))

    retrieved_docs = search_knowledge_base(user_query, limit=3)
    context = _format_context(retrieved_docs)

    rag_prompt = f"""
You are a concise, reliable customer support agent for an ecommerce company.

Use only the knowledge base context below and the recent conversation history.
If the context does not fully answer the question, say what is missing and recommend a human escalation.
Do not invent policy details.

Conversation history:
{history}

Knowledge base context:
{context}

Customer query:
{user_query}

Return a direct answer first, then optionally a short next step.
"""

    response = model.invoke(rag_prompt)
    bot_reply = response.content.strip()

    sources = [
        KnowledgeSnippet(
            title=doc["title"],
            category=doc["category"],
            content=doc["content"],
            score=doc.get("score"),
        ).model_dump()
        for doc in retrieved_docs
    ]

    record_query(
        session_id=session_id,
        customer_id=state.get("customer_id"),
        user_query=user_query,
        query_type=topic,
        route="self_service",
        confidence=confidence,
        bot_reply=bot_reply,
        sources=sources,
    )

    return {
        "bot_reply": bot_reply,
        "retrieved_docs": retrieved_docs,
        "ticket_id": None,
    }


def escalate_node(state: CustomerSupportState):
    session_id = state["session_id"]
    topic = state.get("query_type", "other")
    reason = state.get("reason", "The request requires human review.")
    user_query = state["user_query"]

    ticket_id = create_ticket(
        session_id=session_id,
        customer_id=state.get("customer_id"),
        user_query=user_query,
        topic=topic,
        reason=reason,
    )

    bot_reply = (
        "I’ve created a support ticket for this request and flagged it for human review. "
        f"Your ticket ID is {ticket_id}."
    )

    record_query(
        session_id=session_id,
        customer_id=state.get("customer_id"),
        user_query=user_query,
        query_type=topic,
        route="human_escalation",
        confidence=state.get("confidence", 0.0),
        bot_reply=bot_reply,
        ticket_id=ticket_id,
        sources=[],
    )

    return {
        "bot_reply": bot_reply,
        "ticket_id": ticket_id,
        "retrieved_docs": [],
    }


def route_query(state: CustomerSupportState):
    if state.get("route") == "self_service":
        return "rag_node"
    return "escalate_node"


graph = StateGraph(CustomerSupportState)

graph.add_node("classifier_node", classify_query)
graph.add_node("rag_node", rag_node)
graph.add_node("escalate_node", escalate_node)

graph.add_edge(START, "classifier_node")
graph.add_conditional_edges(
    "classifier_node",
    route_query,
    {
        "rag_node": "rag_node",
        "escalate_node": "escalate_node",
    },
)
graph.add_edge("rag_node", END)
graph.add_edge("escalate_node", END)

workflow = graph.compile()
