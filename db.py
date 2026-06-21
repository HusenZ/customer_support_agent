from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from schema import KnowledgeBaseItem

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DIR = DATA_DIR / "vectorstore"
KNOWLEDGE_BASE_PATH = DATA_DIR / "knowledge_base.json"
SUPPORT_DB_PATH = DATA_DIR / "support.db"


DEFAULT_KNOWLEDGE_BASE: list[dict[str, Any]] = [
    {
        "id": "order-cancellation",
        "title": "Order Cancellation Policy",
        "category": "orders",
        "content": (
            "Users can cancel an order within 2 hours of purchase from their dashboard. "
            "After 2 hours, orders enter processing and cannot be canceled manually."
        ),
        "keywords": ["cancel", "cancellation", "order", "purchase", "dashboard"],
    },
    {
        "id": "refund-policy",
        "title": "Refund Eligibility",
        "category": "refunds",
        "content": (
            "Refund requests are accepted within 30 days of product delivery. "
            "Items must be unopened, unused, and returned in the original packaging."
        ),
        "keywords": ["refund", "return", "package", "delivery", "unopened"],
    },
    {
        "id": "shipping-times",
        "title": "Shipping Timelines",
        "category": "shipping",
        "content": (
            "Standard shipping takes 3 to 5 business days. Express shipping takes 1 to 2 business days."
        ),
        "keywords": ["shipping", "delivery", "express", "standard", "days"],
    },
    {
        "id": "damaged-items",
        "title": "Damaged or Broken Items",
        "category": "damaged_items",
        "content": (
            "If an item arrives damaged, customers should contact support immediately with photos. "
            "A replacement or refund can be issued after verification."
        ),
        "keywords": ["damaged", "broken", "photos", "replacement", "refund"],
    },
    {
        "id": "billing-disputes",
        "title": "Billing Disputes and Chargebacks",
        "category": "billing",
        "content": (
            "Billing disputes, chargebacks, and payment failures require human review. "
            "Customers should contact support with the last four digits of the payment method and the transaction date."
        ),
        "keywords": ["billing", "chargeback", "payment", "invoice", "dispute"],
    },
    {
        "id": "account-security",
        "title": "Account Security Guidance",
        "category": "account",
        "content": (
            "If a customer suspects account takeover or unauthorized access, they should reset their password immediately, "
            "review recent sessions, and contact support for manual account review."
        ),
        "keywords": ["account", "security", "hack", "takeover", "password"],
    },
    {
        "id": "subscription-management",
        "title": "Subscription Management",
        "category": "subscription",
        "content": (
            "Subscriptions can be paused or canceled from the billing portal. "
            "Changes apply to the next billing cycle unless the current cycle is still within the free-trial window."
        ),
        "keywords": ["subscription", "pause", "cancel", "billing portal", "trial"],
    },
]


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_knowledge_items(raw_items: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for raw in raw_items:
        item = KnowledgeBaseItem.model_validate(raw)
        items.append(item.model_dump())
    return items


def load_knowledge_base() -> list[dict[str, Any]]:
    _ensure_data_dir()

    if KNOWLEDGE_BASE_PATH.exists():
        with KNOWLEDGE_BASE_PATH.open("r", encoding="utf-8") as handle:
            raw_items = json.load(handle)
            if isinstance(raw_items, list) and raw_items:
                return _normalize_knowledge_items(raw_items)

    items = _normalize_knowledge_items(DEFAULT_KNOWLEDGE_BASE)
    with KNOWLEDGE_BASE_PATH.open("w", encoding="utf-8") as handle:
        json.dump(items, handle, indent=2)
    return items


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings()


@lru_cache(maxsize=1)
def get_vector_store() -> FAISS:
    _ensure_data_dir()
    embeddings = get_embeddings()

    if (VECTOR_DIR / "index.faiss").exists() and (VECTOR_DIR / "index.pkl").exists():
        try:
            return FAISS.load_local(
                str(VECTOR_DIR),
                embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception:
            pass

    docs = load_knowledge_base()
    texts = [doc["content"] for doc in docs]
    metadatas = [
        {
            "id": doc["id"],
            "title": doc["title"],
            "category": doc["category"],
            "keywords": ", ".join(doc.get("keywords", [])),
        }
        for doc in docs
    ]

    vector_store = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(VECTOR_DIR))
    return vector_store


@lru_cache(maxsize=1)
def get_retriever():
    return get_vector_store().as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 8})


def search_knowledge_base(query: str, limit: int = 3) -> list[dict[str, Any]]:
    store = get_vector_store()

    try:
        matches = store.similarity_search_with_relevance_scores(query, k=limit)
    except Exception:
        matches = [(doc, None) for doc in store.similarity_search(query, k=limit)]

    results: list[dict[str, Any]] = []
    for doc, score in matches:
        results.append(
            {
                "title": doc.metadata.get("title", "Knowledge Base"),
                "category": doc.metadata.get("category", "general"),
                "content": doc.page_content,
                "score": float(score) if score is not None else None,
            }
        )
    return results


def _connect() -> sqlite3.Connection:
    _ensure_data_dir()
    connection = sqlite3.connect(SUPPORT_DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_support_db() -> None:
    with _connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS query_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                customer_id TEXT,
                user_query TEXT NOT NULL,
                query_type TEXT NOT NULL,
                route TEXT NOT NULL,
                confidence REAL NOT NULL,
                bot_reply TEXT NOT NULL,
                ticket_id TEXT,
                sources_json TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS support_tickets (
                ticket_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                customer_id TEXT,
                user_query TEXT NOT NULL,
                topic TEXT NOT NULL,
                reason TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_query_logs_session_id
                ON query_logs(session_id, created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_support_tickets_session_id
                ON support_tickets(session_id, created_at DESC);
            """
        )


def create_ticket(
    *,
    session_id: str,
    customer_id: str | None,
    user_query: str,
    topic: str,
    reason: str,
    status: str = "open",
) -> str:
    from uuid import uuid4

    ticket_id = f"TKT-{uuid4().hex[:10].upper()}"
    timestamp = _utc_now()
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO support_tickets (
                ticket_id, session_id, customer_id, user_query, topic, reason, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticket_id,
                session_id,
                customer_id,
                user_query,
                topic,
                reason,
                status,
                timestamp,
                timestamp,
            ),
        )
    return ticket_id


def record_query(
    *,
    session_id: str,
    customer_id: str | None,
    user_query: str,
    query_type: str,
    route: str,
    confidence: float,
    bot_reply: str,
    ticket_id: str | None = None,
    sources: list[dict[str, Any]] | None = None,
) -> None:
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO query_logs (
                session_id, customer_id, user_query, query_type, route, confidence, bot_reply,
                ticket_id, sources_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                customer_id,
                user_query,
                query_type,
                route,
                confidence,
                bot_reply,
                ticket_id,
                json.dumps(sources or []),
                _utc_now(),
            ),
        )


def get_session_history(session_id: str, limit: int = 4) -> list[dict[str, Any]]:
    if not session_id:
        return []

    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT user_query, query_type, route, confidence, bot_reply, ticket_id, created_at
            FROM query_logs
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()

    history = [dict(row) for row in reversed(rows)]
    return history


def list_recent_tickets(limit: int = 20) -> list[dict[str, Any]]:
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT ticket_id, session_id, customer_id, user_query, topic, reason, status, created_at, updated_at
            FROM support_tickets
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_support_stats() -> dict[str, Any]:
    with _connect() as connection:
        total_queries = connection.execute("SELECT COUNT(*) FROM query_logs").fetchone()[0]
        self_service_queries = connection.execute(
            "SELECT COUNT(*) FROM query_logs WHERE route = 'self_service'"
        ).fetchone()[0]
        escalated_queries = connection.execute(
            "SELECT COUNT(*) FROM query_logs WHERE route = 'human_escalation'"
        ).fetchone()[0]
        open_tickets = connection.execute(
            "SELECT COUNT(*) FROM support_tickets WHERE status = 'open'"
        ).fetchone()[0]
        top_topics_rows = connection.execute(
            """
            SELECT query_type, COUNT(*) AS count
            FROM query_logs
            GROUP BY query_type
            ORDER BY count DESC, query_type ASC
            LIMIT 5
            """
        ).fetchall()

    knowledge_base_documents = len(load_knowledge_base())
    self_service_rate = (
        round((self_service_queries / total_queries) * 100, 2) if total_queries else 0.0
    )

    return {
        "total_queries": total_queries,
        "self_service_queries": self_service_queries,
        "escalated_queries": escalated_queries,
        "open_tickets": open_tickets,
        "knowledge_base_documents": knowledge_base_documents,
        "self_service_rate": self_service_rate,
        "top_topics": [dict(row) for row in top_topics_rows],
    }


def list_knowledge_base_items() -> list[dict[str, Any]]:
    return load_knowledge_base()
