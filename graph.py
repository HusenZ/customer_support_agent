from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

from schema import CustomerSupportState, ClassifierSchema
from db import retriever

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")
structured_model = model.with_structured_output(ClassifierSchema)

# --- NODES ---

def classify_query(state: CustomerSupportState):
    user_query = state["user_query"]
    prompt1 = f"Evaluate the users query for a ecommerce platform and check whether this require human support or company legal database is enough for this based on that classify the users query into simple or critical. User query:\n{user_query}"
    
    output = structured_model.invoke(prompt1)
    return {"query_type": output.query_type}

def rag_node(state: CustomerSupportState):
    print("--- ENTERING RAG NODE ---")
    user_query = state["user_query"]
    
    docs = retriever.invoke(user_query)
    context = docs[0].page_content
    
    rag_prompt = f"""You are a helpful customer support bot. 
    Answer the user's query using ONLY the provided context below.
    
    Context:
    {context}
    
    User Query: {user_query}
    """
    
    response = model.invoke(rag_prompt)
    return {"bot_reply": response.content}

def escalate_node(state: CustomerSupportState):
    print("--- ENTERING ESCALATE NODE ---")
    return {"bot_reply": "Your request requires human assistance. We have logged an escalation ticket for you."}

# --- ROUTING LOGIC ---

def check_query_type(state: CustomerSupportState):
    query_type = state["query_type"]
    if query_type == "simple":
        return "rag_node"
    else:
        return "escalate_node"

# --- GRAPH BUILDER ---

graph = StateGraph(CustomerSupportState)

graph.add_node("classifier_node", classify_query)
graph.add_node("rag_node", rag_node)
graph.add_node("escalate_node", escalate_node)

graph.add_edge(START, "classifier_node")
graph.add_conditional_edges(
    "classifier_node",
    check_query_type,
    {
        "rag_node": "rag_node",
        "escalate_node": "escalate_node"
    }
)
graph.add_edge("rag_node", END)
graph.add_edge("escalate_node", END)

workflow = graph.compile()