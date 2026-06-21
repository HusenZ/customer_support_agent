from fastapi import FastAPI
from graph import workflow

app = FastAPI()

@app.get("/health")
def check_api_health():
    return {
        "status":"200",
        "message":"all ok"
    }

@app.post("/customer/support")
def customer_support(user_query:str):
    try:
        initial_state = {"user_query": user_query}
        res = workflow.invoke(initial_state)
        return {
            "query_type": res.get("query_type"),
            "bot_reply": res.get("bot_reply")
        }
    except Exception as e:
        return {
            "error": "An error occurred while processing the request.",
            "details": str(e)
        }