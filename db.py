from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Sample knowledge base policies
faq_policies = [
    "Cancellation Policy: Users can cancel any order within 2 hours of purchase by clicking 'Cancel Order' in their dashboard. After 2 hours, orders enter processing and cannot be manually cancelled.",
    "Refund Policy: Refund requests are accepted within 30 days of product delivery. Items must be unopened and in original packaging.",
    "Shipping Info: Standard shipping takes 3 to 5 business days. Express shipping takes 1 to 2 business days.",
    "Damaged Items: If an item arrives broken, please contact support immediately with photos. We will ship a free replacement."
]

embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_texts(faq_policies, embeddings)

retriever = vector_store.as_retriever(search_kwargs={"k": 1})