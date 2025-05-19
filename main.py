# main.py (production-ready)
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Configuration
app = Flask(__name__)
CORS(app)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Initialize services
def initialize_services():
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    return PineconeVectorStore(index, embeddings)

vector_store = initialize_services()
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

llm = ChatOpenAI(
    model="gpt-3.5-turbo-0125",  # Updated to latest 3.5 turbo
    temperature=0.7,
    max_tokens=150
)

# Response generation
def generate_reply(query: str) -> str:
    try:
        docs = retriever.invoke(query)
        context = "\n".join(d.page_content for d in docs) or "No context"
        
        prompt = PromptTemplate.from_template("""
        You're American Hairline's YouTube responder. Be friendly and concise.
        Context: {context}
        Comment: {input}
        Response:""")
        
        chain = create_stuff_documents_chain(llm, prompt)
        return chain.invoke({"input": query, "context": context}).strip()
    
    except Exception as e:
        print(f"Generation error: {str(e)}")
        return "Thanks for your comment! 😊 We'll respond shortly."

# API endpoint
@app.route("/youtube-reply", methods=["POST"])
def handle_comment():
    if not (query := request.json.get("query")):
        return jsonify(error="Missing comment"), 400
    return jsonify(response=generate_reply(query))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
