# main.py (YouTube Comment API)
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from pinecone import Pinecone

# Load environment variables
load_dotenv()

# Initialize services
app = Flask(__name__)
CORS(app)
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

def initialize_services():
    index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    return PineconeVectorStore(index, embeddings)

vector_store = initialize_services()
retriever = vector_store.as_retriever(search_kwargs={"k": 3})  # Optimized for comment context

llm = ChatOpenAI(
    model="gpt-3.5-turbo-0125",
    temperature=0.6,
    max_tokens=150
)

PROMPT_TEMPLATE = """
You are American Hairline's YouTube comment responder. Follow these rules:

1. Keep responses under 100 words
2. Use emojis sparingly (max 2 per response)
3. Never discuss pricing
4. Focus on engagement and driving to consultations
5. Use casual YouTube-friendly language

Context about our services: {context}

YouTube comment to respond to: {input}

Craft a friendly response that:
- Acknowledges the comment
- Provides value from context
- Encourages further engagement
"""

def generate_youtube_response(comment: str) -> str:
    try:
        docs = retriever.invoke(comment)
        context = "\n".join(d.page_content for d in docs) or "No specific context"
        
        prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
        chain = create_stuff_documents_chain(llm, prompt)
        
        return chain.invoke({"input": comment, "context": context}).strip()
    
    except Exception as e:
        print(f"Response Error: {str(e)}")
        return "Thanks for your comment! 🙌 We'll get back to you shortly."

@app.route("/youtube-response", methods=["POST"])
def handle_comment():
    if not (comment := request.json.get("comment")):
        return jsonify(error="Comment required"), 400
    
    response = generate_youtube_response(comment)
    return jsonify(response=response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
