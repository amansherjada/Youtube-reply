# main.py (Production YouTube Comment API)
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from pinecone import Pinecone

# Environment configuration
if os.environ.get("ENV") != "production":
    from dotenv import load_dotenv
    load_dotenv()

# LangChain imports (production-safe)
try:
    from langchain_pinecone import PineconeVectorStore
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_core.prompts import PromptTemplate
    from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
except ImportError as e:
    raise RuntimeError("Missing dependencies! Check requirements.txt") from e

# Initialize services
app = Flask(__name__)
CORS(app)

# Pinecone initialization
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(os.environ["PINECONE_INDEX_NAME"])

# OpenAI setup
embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
vector_store = PineconeVectorStore(index, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# LLM configuration
llm = ChatOpenAI(
    model="gpt-3.5-turbo-0125",
    temperature=0.6,
    max_tokens=150,
    openai_api_key=os.environ["OPENAI_API_KEY"]
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
    app.run(host=os.environ.get("HOST", "0.0.0.0"), 
            port=int(os.environ.get("PORT", 8080)))
