from flask import Flask, request, jsonify
from flask_cors import CORS
import os

# Load environment variables in development only
if os.environ.get("ENV") != "production":
    from dotenv import load_dotenv
    load_dotenv()

app = Flask(__name__)
CORS(app)

# Minimal flag to skip loading services until needed
services_initialized = False
retriever = None
llm = None
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

@app.route("/health")
def health():
    return "OK", 200 if services_initialized else "Initializing", 503

@app.route("/youtube-response", methods=["POST"])
def handle_comment():
    global services_initialized
    if not services_initialized:
        initialize_services()
        services_initialized = True

    data = request.get_json(silent=True)
    if not data or not data.get("comment"):
        return jsonify(error="Comment required"), 400

    try:
        response = generate_youtube_response(data["comment"])
        return jsonify(response=response)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify(error="Internal server error"), 500

def initialize_services():
    """Lazy-load heavy dependencies for fast startup"""
    global retriever, llm

    try:
        from pinecone import Pinecone
        from langchain_pinecone import PineconeVectorStore
        from langchain_openai import OpenAIEmbeddings, ChatOpenAI
        from langchain_core.prompts import PromptTemplate
        from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

        pinecone_api_key = os.environ["PINECONE_API_KEY"]
        pinecone_index_name = os.environ["PINECONE_INDEX_NAME"]
        openai_api_key = os.environ["OPENAI_API_KEY"]

        pc = Pinecone(api_key=pinecone_api_key)
        index = pc.Index(pinecone_index_name)
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vector_store = PineconeVectorStore(index, embeddings)

        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        llm = ChatOpenAI(
            model="gpt-3.5-turbo-0125",
            temperature=0.6,
            max_tokens=150,
            openai_api_key=openai_api_key
        )

        print("✅ Services initialized.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise RuntimeError("❌ Service initialization failed") from e

def generate_youtube_response(comment: str) -> str:
    from langchain_core.prompts import PromptTemplate
    from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

    docs = retriever.invoke(comment)
    context = "\n".join(d.page_content for d in docs) or "No specific context"
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = create_stuff_documents_chain(llm, prompt)
    return chain.invoke({"input": comment, "context": context}).strip()

if __name__ == "__main__":
    app.run(host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", 8080)))
