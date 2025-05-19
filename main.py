# main.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore  # Updated import
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from pinecone import Pinecone  # Updated client initialization

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize Pinecone client (new method)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# OpenAI Embeddings
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Vector Store & Retriever
vector_store = PineconeVectorStore(index, embedding_model)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# LLM with fixed parameters
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=OPENAI_API_KEY,
    temperature=0.7,
    max_tokens=100
)  # Fixed missing parenthesis

def generate_youtube_reply(query):
    try:
        # Updated document retrieval
        documents = retriever.invoke(query)
        context_text = "\n\n".join([doc.page_content for doc in documents]) or "No relevant info found."
        
        prompt_template = """
        You are the YouTube comment responder for American Hairline.
        Your tone is friendly, short, and conversational. Don’t sound robotic.
        Context: {context}
        YouTube Comment: {question}
        Reply:"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        qa_chain = create_stuff_documents_chain(llm, prompt)
        result = qa_chain.invoke({"context": context_text, "question": query})
        return result.strip()
    
    except Exception as e:
        print(f"Error: {e}")
        return "Thanks for your comment! 😊 Our team will get back to you soon."

@app.route("/youtube-reply", methods=["POST"])
def youtube_comment_reply():
    data = request.get_json()
    query = data.get("query")
    if not query:
        return jsonify({"response": "Please provide a comment."})
    reply = generate_youtube_reply(query)
    return jsonify({"response": reply})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
