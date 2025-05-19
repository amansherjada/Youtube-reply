from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import pinecone

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize Pinecone
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = pinecone.Index(PINECONE_INDEX_NAME)

# OpenAI Embeddings
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Vector Store & Retriever
vector_store = PineconeVectorStore(index=index, embedding=embedding_model)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key=OPENAI_API_KEY,
    temperature=0.7,
    max_tokens=100
)

# Generate a YouTube-friendly reply
def generate_youtube_reply(query):
    try:
        documents = retriever.get_relevant_documents(query)
        context_text = "\n\n".join([doc.page_content for doc in documents]) or "No relevant info found."

        prompt_template = """
You are the YouTube comment responder for American Hairline.
Your tone is friendly, short, and conversational. Don’t sound robotic. Make the brand feel human.
Use the info below to answer comments or questions related to hair patches, non-surgical hair replacement, or hair systems.

✅ Avoid:
- Exact prices
- Medical advice
- Revealing celebrity names
- Complex instructions

✅ Do:
- Use casual tone like: “Hey! Thanks for asking 😊” or “Glad you noticed!”
- Mention WhatsApp only if question is detailed: +91 9222666111
- Keep it within 2-3 short lines.

Context:
{context}

YouTube Comment:
{question}

Reply:
        """

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        qa_chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

        result = qa_chain({
            "input_documents": documents,
            "context": context_text,
            "question": query
        })

        return result["output_text"].strip()

    except Exception as e:
        print(f"Error: {e}")
        return "Thanks for your comment! 😊 Our team will get back to you soon."

# Endpoint for YouTube comment webhook
@app.route("/youtube-reply", methods=["POST"])
def youtube_comment_reply():
    data = request.get_json()
    query = data.get("query")

    if not query:
        return jsonify({"response": "Please provide a comment."})

    reply = generate_youtube_reply(query)
    return jsonify({"response": reply})

# Cloud Run compatibility
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=True, host="0.0.0.0", port=port)
