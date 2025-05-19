from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# Load environment variables from .env (for local development)
load_dotenv()

# Required environment variables: PINECONE_API_KEY, PINECONE_ENV (optional), PINECONE_INDEX, OPENAI_API_KEY
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV") or os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = os.getenv("PINECONE_INDEX")

if not PINECONE_API_KEY or not INDEX_NAME:
    raise RuntimeError("PINECONE_API_KEY and PINECONE_INDEX must be set as environment variables.")

# Initialize Pinecone client using Pinecone object (no deprecated pinecone.init)
import pinecone
if PINECONE_ENV:
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
else:
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
try:
    index = pc.Index(INDEX_NAME)
except Exception as e:
    raise RuntimeError(f"Failed to connect to Pinecone index '{INDEX_NAME}': {e}")

# Initialize LangChain components (OpenAI LLM and Pinecone vector store)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

# Embedding model for vector search (uses OpenAI Embeddings)
embedding_model = OpenAIEmbeddings()  # uses OPENAI_API_KEY from environment
# Create vector store from existing Pinecone index and embedding model
vector_store = PineconeVectorStore(index=index, embedding=embedding_model)  # :contentReference[oaicite:2]{index=2}

# Language model for reply generation (OpenAI GPT-3.5 Turbo)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# Define prompt templates for system and user messages
system_template = (
    "You are a helpful assistant that generates casual and friendly replies to YouTube comments. "
    "Keep responses short and conversational. "
    "Do not provide any medical advice. Do not mention prices or costs. "
    "Do not mention any celebrity names."
)
human_template = (
    "Relevant context (if any):\n{context}\n\n"
    "User comment: \"{comment}\"\n\n"
    "Write a short, casual reply to this comment."
)
chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(human_template)
])

# Create a chain that combines context documents and the LLM (using the 'stuff' method)
qa_chain = create_stuff_documents_chain(llm=llm, prompt=chat_prompt)  # :contentReference[oaicite:3]{index=3}

# Pydantic models for request and response bodies
class CommentRequest(BaseModel):
    query: str

class ReplyResponse(BaseModel):
    reply: str

# Instantiate FastAPI app
app = FastAPI()

@app.post("/youtube-reply", response_model=ReplyResponse)
def generate_reply(request: CommentRequest):
    """
    Generate a short, casual reply to the given YouTube comment.
    """
    user_query = request.query
    try:
        # Retrieve relevant documents from Pinecone via similarity search
        docs = vector_store.similarity_search(user_query, k=5)
        # Run the chain with retrieved docs and user comment to generate a reply
        result = qa_chain.invoke({"context": docs, "comment": user_query})
    except Exception as e:
        # Return HTTP 500 on any error (e.g., Pinecone or OpenAI issues)
        raise HTTPException(status_code=500, detail=str(e))
    # Return the reply text
    return {"reply": str(result).strip()}

# Run the app with Uvicorn for local testing (honoring PORT if set)
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
