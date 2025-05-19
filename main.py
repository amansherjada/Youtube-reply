from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from pinecone import Pinecone, ServerlessSpec
import os

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")  # Default to 'us-east-1'

# Validations
if not OPENAI_API_KEY:
    raise RuntimeError("❌ OPENAI_API_KEY not found in environment.")
if not PINECONE_API_KEY:
    raise RuntimeError("❌ PINECONE_API_KEY not found in environment.")
if not PINECONE_INDEX_NAME:
    raise RuntimeError("❌ PINECONE_INDEX_NAME not found in environment.")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT),
    )
index = pc.Index(PINECONE_INDEX_NAME)

# Initialize embedding model
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Initialize vector store
vector_store = PineconeVectorStore(index=index, embedding=embedding_model)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# Initialize language model
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY, temperature=0.7, max_tokens=100)

# Define prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
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
"""),
    ("human", "{input}")
])

# Create document chain
document_chain = create_stuff_documents_chain(llm, prompt_template)

# Create retrieval chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define endpoint
@app.post("/youtube-reply")
async def youtube_comment_reply(request: Request):
    try:
        data = await request.json()
        query = data.get("query")
        if not query:
            return JSONResponse(content={"response": "Please provide a comment."}, status_code=400)
        result = retrieval_chain.invoke({"input": query})
        return JSONResponse(content={"response": result["answer"].strip()})
    except Exception as e:
        print(f"Error generating response: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
