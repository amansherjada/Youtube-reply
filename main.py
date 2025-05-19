from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from pinecone import Pinecone
import os

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Validations
if not OPENAI_API_KEY:
    raise RuntimeError("❌ OPENAI_API_KEY not found in environment.")
if not PINECONE_API_KEY:
    raise RuntimeError("❌ PINECONE_API_KEY not found in environment.")
if not PINECONE_INDEX_NAME:
    raise RuntimeError("❌ PINECONE_INDEX_NAME not found in environment.")

# Initialize clients
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vector_store = PineconeVectorStore(index=index, embedding=embedding_model)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY, temperature=0.7, max_tokens=100)

# FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Core function
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
        print(f"Error generating response: {e}")
        return "Thanks for your comment! 😊 Our team will get back to you soon."

# Endpoint
@app.post("/youtube-reply")
async def youtube_comment_reply(request: Request):
    try:
        data = await request.json()
        query = data.get("query")
        if not query:
            return JSONResponse(content={"response": "Please provide a comment."}, status_code=400)
        reply = generate_youtube_reply(query)
        return JSONResponse(content={"response": reply})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
