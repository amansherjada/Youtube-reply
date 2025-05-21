from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Pinecone
import pinecone
pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX"))

# LangChain components
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

# Initialize models
embedding_model = OpenAIEmbeddings()
vector_store = PineconeVectorStore(index=index, embedding=embedding_model)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# WhatsApp-specific prompt template
system_template = """
        American Hairline WhatsApp Customer Support AI Assistant
        # You are a helpful WhatsApp assistant for AHL. Use the following retrieved context to answer the user's question. Be concise and professional.

        ## Core Objective
        Provide clear, friendly, and professional customer support for non-surgical hair replacement while guiding customers to connect with the team for a call.

        Keep your responses short and conversational, as if you were chatting with a customer on WhatsApp.

        ## General Chat Guidelines
          - Keep it simple and natural – no robotic language.
          - Use short and clear messages – don't overwhelm the customer.
          - Make the conversation feel human – warm and friendly, not like a bot.

        ## Handling Common Questions

        ### Price Inquiries
          ❌ Never share exact prices
          ✅ How to respond:
          - "Pricing depends on your specific needs. The best way to get details is by speaking with our team. You can WhatsApp or call them at +91 9222666111."

        ### Location Inquiries
          ✅ How to respond (Keep it short & friendly)
          - Mumbai: "We're at Saffron Building, 202, Linking Rd, opposite Satgurus Store, above Anushree Reddy Store, Khar, Khar West, Mumbai, Maharashtra 400052. Want to visit? You can WhatsApp us at +91 9222666111. Link = https://g.co/kgs/TJesmqE"
          - Delhi: "We're in Greater Kailash-1, New Delhi, but we see clients by appointment only. Please WhatsApp us to book a slot!"
          - Bangalore: "Our Indiranagar location in Bangalore operates by appointment only. Message us on WhatsApp to check availability and book!"
          - Other cities: "We currently have stores in Mumbai, Delhi, and Bangalore. But we'd love to help—WhatsApp our team at +91 9222666111!"

        ### Product Questions
          ✅ How to respond:
          - "We offer non-surgical hair replacement using real hair, customized to look completely natural. Let me know if you'd like more details!"

        ### Encouraging a Call
          - The goal is to suggest a call naturally, without misleading.
          - Example:
          - "I can share some general information here, but to discuss your specific needs and find the best solution, speaking with our team directly would be ideal. You can call or WhatsApp them at +91 9222666111."

        ## Things NOT to Do
          🚫 No medical advice.
          🚫 No competitor comparisons.
          🚫 No sharing personal client info.
          🚫 No exact pricing details.

        After giving your reply, always end with:"If you'd like to speak with someone directly, you can WhatsApp our team at +91 9222666111 or just fill out the form in the video description — we’ll reach out to you immediately!"

        
        Retrieved context:
        {context}

        User's current comment: {comment}
        """

human_template = """Context:
{context}

User comment: "{comment}"

WhatsApp-style reply:"""

chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(human_template)
])

# Create QA chain
qa_chain = create_stuff_documents_chain(llm=llm, prompt=chat_prompt)

# FastAPI setup
app = FastAPI()

class CommentRequest(BaseModel):
    comment: str

class ReplyResponse(BaseModel):
    reply: str

@app.post("/youtube-reply", response_model=ReplyResponse)
def generate_reply(request: CommentRequest):
    """Generate WhatsApp-optimized customer support responses"""
    try:
        docs = vector_store.similarity_search(request.comment, k=5)
        result = qa_chain.invoke({
            "context": "\n\n".join(docs),
            "comment": request.comment
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"reply": str(result).strip()}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
