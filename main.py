import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# New imports for Google Generative AI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
FAISS_INDEX_PATH = "faiss_index" # Must match the path from process_pdf.py
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Ensure the Google API key is available
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file or environment variables.")

# --- FastAPI App Initialization ---
app = FastAPI(
    title="PDF Chatbot API",
    description="API for asking questions to your PDF documents using RAG.",
    version="1.0.0",
)

# --- Global Variables for Loaded Models ---
embeddings = None
docsearch = None
llm = None # Will be ChatGoogleGenerativeAI
rag_chain = None
prompt = None # Declare prompt globally


# Helper function to retrieve documents and print them for debugging
def retrieve_documents(question: str):
    docs = docsearch.similarity_search(question)
    print(f"\n--- Retrieved Documents for Question: '{question}' ---")
    if docs:
        for i, doc in enumerate(docs):
            # Safely get source, default to N/A if not present
            source_info = doc.metadata.get('source', 'N/A')
            # Limit page_content to avoid flooding console with very long documents
            print(f"Document {i+1} (Source: {source_info}):\n{doc.page_content[:500]}...\n") # Print first 500 chars
    else:
        print("No documents found for this query.")
    print("--------------------------------------------------\n")
    return docs


# --- Startup Event: Load Models ---
@app.on_event("startup")
async def startup_event():
    """
    Load the FAISS index, embedding model, and LLM chain once when the application starts.
    """
    global embeddings, docsearch, llm, rag_chain, prompt

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    print(f"Loading FAISS index from {FAISS_INDEX_PATH}...")
    if not os.path.exists(FAISS_INDEX_PATH):
        print(f"Error: FAISS index directory '{FAISS_INDEX_PATH}' not found.")
        print("Please run process_pdf.py first to create the index.")
        exit(1) # Exit if the index is not found

    docsearch = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    print("FAISS index loaded successfully!")

    print("Loading LLM: Google Gemini Flash (gemini-1.5-flash)...")
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.1)
    print("LLM loaded successfully (Google Gemini Flash)!")

    # Define the Prompt Template for RAG for ChatGoogleGenerativeAI
    # Define the Prompt Template for RAG for ChatGoogleGenerativeAI
    prompt = ChatPromptTemplate.from_template("""
    You are an AI assistant for question-answering over documents.
    Use the following retrieved context to answer the question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Answer in a concise and clear manner.

    Context:
    {context}

    Question: {question}

    Answer:
    """)

    # Define the RAG Chain using LangChain Expression Language (LCEL)
    rag_chain = (
        {"context": RunnableLambda(lambda x: retrieve_documents(x["question"])),
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("RAG chain constructed successfully!")


# --- Request Body Model for Chat Endpoint ---
class ChatRequest(BaseModel):
    query: str

# --- API Endpoint: Chat ---
@app.post("/chat")
async def chat_with_pdf(request: ChatRequest):
    """
    Answers questions based on the content of the processed PDF documents.
    """
    if not docsearch or not rag_chain:
        return JSONResponse(
            status_code=500,
            content={"error": "Chatbot models are not loaded. Server is still starting or encountered an error during startup."}
        )

    try:
        print(f"Processing query: {request.query}")
        answer = rag_chain.invoke({"question": request.query})
        print("Answer generated.")

        return JSONResponse(
            status_code=200,
            content={"answer": answer}
        )
    except Exception as e:
        import sys
        import traceback

        print(f"An error occurred during chat processing:")
        print(f"Error Type: {type(e)}")
        print(f"Error Args: {e.args}")
        traceback.print_exc(file=sys.stdout)

        return JSONResponse(
            status_code=500,
            content={"error": f"An internal server error occurred: {str(e)}"}
        )

# --- Root Endpoint (Optional, for testing server status) ---
@app.get("/")
async def read_root():
    return {"message": "PDF Chatbot API is running. Go to /docs for API documentation."}