import streamlit as st
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq         # For Groq AI; replace if using OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os

# Load environment variables (ensure .env is configured)
from dotenv import load_dotenv
load_dotenv()

# Constants / Environment variables
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROK_API_KEY = os.environ.get('GROK_API_KEY')

if not PINECONE_API_KEY or not GROK_API_KEY:
    st.error("API keys for Pinecone or Groq/Grok AI are not set in environment variables.")
    st.stop()

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY if PINECONE_API_KEY is not None else ""
os.environ["GROK_API_KEY"] = GROK_API_KEY if GROK_API_KEY is not None else ""

# Initialization (cache for performance)
@st.cache_resource
def init_doc_search():
    embeddings = download_embeddings()
    index_name = "medical-chatbot"
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    return docsearch

docsearch = init_doc_search()
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize Chat Model (using Groq AI - adjust if you use OpenAI)
chatModel = ChatGroq(model="llama3-8b-8192")

# Your system prompt defined in src.prompt or redefine briefly here
system_prompt = (
    "You are a Medical assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise.\n\n{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Streamlit UI
st.title("Medical Chatbot with LangChain and Groq AI")
st.write("Ask your medical questions, and the chatbot will assist you!")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input = st.text_input("You:", key="input")

if user_input:
    # Show user message
    st.session_state.chat_history.append(("You", user_input))

    # Get chatbot response
    with st.spinner("Fetching response..."):
        response = rag_chain.invoke({"input": user_input})
        answer = response["answer"]

    st.session_state.chat_history.append(("Bot", answer))

# Display chat history
for sender, msg in st.session_state.chat_history:
    if sender == "You":
        st.markdown(f"**{sender}:** {msg}")
    else:
        st.markdown(f"**{sender}:** {msg}")
