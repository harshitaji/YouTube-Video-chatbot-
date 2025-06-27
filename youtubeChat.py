import streamlit as st
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# Streamlit config
st.set_page_config(page_title="YouTube Q&A", layout="wide")
st.title("🎥 YouTube Video Q&A (RAG with Groq)")

# Initialize session state
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "llm" not in st.session_state:
    st.session_state.llm = None

# Input for Groq API Key and YouTube URL
api_key = st.text_input("🔐 Enter your Groq API Key", type="password")
video_url = st.text_input("📺 Enter YouTube Video URL", value="https://www.youtube.com/watch?v=MdeQMVBuGgY")

# Process the video
if st.button("📄 Process Video"):
    if not api_key:
        st.warning("Please enter your Groq API key.")
    else:
        try:
            with st.spinner("🔍 Loading transcript..."):
                loader = YoutubeLoader.from_youtube_url(video_url)
                docs = loader.load()

            with st.spinner("✂️ Splitting into chunks..."):
                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = splitter.split_documents(docs)

            with st.spinner("🔎 Creating embeddings..."):
                embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vector_db = FAISS.from_documents(chunks, embedding=embedder)

            with st.spinner("🤖 Connecting to LLM..."):
                llm = ChatGroq(groq_api_key=api_key, model="llama3-70b-8192")
                rag_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=vector_db.as_retriever(),
                    chain_type="stuff"
                )

            # Save state
            st.session_state.vector_db = vector_db
            st.session_state.llm = llm
            st.session_state.rag_chain = rag_chain
            st.session_state.chat_history = []  # clear old Q&A
            st.success("✅ Video processed! You can now ask questions.")

        except Exception as e:
            st.error(f"❌ Error: {e}")

# Question-answer loop
if st.session_state.rag_chain:
    st.subheader("💬 Ask a Question")
    query = st.text_input("Type your question here:")

    if query:
        with st.spinner("💡 Generating answer..."):
            try:
                response = st.session_state.rag_chain.invoke(query)
                answer = response["result"]

                # Store and display
                st.session_state.chat_history.append({"question": query, "answer": answer})
                st.markdown("### 🧠 Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"Error while generating answer: {e}")

    # Display full chat history
    if st.session_state.chat_history:
        st.subheader("📜 Chat History")
        for chat in st.session_state.chat_history:
            st.markdown(f"**🧑 You:** {chat['question']}")
            st.markdown(f"**🤖 AI:** {chat['answer']}")
