import streamlit as st
import os
import time
from dotenv import load_dotenv

st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("📄 PDF Question Answering System")

load_dotenv()

uploaded_file = st.file_uploader("📤 Upload a PDF", type="pdf")
question = st.text_input("🤔 Ask a question")

if st.button("🔍 Search Answer"):

    if uploaded_file is None:
        st.warning("Please upload a PDF")
        st.stop()

    if not question:
        st.warning("Please enter a question")
        st.stop()

    with st.spinner("⏳ Processing PDF..."):

        # Save uploaded PDF
        pdf_path = "uploaded.pdf"
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # SAFE IMPORTS
        from langchain_groq import ChatGroq
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_core.prompts import ChatPromptTemplate

        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        vectordb = FAISS.from_documents(docs, embeddings)
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            st.error("❌ GROQ_API_KEY missing in .env")
            st.stop()

        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.1-8b-instant",
            temperature=0
        )

        prompt = ChatPromptTemplate.from_template(
            """
            Answer the question using only the context below.

            Context:
            {context}

            Question:
            {question}

            Answer:
            """
        )

        start = time.time()

        # ✅ FIXED LINE (IMPORTANT)
        retrieved_docs = retriever.invoke(question)

        context = "\n\n".join([d.page_content for d in retrieved_docs])

        answer = llm.invoke(
            prompt.format(context=context, question=question)
        )

        elapsed = time.time() - start

    st.subheader("💡 Answer")
    st.write(answer.content)
    st.info(f"⏱️ Response time: {elapsed:.2f} seconds")

    with st.expander("📄 Source Chunks"):
        for i, doc in enumerate(retrieved_docs):
            st.markdown(f"**Chunk {i+1}**")
            st.write(doc.page_content)

# Instructions
st.sidebar.header("📖 Instructions")
st.sidebar.write("""
1. Make sure you have a PDF file in the same folder
2. Enter your question in the text box
3. Click 'Search Answer' button
4. Wait for the response
""")

st.sidebar.header("⚙️ Setup Required")
st.sidebar.write("""
You need to create a `.env` file with the following variables:

- GROQ_API_KEY
- OPEN_AI_KEY
- HUGGINGFACE_API_KEY

You can get these from the following services:

- Groq: https://groq.com/
- OpenAI: https://openai.com/
- HuggingFace: https://huggingface.co/
""")