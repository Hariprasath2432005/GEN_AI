# Import Required Libraries

import streamlit as st
import os
from dotenv import load_dotenv

# ML Layer
from sklearn.metrics.pairwise import cosine_similarity  #Compares user answer with ideal answer
import numpy as np   #Handles numerical data

# DL Layer
from sentence_transformers import SentenceTransformer    #Converts text into embeddings (numbers)
from langchain_groq import ChatGroq     #Connects to Groq LLM

# Resume Parsing
from pypdf import PdfReader   #Reads PDF resumes
import docx    #Reads Word resumes


# Load Environment Variables

load_dotenv()  #Loads API key securely from env file
GROQ_API_KEY = os.getenv("GROQ_API_KEY")



# Initialize Deep Learning Models
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant"
)

# Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")    


# Streamlit Page Configuration
st.set_page_config(page_title="AI Interview System", layout="wide")
st.title("🎓 AI-Powered Smart Interview & Resume Evaluation System")


# ----------------------------
# Resume Text Extraction Function
# ----------------------------

def extract_resume_text(uploaded_file):

    text = ""

    try:
        if uploaded_file.name.endswith(".pdf"):
            pdf = PdfReader(uploaded_file)
            for page in pdf.pages:
                if page.extract_text():
                    text += page.extract_text()
        elif uploaded_file.name.endswith(".docx"):
            doc = docx.Document(uploaded_file)
            for para in doc.paragraphs:
                text += para.text
    except Exception as e:
        st.error(f"Error reading file: {e}")
    return text

# ----------------------------
# Resume Evaluation
# ----------------------------

def evaluate_resume(resume_text, job_role):        #Creates a prompt message for the AI model
    prompt = f"""                                                                
    You are an expert HR.
    Evaluate this resume for the role of {job_role}.
    Resume:
    {resume_text}
    Provide:
    ATS Score out of 100
    Strengths
    Weaknesses
    Improvement Suggestions
    """
    response = llm.invoke(prompt)
    return response.content

# ----------------------------
# Generate Interview Questions
# ----------------------------

def generate_questions(job_role):

    prompt = f"""
    Generate:
    5 technical interview questions
    2 behavioral interview questions
    for the role of {job_role}.
    """
    response = llm.invoke(prompt)

    return response.content


# ----------------------------
# Answer Evaluation
# ----------------------------

def evaluate_answer(question, user_answer):

    ideal_prompt = f"""
    Provide a detailed ideal answer for this interview question:

    {question}
    """

    ideal_answer = llm.invoke(ideal_prompt).content

    user_embedding = embedding_model.encode([user_answer])
    ideal_embedding = embedding_model.encode([ideal_answer])

    similarity_score = cosine_similarity(
        user_embedding,
        ideal_embedding
    )[0][0]

    final_score = round(similarity_score * 10, 2)

    feedback_prompt = f"""
    Evaluate this interview answer.

    Question:
    {question}

    User Answer:
    {user_answer}

    Provide:

    Score out of 10
    Strength
    Weakness
    Improvement Suggestion
    """

    feedback = llm.invoke(feedback_prompt).content

    return final_score, feedback


# ----------------------------
# Sidebar Resume Section
# ----------------------------

st.sidebar.header("📄 Resume Section")

# ✅ ADDED INSTRUCTIONS HERE
st.sidebar.info("""
📌 Instructions:

1. Upload your resume (PDF or DOCX)
2. Enter your target job role
3. Click 'Evaluate Resume' to get ATS score
4. Click 'Generate Interview Questions' to practice interview
""")

uploaded_file = st.sidebar.file_uploader(   #Creates file upload button.
    "Upload Resume (PDF or DOCX)",
    type=["pdf", "docx"]
)

job_role = st.sidebar.text_input("Enter Job Role")    #Creates text box.

if uploaded_file and job_role:   #Ensures both inputs are given.
    resume_text = extract_resume_text(uploaded_file)   #Converts resume file into text.
    if st.sidebar.button("Evaluate Resume"):    #Creates button:
        with st.spinner("Analyzing Resume..."):   #Shows loading animation:
            result = evaluate_resume(resume_text, job_role)  #This sends data to AI model.
            st.subheader("📊 Resume Evaluation")
            st.write(result)


    if st.sidebar.button("Generate Interview Questions"):

        with st.spinner("Generating Questions..."):

            questions = generate_questions(job_role)

            st.subheader("🎤 Interview Questions")

            st.write(questions)


# ----------------------------
# Interview Practice Section
# ----------------------------

st.divider()

st.header("🧠 Practice Interview")

# ✅ ADDED INSTRUCTIONS HERE
st.info("""
📌 Instructions:

1. Enter an interview question
2. Type your answer in the answer box
3. Click 'Evaluate My Answer'
4. System will provide:
   • ML similarity score (0–10)
   • AI feedback
   • Strengths and weaknesses
   • Improvement suggestions
""")
question_input = st.text_input("Enter Interview Question")

answer_input = st.text_area("Type Your Answer")


if st.button("Evaluate My Answer"):
    if question_input and answer_input:
        with st.spinner("Evaluating Answer..."):
            score, feedback = evaluate_answer(
                question_input,
                answer_input
            )
            st.subheader("📈 Similarity Score (ML Based)")
            st.write(f"Score: {score}/10")
            st.subheader("📝 Detailed Feedback (LLM Based)")
            st.write(feedback)
    else:
        st.warning("Please enter both question and answer.")
