import random
import streamlit as st
import google.generativeai as genai
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tempfile
import os
import PyPDF2
from docx import Document

# -------------------------------
# Rule-based question bank
# -------------------------------
question_bank = {
    "python": [
        "What are Python decorators?",
        "Explain the difference between list and tuple.",
        "How does garbage collection work in Python?",
        "What are Python generators?",
        "Explain Python's Global Interpreter Lock (GIL)."
    ],
    "sql": [
        "What is the difference between INNER JOIN and LEFT JOIN?",
        "Explain the concept of normalization.",
        "What are indexes and why are they used?",
        "What is the difference between WHERE and HAVING?",
        "Explain ACID properties in databases."
    ],
    "ml": [
        "What is the difference between supervised and unsupervised learning?",
        "Explain the bias-variance tradeoff.",
        "What is regularization in machine learning?",
        "What are precision, recall, and F1-score?",
        "What is gradient descent and how does it work?"
    ]
}

# -------------------------------
# Resume text extractor
# -------------------------------
def extract_text_from_resume(uploaded_file):
    text = ""
    if uploaded_file.name.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        for page in pdf_reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    elif uploaded_file.name.endswith(".docx"):
        doc = Document(uploaded_file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    else:
        text = "⚠️ Unsupported file format. Please upload PDF or DOCX."
    return text.strip()

# -------------------------------
# AI-based generator (Gemini)
# -------------------------------
def ai_based_generator(api_key, role, skills_text, num_qs):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = f"""
        Generate {num_qs} interview questions for a {role} role.
        Use the following skills and experience from the candidate's resume:
        {skills_text}

        Provide only the questions in numbered list format.
        """

        response = model.generate_content(prompt)

        if response and response.text:
            questions = response.text.strip().split("\n")
            return [q.strip() for q in questions if q.strip()]
        else:
            return ["⚠️ No response from Gemini. Try again."]
    except Exception as e:
        return [f"⚠️ Error: {str(e)}"]

# -------------------------------
# Rule-based generator
# -------------------------------
def rule_based_generator(skills, num_qs):
    questions = []
    for skill in skills.split(","):
        skill = skill.strip().lower()
        if skill in question_bank:
            questions.extend(question_bank[skill])
    if not questions:
        return ["⚠️ No predefined questions available for given skills."]
    return random.sample(questions, min(num_qs, len(questions)))

# -------------------------------
# PDF Export Function
# -------------------------------
def save_questions_to_pdf(questions, role, skills_text):
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    c = canvas.Canvas(tmp_file.name, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, f"Interview Prep Guide for {role}")
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 70, f"Extracted Skills/Resume Summary:")
    text_lines = skills_text.split("\n")[:5]  # only first 5 lines to keep it clean
    y = height - 90
    for line in text_lines:
        c.drawString(50, y, line[:90])
        y -= 15

    y -= 20
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, "Interview Questions:")
    y -= 20

    for i, q in enumerate(questions, 1):
        if y < 100:  # new page
            c.showPage()
            y = height - 50
        c.setFont("Helvetica", 11)
        c.drawString(50, y, f"{i}. {q}")
        y -= 20

    c.save()
    return tmp_file.name

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🤖 Resume-Based AI Interview Question Generator")

mode = st.radio("Choose Mode:", ["Rule-Based (Free)", "AI-Powered (Gemini API Key)"])

role = st.text_input("Enter the Job Role:", "Data Scientist")
skills = st.text_input("Enter Skills (comma-separated):", "python, sql, ml")
num_qs = st.slider("Number of Questions:", 1, 10, 5)

# Resume Upload
uploaded_resume = st.file_uploader("📂 Upload your Resume (PDF or DOCX)", type=["pdf", "docx"])
resume_text = ""
if uploaded_resume is not None:
    resume_text = extract_text_from_resume(uploaded_resume)
    st.text_area("📄 Extracted Resume Content:", resume_text[:1000], height=200)

if mode == "AI-Powered (Gemini API Key)":
    api_key = st.text_input("Enter your Gemini API Key:", type="password")
else:
    api_key = None

if st.button("Generate Questions"):
    if mode == "Rule-Based (Free)":
        qs = rule_based_generator(skills, num_qs)
    else:
        if not api_key:
            st.error("Please enter a Gemini API key.")
            qs = []
        else:
            input_text = resume_text if resume_text else skills
            qs = ai_based_generator(api_key, role, input_text, num_qs)

    if qs:
        st.subheader("📋 Generated Interview Questions:")
        for i, q in enumerate(qs, 1):
            st.write(f"{i}. {q}")

        # PDF Export button
        pdf_path = save_questions_to_pdf(qs, role, resume_text if resume_text else skills)
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="📄 Download Interview Prep as PDF",
                data=f,
                file_name=f"{role}_interview_prep.pdf",
                mime="application/pdf"
            )
        os.remove(pdf_path)  # clean up
