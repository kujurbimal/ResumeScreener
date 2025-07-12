import streamlit as st
import PyPDF2
import docx2txt
import io
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Page config
st.set_page_config(page_title="Resume Screener", layout="wide")
st.title("📄 Resume Screener")
st.markdown("Upload a resume and job description to check how well they match.")

# File uploads
resume_file = st.file_uploader("Upload Resume (PDF or DOCX)", type=["pdf", "docx"])
job_description = st.text_area("Paste the Job Description here", height=200)

# Text extraction function
def extract_text(file):
    text = ""
    if file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    elif file.name.endswith(".docx"):
        text = docx2txt.process(file)
    return text

# Screening logic
if resume_file and job_description:
    resume_text = extract_text(resume_file)

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = [w for w in text.split() if w not in stop_words and len(w) > 2]
        return " ".join(tokens)

    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(job_description)

    # Vectorize
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([resume_clean, jd_clean])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    # Show score
    st.subheader("📊 Match Score")
    match_percentage = round(similarity * 100, 2)
    st.metric(label="Resume-Job Match", value=f"{match_percentage}%")

    # Suggestions
    st.subheader("💡 Suggestions")
    if match_percentage > 70:
        st.success("✅ Strong match! Your resume aligns well.")
    elif match_percentage > 40:
        st.warning("⚠️ Decent match. Consider adding more keywords.")
    else:
        st.error("❌ Low match. Customize your resume further.")

else:
    st.info("Please upload a resume and paste the job description to begin.")
