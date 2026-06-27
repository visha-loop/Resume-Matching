import streamlit as st
from utils import extract_text_from_pdf
#from predict import predict_match

st.set_page_config(
    page_title="AI Resume Matcher",
    page_icon="📄",
    layout="wide"
)

st.title("🧠 AI Resume Matcher")
st.write("Upload your Resume and paste the Job Description.")

st.divider()

resume = st.file_uploader(
    "📄 Upload Resume (PDF)",
    type=["pdf"]
)

job_description = st.text_area(
    "📋 Paste Job Description",
    height=250,
    placeholder="Paste the complete Job Description here..."
)

if st.button("🚀 Analyze Resume"):

    if resume is None:
        st.error("Please upload a resume.")
        st.stop()

    if job_description.strip() == "":
        st.error("Please paste a Job Description.")
        st.stop()

    resume_text = extract_text_from_pdf(resume)

    st.success("Resume uploaded successfully!")

    st.write("### Resume Preview")
    st.write(resume_text[:500])
    with st.spinner("Analyzing Resume..."):
        from predict import predict_match

    result = predict_match(
        resume_text,
        resume_text,
        job_description,
        resume_text
    )

    st.write(result)