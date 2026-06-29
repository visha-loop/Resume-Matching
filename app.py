import streamlit as st
import xgboost  # Import xgboost first to avoid OpenMP duplicate library conflict on macOS
from utils import extract_text_from_pdf
from predict import predict_match

st.set_page_config(
    page_title="AI Resume Matcher",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Inject custom CSS for premium design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Font Override */
    .stApp {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Custom Card Style */
    .custom-card {
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
    }
    
    /* Metric Card */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
        background: rgba(255, 255, 255, 0.08);
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        font-weight: 500;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f1f5f9;
    }
    
    /* Skill Badge */
    .skill-badge {
        display: inline-block;
        padding: 6px 14px;
        margin: 4px;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 600;
        background-color: rgba(239, 68, 68, 0.1);
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.25);
        transition: all 0.2s ease;
    }
    .skill-badge:hover {
        background-color: rgba(239, 68, 68, 0.2);
        border-color: rgba(239, 68, 68, 0.4);
    }
    
    .skill-badge-all {
        display: inline-block;
        padding: 6px 14px;
        margin: 4px;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 600;
        background-color: rgba(16, 185, 129, 0.1);
        color: #a7f3d0;
        border: 1px solid rgba(16, 185, 129, 0.25);
    }
</style>
""", unsafe_allow_html=True)

st.title("🧠 AI Resume Matcher")
st.markdown("##### Evaluate resume relevance and identify missing key skills using machine learning.")
st.divider()

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.subheader("📥 Upload & Input")
    
    resume = st.file_uploader(
        "Upload Resume (PDF only)",
        type=["pdf"],
        help="Upload the candidate's resume in PDF format."
    )
    
    job_description = st.text_area(
        "Job Description",
        height=300,
        placeholder="Paste the full job description here...",
        help="Enter the description of the role you are matching against."
    )
    
    analyze_btn = st.button("🚀 Analyze Resume", use_container_width=True)

with col2:
    st.subheader("📊 Analysis Report")
    
    if analyze_btn:
        if resume is None:
            st.error("Please upload a resume first.")
        elif job_description.strip() == "":
            st.error("Please paste the job description first.")
        else:
            with st.spinner("Processing files and running machine learning model..."):
                try:
                    resume_text = extract_text_from_pdf(resume)
                    
                    # Call backend predictor
                    result = predict_match(
                        resume_text,
                        resume_text,
                        job_description,
                        resume_text
                    )
                    
                    st.session_state['analysis_results'] = {
                        'result': result,
                        'resume_text': resume_text
                    }
                except Exception as e:
                    st.error(f"Error executing analysis: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    
    if 'analysis_results' in st.session_state:
        results_data = st.session_state['analysis_results']
        res = results_data['result']
        resume_txt = results_data['resume_text']
        
        score = res['prediction']
        semantic = res['semantic_similarity']
        career = res['career_similarity']
        bleu = res['bleu_score']
        exp = res['experience']
        missing = res['missing_skills']
        
        # Determine recommendations and styling
        if score >= 0.80:
            rec_text = "Strong Match"
            rec_color = "linear-gradient(135deg, #059669, #10b981)"
            rec_icon = "✅"
        elif score >= 0.60:
            rec_text = "Good Match"
            rec_color = "linear-gradient(135deg, #d97706, #f59e0b)"
            rec_icon = "🟡"
        else:
            rec_text = "Weak Match"
            rec_color = "linear-gradient(135deg, #dc2626, #ef4444)"
            rec_icon = "❌"
            
        # Large Match Score Banner
        st.markdown(f"""
        <div style="background: {rec_color}; padding: 28px; border-radius: 16px; text-align: center; color: white; margin-bottom: 25px; box-shadow: 0 10px 25px rgba(0,0,0,0.15);">
            <div style="font-size: 0.95rem; opacity: 0.85; font-weight: 600; text-transform: uppercase; letter-spacing: 1.5px;">Recommendation Report</div>
            <div style="font-size: 3.5rem; font-weight: 800; margin: 5px 0;">{score * 100:.1f}%</div>
            <div style="font-size: 1.3rem; font-weight: 700; background: rgba(255, 255, 255, 0.2); display: inline-block; padding: 6px 20px; border-radius: 50px; margin-top: 5px;">
                {rec_icon} {rec_text}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Grid layout for metrics
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        
        with m_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">🧠 Semantic</div>
                <div class="metric-val">{semantic:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">🎯 Career</div>
                <div class="metric-val">{career:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">📝 BLEU Score</div>
                <div class="metric-val">{bleu:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">💼 Experience</div>
                <div class="metric-val">{exp} yr{"s" if exp != 1 else ""}</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.divider()
        
        # Missing Skills Card
        st.subheader("📌 Missing Skills Analysis")
        if missing:
            st.write("The following key skills found in the Job Description were not detected in the resume:")
            badges_html = "".join([f'<span class="skill-badge">{skill}</span>' for skill in sorted(missing)])
            st.markdown(f'<div style="margin-top: 10px; margin-bottom: 20px;">{badges_html}</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="skill-badge-all" style="padding: 12px 20px; border-radius: 12px; display: block; text-align: center;">
                🎉 <b>Excellent!</b> Candidate possesses all the skills identified in the Job Description.
            </div>
            """, unsafe_allow_html=True)
            
        st.divider()
        
        # Resume Preview
        with st.expander("📄 Extracted Resume Text Preview", expanded=False):
            st.text_area("Extracted Text (First 1000 characters)", value=resume_txt[:1000], height=200, disabled=True)
            
    else:
        st.info("💡 Fill out the inputs on the left and click **Analyze Resume** to trigger prediction and view the matching statistics.")