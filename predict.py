import lightgbm as lgb
# import xgboost  # Kept for reference/backward compatibility
import pandas as pd
import numpy as np
import joblib
import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
#clean_df = pd.read_csv("clean_dataset.csv")

#resume_embeddings = np.load("resume_embeddings.npy")

#job_embeddings = np.load("job_embeddings.npy")

#xgb_model = joblib.load("resume_matching_xgboost.pkl")

#embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Step 1")

clean_df = pd.read_csv("clean_dataset.csv")

print("Step 2")

resume_embeddings = np.load("resume_embeddings.npy")

print("Step 3")

job_embeddings = np.load("job_embeddings.npy")

print("Step 4")

# xgb_model = joblib.load("resume_matching_xgboost.pkl")  # Kept for reference
lgb_model = joblib.load("resume_matching_lightgbm.pkl")
print("Step 5")

embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

print("Step 6")
def calculate_semantic_similarity(resume, job):

    resume_embedding = embedding_model.encode([resume])

    job_embedding = embedding_model.encode([job])

    similarity = cosine_similarity(
        resume_embedding,
        job_embedding
    )[0][0]

    return similarity
def calculate_career_similarity(career_objective, job_title):

    career_embedding = embedding_model.encode([career_objective])

    job_embedding = embedding_model.encode([job_title])

    similarity = cosine_similarity(
        career_embedding,
        job_embedding
    )[0][0]

    return similarity
  
def calculate_bleu(resume, job):

    smoothie = SmoothingFunction().method1

    score = sentence_bleu(
        [job.lower().split()],
        resume.lower().split(),
        smoothing_function=smoothie
    )

    return score
import re

def extract_experience(text):

    text = text.lower()

    patterns = [

        r'(\d+)\+?\s+years?',
        r'(\d+)\+?\s+yrs?',
        r'(\d+)\+?\s+year',

    ]

    for pattern in patterns:

        match = re.search(pattern, text)

        if match:
            years = int(match.group(1))

            if years <= 50:        # Ignore phone numbers
                return years

    return 0
SKILLS = {
    "Python", "Java", "C++", "SQL", "MySQL", "PostgreSQL", "MongoDB",
    "Machine Learning", "Deep Learning", "Artificial Intelligence",
    "TensorFlow", "PyTorch", "Keras", "Scikit-Learn", "NumPy", "Pandas",
    "Matplotlib", "FastAPI", "Flask", "Django", "React", "Next.js",
    "Node.js", "Docker", "Kubernetes", "AWS", "Azure", "GCP", "Git",
    "GitHub", "Linux", "LangChain", "LangGraph", "OpenCV", "Hugging Face"
}

def extract_skills(text):
    text = text.lower()
    found = set()
    for skill in SKILLS:
        if skill.lower() in text:
            found.add(skill)
    return found

def predict_match(resume_text,
                  career_objective,
                  job_description,
                  experience_text):

    # Graceful handling for empty/missing inputs
    if not resume_text or not resume_text.strip():
        raise ValueError("Resume text content is empty. Please upload a valid PDF resume containing readable text.")
    if not job_description or not job_description.strip():
        raise ValueError("Job description is empty. Please paste a valid job description.")

    print("A - Starting predict_match")

    semantic = calculate_semantic_similarity(
        resume_text,
        job_description
    )

    print("B - Semantic done")

    career = calculate_career_similarity(
        career_objective,
        job_description
    )

    print("C - Career done")

    bleu = calculate_bleu(
        resume_text,
        job_description
    )

    print("D - BLEU done")

    experience = extract_experience(
        experience_text
    )

    print("E - Experience done")

    # Compute skill overlap
    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_description)
    matched_skills = list(resume_skills.intersection(job_skills))
    missing_skills = list(job_skills - resume_skills)

    if len(job_skills) > 0:
        skill_overlap = len(matched_skills) / len(job_skills)
    else:
        skill_overlap = 0.0

    # Prediction dataframe with exactly 5 features in correct order
    sample = pd.DataFrame({
        "semantic_similarity": [semantic],
        "career_similarity": [career],
        "required_experience": [experience],
        "bleu_score": [bleu],
        "skill_overlap": [skill_overlap]
    })

    print("F - DataFrame created")

    # LightGBM inference
    prediction = lgb_model.predict(sample)[0]

    # Clip match score to a realistic percentage [0.0, 1.0]
    match_score = float(max(0.0, min(1.0, prediction)))

    print("G - Prediction done")
    print("=" * 60)
    print("              AI RESUME MATCH REPORT")
    print("=" * 60)

    print(f"\n🎯 Match Score : {match_score*100:.2f}%")

    print("\n-------------------------------")

    print(f"🧠 Semantic Similarity : {semantic:.3f}")
    print(f"🎯 Career Similarity   : {career:.3f}")
    print(f"📝 BLEU Score          : {bleu:.3f}")
    print(f"💼 Experience          : {experience} years")
    print(f"📊 Skill Overlap       : {skill_overlap:.3f}")

    print("\n===============================")

    # Recommendation logic
    if match_score >= 0.80:
        recommendation = "Strong Match"
    elif match_score >= 0.60:
        recommendation = "Good Match"
    else:
        recommendation = "Weak Match"

    print(f"📌 Recommendation : {recommendation}")
    print(f"📌 Missing Skills : {missing_skills}")

    confidence = round(match_score * 100, 2)

    return {
        "match_score": match_score,
        "semantic_similarity": float(semantic),
        "career_similarity": float(career),
        "bleu_score": float(bleu),
        "required_experience": int(experience),
        "skill_overlap": float(skill_overlap),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "recommendation": recommendation,
        "confidence": confidence
    }
