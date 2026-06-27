#import joblib
#import numpy as np

#model = joblib.load("resume_matching_xgboost.pkl")
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

#xgb_model = joblib.load("resume_matching_xgboost.pkl")
xgb_model = "test"
#xgb_model = None
print("Step 5")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

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
    "python","java","c++","c","sql","mysql","postgresql","mongodb",
    "machine learning","deep learning","artificial intelligence",
    "tensorflow","pytorch","keras","scikit-learn",
    "pandas","numpy","matplotlib","seaborn",
    "fastapi","flask","django",
    "react","next.js","node.js",
    "git","github","linux",
    "aws","azure","docker","kubernetes",
    "langchain","langgraph","hugging face","opencv"
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

    semantic = calculate_semantic_similarity(
        resume_text,
        job_description
    )

    career = calculate_career_similarity(
        career_objective,
        job_description
    )

    bleu = calculate_bleu(
        resume_text,
        job_description
    )

    experience = extract_experience(
        experience_text
    )

    sample = pd.DataFrame({
        "semantic_similarity": [semantic],
        "career_similarity": [career],
        "required_experience": [experience],
        "bleu_score": [bleu]
    })

    prediction = xgb_model.predict(sample)[0]
    #prediction = 0.85
    print("=" * 60)
    print("              AI RESUME MATCH REPORT")
    print("=" * 60)

    print(f"\n🎯 Match Score : {prediction*100:.2f}%")

    print("\n-------------------------------")

    print(f"🧠 Semantic Similarity : {semantic:.3f}")
    print(f"🎯 Career Similarity   : {career:.3f}")
    print(f"📝 BLEU Score          : {bleu:.3f}")
    print(f"💼 Experience          : {experience} years")

    print("\n===============================")

    if prediction >= 0.80:
        print("✅ Recommendation : Strong Match")
    elif prediction >= 0.60:
        print("🟡 Recommendation : Good Match")
    else:
        print("❌ Recommendation : Weak Match")

    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_description)
    missing = job_skills - resume_skills
    print("\n📌 Missing Skills")

    if len(missing) == 0:
        print("None 🎉")
    else:
        for skill in sorted(missing):
            print("-", skill)

    return float(prediction)
