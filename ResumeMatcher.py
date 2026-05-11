import math
import re
from collections import Counter

# =========================
# SKILL ALIASES
# =========================

SKILL_ALIASES = {
    # Languages
    "python": "python",
    "pyhton": "python",
    "java": "java",
    "javascript": "javascript",
    "javascrpit": "javascript",
    "js": "javascript",
    "typescript": "typescript",
    "typescrpit": "typescript",
    "c++": "cpp",
    "cpp": "cpp",
    "r": "r",
    "kotlin": "kotlin",

    # ML / Data
    "machinelearning": "machine_learning",
    "machine learning": "machine_learning",
    "ml": "machine_learning",
    "sklearn": "machine_learning",

    "deeplearning": "deep_learning",
    "deep learning": "deep_learning",
    "deep-learning": "deep_learning",

    "tensorflow": "tensorflow",
    "pytorch": "pytorch",
    "keras": "keras",

    "nlp": "nlp",
    "bert": "bert",
    "xgboost": "xgboost",

    "feature engineering": "feature_engineering",

    "statistics": "statistics",
    "stats": "statistics",

    "regression": "regression",
    "clustering": "clustering",

    "data-viz": "data_visualization",
    "data visualization": "data_visualization",
    "data viz": "data_visualization",
    "matplotlib": "data_visualization",
    "tableau": "data_visualization",
    "power-bi": "data_visualization",
    "power bi": "data_visualization",
    "powerbi": "data_visualization",

    "pandas": "pandas",
    "numpy": "numpy",

    # Frontend
    "react": "react",
    "reacts": "react",
    "reactjs": "react",

    "vue": "vue",
    "vue.js": "vue",
    "vuejs": "vue",

    "redux": "redux",
    "tailwind": "tailwind",

    "html/css": "html_css",
    "html css": "html_css",
    "html": "html_css",
    "css": "html_css",

    "jest": "jest",
    "graphql": "graphql",

    # Backend
    "node.js": "nodejs",
    "nodejs": "nodejs",
    "node js": "nodejs",

    "flask": "flask",

    "spring boot": "spring_boot",
    "springboot": "spring_boot",

    "rest api": "rest_api",
    "rest": "rest_api",
    "restapi": "rest_api",

    "microservices": "microservices",

    # DB
    "sql": "sql",
    "mysql": "mysql",
    "mysq": "mysql",

    "postgresql": "postgresql",
    "postgres": "postgresql",

    "mongodb": "mongodb",
    "redis": "redis",

    # DevOps
    "docker": "docker",

    "kubernetes": "kubernetes",
    "kubernates": "kubernetes",
    "k8s": "kubernetes",

    "ci/cd": "ci_cd",
    "cicd": "ci_cd",
    "ci cd": "ci_cd",

    "aws": "aws",

    # Mobile
    "android": "android",
    "firebase": "firebase",

    # CS
    "algorithms": "algorithms",
    "algoritms": "algorithms",

    "data structure": "data_structures",
    "data structures": "data_structures",

    "competitive programming": "competitive_programming",

    # Design
    "ui/ux": "ui_ux",
    "ui ux": "ui_ux",
    "figma": "figma",
}

# =========================
# RESUME DATA
# =========================

RESUME_DATASET = [
    {"id": "01", "name": "Arjun Sharma", "skills": "Pyhton, MachineLearning, SQL, pandas, numpy, Deep-learning"},
    {"id": "02", "name": "Priya Nair", "skills": "JavaScrpit, Reacts, Node.JS, MongoDb, REST api, HTML/CSS"},
    {"id": "03", "name": "Rahul Gupta", "skills": "Java, Spring Boot, MySql, Microservices, Docker, kubernates"},
    {"id": "04", "name": "Sneha Patel", "skills": "Python, TensorFlow, Keras, NLP, BERT, data-viz, matplotlib"},
    {"id": "05", "name": "Vikram Singh", "skills": "C++, Algoritms, Data Structure, competitive programming, python"},
    {"id": "06", "name": "Ananya Krishnan", "skills": "javascript, vue.js, python, flask, PostgreSQL, AWS, CI/CD"},
    {"id": "07", "name": "Karan Mehta", "skills": "Python, Sklearn, XGboost, feature engineering, SQL, tableau"},
    {"id": "08", "name": "Deepika Rao", "skills": "Java, Android, Kotlin, Firebase, REST, UI/UX, figma"},
    {"id": "09", "name": "Aditya Kumar", "skills": "Reactjs, TypeScrpit, GraphQL, redux, tailwind, nodejs, jest"},
    {"id": "10", "name": "Meera Iyer", "skills": "python, R, statistics, ML, regression, clustering, Power-BI"},
]

# =========================
# JOB DESCRIPTIONS
# =========================

JOBDESCRIPTIONDATASET = [
    {
        "id": "JD-1",
        "company": "Kakao",
        "role": "ML Engineer",
        "skills": [
            "Python", "Machine Learning", "Deep Learning",
            "TensorFlow", "PyTorch", "SQL",
            "Data Visualization", "NLP",
            "BERT", "Feature Engineering", "Statistics"
        ]
    },
    {
        "id": "JD-2",
        "company": "Naver",
        "role": "Backend Engineer",
        "skills": [
            "Java", "Spring Boot", "MySQL",
            "PostgreSQL", "Microservices",
            "Docker", "Kubernetes",
            "REST API", "CI/CD", "Redis"
        ]
    },
    {
        "id": "JD-3",
        "company": "Line",
        "role": "Frontend Engineer",
        "skills": [
            "JavaScript", "React", "Vue",
            "TypeScript", "REST API",
            "HTML/CSS", "Node.js",
            "GraphQL", "Redux",
            "Jest", "AWS"
        ]
    }
]

# =========================
# NORMALIZATION
# =========================

def normalize_skills(skill_string):
    tokens = [s.strip().lower() for s in skill_string.split(",")]

    normalized = []

    for token in tokens:
        if token in SKILL_ALIASES:
            normalized.append(SKILL_ALIASES[token])

    return list(set(normalized))


# =========================
# TF-IDF
# =========================

def compute_idf(all_resumes):
    df = Counter()

    for resume in all_resumes:
        for skill in resume:
            df[skill] += 1

    idf = {}

    for skill in df:
        idf[skill] = math.log(10 / df[skill])

    return idf


def compute_tfidf(all_resumes, idf):
    tfidf_vectors = {}

    for resume in RESUME_DATASET:
        skills = resume["normalized_skills"]

        tfidf = {}

        tf = 1 / len(skills)

        for skill in skills:
            tfidf[skill] = tf * idf[skill]

        tfidf_vectors[resume["name"]] = tfidf

    return tfidf_vectors


# =========================
# JD VECTORS
# =========================

def normalize_jd_skills(skills):
    normalized = []

    for skill in skills:
        skill = skill.lower()

        if skill in SKILL_ALIASES:
            normalized.append(SKILL_ALIASES[skill])

    return normalized


def build_jd_vectors(jds, vocabulary):
    jd_vectors = {}

    for jd in jds:
        normalized_skills = normalize_jd_skills(jd["skills"])

        vector = []

        for vocab_skill in vocabulary:
            if vocab_skill in normalized_skills:
                vector.append(1)
            else:
                vector.append(0)

        jd_vectors[jd["id"]] = vector

    return jd_vectors


# =========================
# COSINE SIMILARITY
# =========================

def cosine_similarity(tfidf_dict, jd_vector, vocabulary):
    resume_vector = []

    for skill in vocabulary:
        resume_vector.append(tfidf_dict.get(skill, 0))

    dot_product = sum(
        resume_vector[i] * jd_vector[i]
        for i in range(len(vocabulary))
    )

    magnitude_resume = math.sqrt(
        sum(x ** 2 for x in resume_vector)
    )

    magnitude_jd = math.sqrt(
        sum(x ** 2 for x in jd_vector)
    )

    if magnitude_resume == 0 or magnitude_jd == 0:
        return 0

    return dot_product / (magnitude_resume * magnitude_jd)


# =========================
# MAIN
# =========================

def main():

    # Normalize resumes
    all_resumes = []

    for resume in RESUME_DATASET:
        normalized = normalize_skills(resume["skills"])

        resume["normalized_skills"] = normalized

        all_resumes.append(normalized)

    # Vocabulary
    vocabulary = sorted(
        set(skill for resume in all_resumes for skill in resume)
    )

    # IDF
    idf = compute_idf(all_resumes)

    # TF-IDF
    tfidf_vectors = compute_tfidf(all_resumes, idf)

    # JD vectors
    jd_vectors = build_jd_vectors(
        JOBDESCRIPTIONDATASET,
        vocabulary
    )

    # Similarities
    results = {}

    for jd in JOBDESCRIPTIONDATASET:

        jd_id = jd["id"]

        scores = []

        for candidate_name, tfidf in tfidf_vectors.items():

            similarity = cosine_similarity(
                tfidf,
                jd_vectors[jd_id],
                vocabulary
            )

            scores.append(
                (candidate_name, similarity)
            )

        scores.sort(
            key=lambda x: (-x[1], x[0])
        )

        results[jd_id] = scores[:3]

    # Output
    for jd in JOBDESCRIPTIONDATASET:

        print(f"\n{jd['id']} — {jd['company']} ({jd['role']})")

        output = []

        for name, score in results[jd["id"]]:
            output.append(f"{name}({score:.2f})")

        print(", ".join(output))


if __name__ == "__main__":
    main()
    