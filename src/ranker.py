from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def rank_resumes(job_description, resumes):
    """
    Rank resumes against a job description using TF-IDF + cosine similarity.
    Expects each resume dict to have 'filename', 'text', 'years_experience'.
    """

    # Prepare documents: JD + resumes
    docs = [job_description] + [r["text"] for r in resumes]

    # TF-IDF encoding
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(docs)

    # Compute cosine similarity of each resume with JD
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # Attach scores back to resumes
    for i, score in enumerate(cosine_similarities):
        resumes[i]["score"] = float(score)

    # Sort by score descending
    ranked = sorted(resumes, key=lambda x: x["score"], reverse=True)
    return ranked