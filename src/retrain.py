import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path

# Function to retrain the model using feedback
def retrain_model(processed_dir="data/processed", feedback_file="data/feedback.csv", model_path="models/feedback_model.pkl"):
    # Load feedback
    feedback = pd.read_csv(feedback_file)
    if feedback.empty:
        print("No feedback data available for retraining.")
        return

    # Load resumes and feedback
    resumes = load_resumes(processed_dir)
    data = resumes.merge(feedback, on="filename", how="inner")  # Merge feedback with resumes

    # Map feedback labels (Selected -> 1, Rejected -> 0)
    data["label"] = data["decision"].map({"selected": 1, "rejected": 0})

    # Build retraining pipeline with TF-IDF and Logistic Regression
    model = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=1000)),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    # Train the model
    model.fit(data["clean_text"], data["label"])

    # Save the retrained model
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model retrained and saved to {model_path}")

# Call the retrain function periodically or after enough feedback is gathered
