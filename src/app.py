from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session, send_file, abort
from pathlib import Path
import pandas as pd
from werkzeug.utils import secure_filename, safe_join
from resume_parser import extract_text, extract_years_of_experience
from ranker import rank_resumes
import json
import uuid
import mimetypes
import os
import base64

app = Flask(__name__, template_folder="../templates")
app.secret_key = "supersecret"

UPLOAD_FOLDER = Path("data/uploads").resolve()
UPLOAD_FOLDER.mkdir(exist_ok=True, parents=True)

TMP_FOLDER = Path("data/tmp")
TMP_FOLDER.mkdir(exist_ok=True, parents=True)

FEEDBACK_FILE = Path("data/feedback.csv")
FEEDBACK_FILE.parent.mkdir(exist_ok=True, parents=True)


def process_uploaded_resumes(files):
    resumes = []
    for f in files:
        filename = secure_filename(f.filename)
        save_path = UPLOAD_FOLDER / filename
        f.save(save_path)

        text = extract_text(save_path)
        years_exp = extract_years_of_experience(text)

        resumes.append({
            "filename": filename,
            "text": text,
            "years_experience": int(years_exp),
            "score": 0.0,
            "path": str(save_path)
        })
    return resumes



@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        jd_text = request.form.get("jd_text", "")
        uploaded_resumes = request.files.getlist("resumes")

        if not jd_text.strip() or not uploaded_resumes:
            return render_template("index.html", error="Please upload resumes and provide a job description.")

        resumes = process_uploaded_resumes(uploaded_resumes)

        ranked = rank_resumes(jd_text, resumes)

        file_id = str(uuid.uuid4())
        results_file = TMP_FOLDER / f"results_{file_id}.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(ranked, f)

        session["results_file"] = str(results_file)

        return render_template("results.html", jd=jd_text, ranked=ranked)

    if request.method == "GET" and "results_file" in session:
        results_file = Path(session["results_file"])
        if results_file.exists():
            with open(results_file, "r", encoding="utf-8") as f:
                ranked = json.load(f)

            min_score = float(request.args.get("min_score", 0))
            min_exp = int(request.args.get("min_exp", 0))
            keyword = request.args.get("keyword", "").lower()

            filtered = [
                r for r in ranked
                if float(r.get("score", 0)) >= min_score
                and int(r.get("years_experience", 0)) >= min_exp
                and (keyword in r.get("text", "").lower() if keyword else True)
            ]

            return render_template("results.html", jd="", ranked=filtered)

    return render_template("index.html")


@app.route("/download/<path:filename>", endpoint="download_resume")
def download_resume(filename):
    safe_name = secure_filename(filename)
    return send_from_directory(
        UPLOAD_FOLDER,
        safe_name,
        as_attachment=True,
        download_name=safe_name
    )
    

@app.get("/preview_bin")
def preview_bin():
    """
    Inline preview stream. URL me extension nahi, aur content-type octet-stream.
    Frontend blob ko 'application/pdf' set karega.
    Use: /preview_bin?id=<base64url(filename)>
    """
    token = request.args.get("id")
    if not token:
        abort(400)

    try:
        # base64url decode without padding
        pad = '=' * (-len(token) % 4)
        filename = base64.urlsafe_b64decode(token + pad).decode("utf-8")
    except Exception:
        abort(400)

    safe_name = secure_filename(filename)
    full_path = safe_join(str(UPLOAD_FOLDER), safe_name)
    if not full_path or not os.path.isfile(full_path):
        abort(404)

    # Check file extension and set the MIME type accordingly
    mime_type, _ = mimetypes.guess_type(full_path)
    if not mime_type:
        mime_type = 'application/octet-stream'

    if filename.lower().endswith(".pdf"):
        mime_type = 'application/pdf'
    
    resp = send_file(full_path, mimetype=mime_type,
                     as_attachment=False, download_name=os.path.basename(full_path))
    resp.headers["Cache-Control"] = "no-store"
    return resp


#  Path to store feedback data
FEEDBACK_FILE = Path("data/feedback.csv")

@app.route("/feedback", methods=["POST"])
def feedback():
    decisions = []
    for key, value in request.form.items():
        if key.startswith("decision_") and value:
            filename = key.replace("decision_", "")
            decisions.append([filename, value])  

    if decisions:
        df = pd.DataFrame(decisions, columns=["filename", "decision"])
        if FEEDBACK_FILE.exists():
            df.to_csv(FEEDBACK_FILE, mode="a", header=False, index=False) 
        else:
            df.to_csv(FEEDBACK_FILE, index=False) 

    return redirect(url_for("history"))



@app.route("/history")
def history():
    if FEEDBACK_FILE.exists():
        df = pd.read_csv(FEEDBACK_FILE)
        tables = [df.to_html(classes="table table-striped", index=False)]
    else:
        tables = ["<p>No feedback yet.</p>"]
    return render_template("history.html", tables=tables)

    from retrain_model import retrain_model

@app.route("/retrain", methods=["GET"])
def retrain():
    retrain_model()
    return "Model retrained successfully!"




@app.route("/reset")
def reset():
    session.pop("results_file", None)
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)