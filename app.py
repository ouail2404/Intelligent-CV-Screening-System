import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from matcher import (
    compute_matches_from_text,
    allowed_file,
    extract_text,
    UPLOAD_DIR,
)

app = Flask(__name__)
CORS(app)

# Ensure upload dir exists
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "Flask API running"}), 200


@app.route("/api/match", methods=["POST"])
def match_from_text():
    """
    JSON mode: JD + CVs as text.
    {
      "job_description": "full JD text...",
      "cvs": [
        { "name": "Candidate 1", "text": "..." },
        { "name": "Candidate 2", "text": "..." }
      ]
    }
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON body provided"}), 400

    jd_text = (data.get("job_description") or "").strip()
    cvs = data.get("cvs") or []

    if not jd_text:
        return jsonify({"error": "job_description is required"}), 400
    if not cvs:
        return jsonify({"error": "At least one CV is required"}), 400

    try:
        results = compute_matches_from_text(jd_text, cvs)
        return jsonify({"results": results}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/match_files", methods=["POST"])
def match_from_files():
    """
    Multipart mode: JD as text + uploaded CV files.

    Form fields:
      - job_description: text
      - files: multiple PDF/DOCX/TXT
    """
    jd_text = (request.form.get("job_description") or "").strip()
    if not jd_text:
        return jsonify({"error": "job_description is required"}), 400

    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    cvs = []
    for f in files:
        if not f or not f.filename:
            continue
        if not allowed_file(f.filename):
            continue
        filename = secure_filename(f.filename)
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        f.save(path)
        text = extract_text(path) or ""
        if text.strip():
            cvs.append({"name": filename, "text": text})

    if not cvs:
        return jsonify({"error": "No valid CV content extracted"}), 400

    try:
        results = compute_matches_from_text(jd_text, cvs)
        return jsonify({"results": results}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
