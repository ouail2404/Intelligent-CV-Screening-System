# main.py — JD-first CV matcher with MUST/NICE coverage, old UI, strict reveal
# Run: python main.py  -> http://127.0.0.1:5000/

import os
import re
from datetime import datetime
from typing import List, Set, Dict, Optional, Tuple

from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer
import numpy as np
import PyPDF2
import docx2txt

# Optional spaCy (nice tokenization but not required)
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None

# ----------------
# Config
# ----------------
UPLOAD_DIR = "uploads"
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
sbert_model = SentenceTransformer(MODEL_NAME)

# ----------------
# Catalog (expanded so NICE TO HAVE lines are fully picked up)
# ----------------
SKILL_CATALOG = {
  "frontend": [
    "HTML","CSS","JavaScript","TypeScript",
    "React","Next.js","Redux","Zustand","React Query",
    "Tailwind CSS","Material UI","Chakra UI","Radix UI","Ant Design",
    "REST","GraphQL","tRPC",
    "Webpack","Vite",
    "Jest","Playwright","Cypress","Storybook",
    "Accessibility","WCAG","PWA","Service Workers","i18n","l10n"
  ],
  "backend": [
    "Node.js","Express","NestJS","Fastify","Python","Django","Flask","FastAPI",
    "Java","Spring Boot","Go","Rust",
    "REST","OpenAPI","gRPC","GraphQL","Microservices","Event-driven Architecture",
    "PostgreSQL","MySQL","MongoDB","Redis","Kafka","RabbitMQ",
    "Docker","Kubernetes","Helm","Terraform",
    "OAuth2","JWT","OWASP","OpenTelemetry","Prometheus","Grafana",
    "ELK","EFK","CI/CD","GitHub Actions","GitLab CI","JUnit","pytest","xUnit"
  ],
  "devops": [
    "Linux","Docker","Kubernetes","Helm","Terraform","Ansible",
    "GitHub Actions","GitLab CI","ArgoCD","Flux",
    "Prometheus","Grafana","OpenTelemetry","ELK","EFK","Nginx","Istio"
  ],
}
ALL_SKILLS: Set[str] = {s.lower() for arr in SKILL_CATALOG.values() for s in arr}

# Aliases
ALIASES = {
  "js":"javascript","ts":"typescript",
  "reactjs":"react","react js":"react","react-js":"react",
  "nextjs":"next.js","restful":"rest","open api":"openapi","open-api":"openapi",

  "oauth":"oauth2","oauth 2":"oauth2","oauth-2":"oauth2",
  "gh actions":"github actions","gha":"github actions","gitlab-ci":"gitlab ci",
  "k8s":"kubernetes","helm chart":"helm","otel":"opentelemetry",
  "elk stack":"elk","efk stack":"efk",
  "ci cd":"ci/cd","continuous integration":"ci/cd","continuous delivery":"ci/cd",
  "containerization":"docker","containers":"docker",

  "postgres":"postgresql","ms sql":"sql server","mssql":"sql server",

  "rtk":"redux","redux toolkit":"redux",
  "react-query":"react query",
  "mui":"material ui","mat ui":"material ui","ant":"ant design",

  "cypress e2e":"cypress","cypress.io":"cypress","e2e":"e2e",
  "i18n/l10n":"i18n","i18n-l10n":"i18n",
  "service worker":"service workers"
}

# ----------------
# Regex helpers
# ----------------
EXPERIENCE_RE = re.compile(r"(\d+)\s*(\+)?\s*(years?|yrs?)", re.IGNORECASE)
RANGE_RE = re.compile(r"(?P<start>(19|20)\d{2})\s*[-–to]+\s*(?P<end>(19|20)\d{2}|present|current|now)", re.IGNORECASE)

MUST_HEADERS = [
    r"must\s*have", r"requirements?", r"required", r"minimum\s+qualifications?",
    r"what\s+you'?ll\s+need", r"you\s+will\s+need"
]
NICE_HEADERS = [
    r"nice\s*to\s*have", r"preferred", r"preferred\s+qualifications?", r"bonus",
    r"bonus\s+points", r"plus", r"good\s*to\s*have", r"optional", r"nice\s*have"
]
HEADER_RE = re.compile(
    r"^\s*(?P<h>(" + "|".join(MUST_HEADERS + NICE_HEADERS) + r"))\s*:?\s*$",
    re.IGNORECASE
)
BULLET_LINE = re.compile(r"^\s*(?:[-*•]+|\d+\.)\s+(.*)$")

# ----------------
# IO helpers
# ----------------
def allowed_file(filename: str) -> bool:
    low = filename.lower()
    return any(low.endswith(ext) for ext in ALLOWED_EXTENSIONS)

def extract_text(path: str) -> str:
    low = path.lower()
    if low.endswith(".pdf"):
        parts: List[str] = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for p in reader.pages:
                parts.append(p.extract_text() or "")
        return "\n".join(parts)
    if low.endswith(".docx"):
        return docx2txt.process(path) or ""
    if low.endswith(".txt"):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    return ""

def normalize_text(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip()

# ----------------
# NLP helpers
# ----------------
def tokenize(text: str) -> Set[str]:
    if not text:
        return set()
    if nlp:
        return {t.text.lower() for t in nlp(text)}
    return set(re.findall(r"[a-zA-Z0-9+.#/_;| -]+", text.lower()))

def apply_aliases_to_tokens(tokens: Set[str]) -> Set[str]:
    # Expand slash/semicolon/pipe combos: "i18n/l10n", "Material UI/Chakra UI"
    mapped = set(tokens)
    extra = set()
    for t in list(mapped):
        if "/" in t or ";" in t or "|" in t:
            for part in re.split(r"[\/;\|]", t):
                part = part.strip()
                if part:
                    extra.add(part)
    mapped |= extra

    # Apply alias substitutions
    s = " " + " ".join(mapped) + " "
    for k, v in ALIASES.items():
        if k in mapped or (" " + k + " ") in s:
            mapped.add(v)
    return mapped

def detect_skills(text: str) -> Set[str]:
    tlow = text.lower()
    toks = apply_aliases_to_tokens(tokenize(text))
    hits = set()
    # Multi-word
    for s in ALL_SKILLS:
        if " " in s and s in tlow:
            hits.add(s)
    # Single-token
    for s in ALL_SKILLS:
        if " " not in s and s in toks:
            hits.add(s)
    return hits

def extract_years(text: str) -> int:
    years = 0
    for m in EXPERIENCE_RE.finditer(text):
        years = max(years, int(m.group(1)))
    if years > 0:
        return years
    # derive from date ranges
    now_year = datetime.now().year
    total = 0.0
    for m in RANGE_RE.finditer(text):
        start = int(m.group("start"))
        end_raw = m.group("end").lower()
        end = now_year if end_raw in {"present","current","now"} else int(end_raw)
        if end >= start:
            total += min(end - start, 10)
            if total >= 15:
                break
    return int(round(total))

def extract_education(text: str) -> Optional[str]:
    pats = [
        r"(ph\.?d\.?)", r"(doctorate)", r"(m\.?sc\.?|master(?:'s)?)",
        r"(b\.?sc\.?|bachelor(?:'s)?)", r"(mba)", r"(associate(?:'s)?)",
        r"(engineer(?:ing)?\s+diploma)"
    ]
    order = ["phd","doctorate","msc","master","mba","bsc","bachelor","associate","engineering diploma"]
    tlow = text.lower()
    found = []
    for pat in pats:
        for m in re.finditer(pat, tlow):
            found.append(m.group(1))
    if not found:
        return None
    def rank(d: str) -> int:
        d = d.replace(".", "")
        for i, key in enumerate(order):
            if key in d:
                return i
        return len(order) + 1
    best = sorted({d for d in found}, key=rank)[0]
    for line in text.splitlines():
        if best.split()[0] in line.lower():
            return normalize_text(line)
    return best

def embed(texts: List[str]) -> np.ndarray:
    return sbert_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

# ----------------
# JD parsing (MUST/NICE)
# ----------------
def _section_kind(header_text: str) -> str:
    ht = header_text.lower()
    for p in MUST_HEADERS:
        if re.search(p, ht): return "must"
    for p in NICE_HEADERS:
        if re.search(p, ht): return "nice"
    return "other"

def parse_jd_sections_full(jd: str) -> Dict[str, str]:
    lines = [ln.rstrip() for ln in jd.splitlines()]
    chunks: Dict[str, List[str]] = {"must": [], "nice": []}
    mode: Optional[str] = None

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        m = HEADER_RE.match(line)
        if m:
            mode = _section_kind(m.group("h"))
            continue
        if mode not in {"must","nice"}:
            # Also capture inline comma/semicolon lists under the same stanza
            continue
        b = BULLET_LINE.match(line)
        captured = b.group(1) if b else line
        chunks[mode].append(captured)

    return {
        "must": "\n".join(chunks["must"]).strip(),
        "nice": "\n".join(chunks["nice"]).strip()
    }

def jd_profile(jd: str) -> Tuple[Set[str], Set[str], int]:
    sec = parse_jd_sections_full(jd)
    must_raw, nice_raw = sec["must"], sec["nice"]

    must_sk = detect_skills(must_raw) if must_raw else set()
    nice_sk = detect_skills(nice_raw) if nice_raw else set()

    if not must_sk and not nice_sk:
        # Fallback: detect over whole JD if headers missing
        detected = detect_skills(jd)
        must_sk, nice_sk = detected, set()

    # ensure unique
    nice_sk -= (must_sk & nice_sk)

    # Years
    min_years = 0
    for y, plus, _ in EXPERIENCE_RE.findall(jd):
        min_years = max(min_years, int(y))

    return {s.lower() for s in must_sk}, {s.lower() for s in nice_sk}, min_years

# coverage with MUST weight
MUST_WEIGHT = 0.70

def coverage_counts(cv_sk: Set[str], must: Set[str], nice: Set[str]) -> Tuple[float, int, int, int, int, Optional[float]]:
    must_total = len(must)
    nice_total = len(nice)

    must_hit = len(cv_sk & must)
    nice_hit = len(cv_sk & nice) if nice_total else 0

    must_cov = (must_hit / must_total) if must_total else 0.0
    if nice_total:
        nice_cov = nice_hit / float(nice_total)
        weighted = MUST_WEIGHT * must_cov + (1 - MUST_WEIGHT) * nice_cov
        return weighted, must_hit, must_total, nice_hit, nice_total, nice_cov
    else:
        return must_cov, must_hit, must_total, 0, 0, None

def fit_label(p: float) -> str:
    if p >= 0.92: return "Perfect fit"
    if p >= 0.85: return "Excellent fit"
    if p >= 0.75: return "Great fit"
    if p >= 0.65: return "Strong fit"
    if p >= 0.50: return "Fair fit"
    return "Developing"

# ----------------
# Flask
# ----------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

@app.route("/", methods=["GET"])
def home():
    return render_template("app.html")

@app.route("/upload", methods=["POST"])
def upload():
    jd_text = (request.form.get("resumeText") or "").strip()
    files = request.files.getlist("resumeFile")

    if not jd_text and not files:
        return render_template("app.html", message="Please upload resumes and/or enter a JD.")
    if not jd_text:
        return render_template("app.html", message="Please enter a job description.")

    valid_files = [f for f in files if f and f.filename and allowed_file(f.filename)]
    if not valid_files:
        return render_template("app.html", message="Please upload at least one .pdf, .docx, or .txt file.")

    saved_names, cv_texts = [], []
    for f in valid_files:
        path = os.path.join(app.config["UPLOAD_FOLDER"], f.filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        f.save(path)
        saved_names.append(f.filename)
        cv_texts.append(extract_text(path) or "")

    # JD-first profile
    must_sk, nice_sk, min_years = jd_profile(jd_text)
    req_all = (must_sk | nice_sk)
    jd_vec = embed([jd_text])[0]

    # weights: emphasize coverage (must-weighted), then semantic, overlap, experience
    WEIGHTS = {"sem": 0.30, "jac": 0.15, "exp": 0.10, "cov": 0.45}
    rows = []

    for idx, (name, text) in enumerate(zip(saved_names, cv_texts), start=1):
        raw = normalize_text(text)
        txt_for_embed = raw[:4000] if len(raw) > 4000 else raw
        cv_vec = embed([txt_for_embed])[0] if raw else embed([""])[0]

        cv_skills_sorted = sorted(detect_skills(raw))
        cv_sk_set = {s.lower() for s in cv_skills_sorted}
        cv_years  = extract_years(raw)
        cv_edu    = extract_education(raw) or ""

        sem = cosine_sim(cv_vec, jd_vec)

        weighted_cov, mh, mt, nh, nt, nice_cov = coverage_counts(cv_sk_set, must_sk, nice_sk)

        matched = sorted(list(cv_sk_set & req_all))
        missing_must = sorted(list(must_sk - cv_sk_set))
        missing_nice = sorted(list(nice_sk - cv_sk_set))
        extra = sorted(list(cv_sk_set - req_all))

        # suggestions: MUST first then NICE
        suggestions = (missing_must + missing_nice)[:10]

        req_count = len(req_all)
        matched_count = len(matched)
        jaccard = (matched_count / req_count) if req_count else 0.0
        exp = min(cv_years / float(min_years), 1.0) if min_years > 0 else 0.6

        final = (
            WEIGHTS["sem"] * sem +
            WEIGHTS["jac"] * jaccard +
            WEIGHTS["exp"] * exp +
            WEIGHTS["cov"] * weighted_cov
        )
        final = max(0.0, min(1.0, final))

        rows.append({
            "id": f"cv-{idx}",
            "name": name,
            "score": round(final, 3),
            "fit": fit_label(final),
            "years": cv_years,
            "education": cv_edu,
            "cv_skills": cv_skills_sorted,
            "matched_skills": matched,
            "missing_skills": sorted(list(req_all - cv_sk_set)),
            "missing_must": missing_must,
            "missing_nice": missing_nice,
            "extra_skills": extra,
            "suggestions": suggestions,
            "match_count": matched_count,
            "req_count": req_count,
            "min_years": min_years,
            "components": {
                "semantic": round(sem, 3),
                "skill_overlap": round(jaccard, 3),
                "experience_score": round(exp, 3),
                "coverage": round(weighted_cov, 3),
                "must_coverage": round((mh / mt), 3) if mt else 0.0,
                "nice_coverage": (round(nice_cov, 3) if nice_cov is not None else None),
                "must_counts": f"{mh}/{mt}" if mt else "0/0",
                "nice_counts": (f"{nh}/{nt}" if nt else "N/A")
            }
        })

    order = sorted(
        range(len(rows)),
        key=lambda i: (
            rows[i]["score"],
            rows[i]["components"]["must_coverage"],
            rows[i]["components"]["coverage"],
            rows[i]["components"]["skill_overlap"],
            rows[i]["components"]["semantic"],
            rows[i]["years"],
        ),
        reverse=True
    )

    top_rows = [rows[i] for i in order]
    all_results = [{"filename": rows[i]["name"], "score": rows[i]["score"]} for i in order]

    msg = "JD mode (strict): MUST/NICE parsed; click a row for details."
    return render_template("app.html", message=msg, top_rows=top_rows, all_results=all_results)

if __name__ == "__main__":
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    app.run(debug=True)
