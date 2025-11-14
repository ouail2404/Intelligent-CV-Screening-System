import os
import re
from datetime import datetime
from typing import List, Set, Dict, Optional, Tuple

from sentence_transformers import SentenceTransformer
import numpy as np
import PyPDF2
import docx2txt

# ----------------
# Config
# ----------------
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
sbert_model = SentenceTransformer(MODEL_NAME)

UPLOAD_DIR = "uploads"
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

# ----------------
# Skill catalog & aliases
# ----------------
SKILL_CATALOG = {
    "frontend": [
        "HTML", "CSS", "JavaScript", "TypeScript",
        "React", "Next.js", "Redux", "Zustand", "React Query",
        "Tailwind CSS", "Material UI", "Chakra UI", "Radix UI", "Ant Design",
        "REST", "GraphQL", "tRPC",
        "Webpack", "Vite",
        "Jest", "Playwright", "Cypress", "Storybook",
        "Accessibility", "WCAG", "PWA", "Service Workers", "i18n", "l10n"
    ],
    "backend": [
        "Node.js", "Express", "NestJS", "Fastify", "Python", "Django", "Flask", "FastAPI",
        "Java", "Spring Boot", "Go", "Rust",
        "REST", "OpenAPI", "gRPC", "GraphQL", "Microservices", "Event-driven Architecture",
        "PostgreSQL", "MySQL", "MongoDB", "Redis", "Kafka", "RabbitMQ",
        "Docker", "Kubernetes", "Helm", "Terraform",
        "OAuth2", "JWT", "OWASP", "OpenTelemetry", "Prometheus", "Grafana",
        "ELK", "EFK", "CI/CD", "GitHub Actions", "GitLab CI", "JUnit", "pytest", "xUnit"
    ],
    "devops": [
        "Linux", "Docker", "Kubernetes", "Helm", "Terraform", "Ansible",
        "GitHub Actions", "GitLab CI", "ArgoCD", "Flux",
        "Prometheus", "Grafana", "OpenTelemetry", "ELK", "EFK", "Nginx", "Istio"
    ],
}
ALL_SKILLS: Set[str] = {s.lower() for arr in SKILL_CATALOG.values() for s in arr}

ALIASES = {
    "js": "javascript",
    "ts": "typescript",
    "reactjs": "react",
    "react js": "react",
    "react-js": "react",
    "nextjs": "next.js",
    "restful": "rest",
    "open api": "openapi",
    "open-api": "openapi",
    "oauth": "oauth2",
    "oauth 2": "oauth2",
    "oauth-2": "oauth2",
    "gh actions": "github actions",
    "gha": "github actions",
    "gitlab-ci": "gitlab ci",
    "k8s": "kubernetes",
    "helm chart": "helm",
    "otel": "opentelemetry",
    "elk stack": "elk",
    "efk stack": "efk",
    "ci cd": "ci/cd",
    "continuous integration": "ci/cd",
    "continuous delivery": "ci/cd",
    "containerization": "docker",
    "containers": "docker",
    "postgres": "postgresql",
    "ms sql": "sql server",
    "mssql": "sql server",
    "rtk": "redux",
    "redux toolkit": "redux",
    "react-query": "react query",
    "mui": "material ui",
    "mat ui": "material ui",
    "ant": "ant design",
    "cypress e2e": "cypress",
    "cypress.io": "cypress",
    "e2e": "e2e",
    "i18n/l10n": "i18n",
    "i18n-l10n": "i18n",
    "service worker": "service workers",
}

EXPERIENCE_RE = re.compile(r"(\d+)\s*(\+)?\s*(years?|yrs?)", re.IGNORECASE)
RANGE_RE = re.compile(
    r"(?P<start>(19|20)\d{2})\s*[-–to]+\s*(?P<end>(19|20)\d{2}|present|current|now)",
    re.IGNORECASE,
)

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
    re.IGNORECASE,
)
BULLET_LINE = re.compile(r"^\s*(?:[-*•]+|\d+\.)\s+(.*)$")

MUST_WEIGHT = 0.70


# ----------------
# File helpers
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


# ----------------
# Text helpers
# ----------------
def normalize_text(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip()


def tokenize(text: str) -> Set[str]:
    if not text:
        return set()
    return set(re.findall(r"[a-z0-9+.#/_;|]+", text.lower()))


def apply_aliases_to_tokens(tokens: Set[str]) -> Set[str]:
    mapped = set(tokens)
    extra = set()
    for t in list(mapped):
        if "/" in t or ";" in t or "|":
            for part in re.split(r"[\/;\|]", t):
                part = part.strip()
                if part:
                    extra.add(part)
    mapped |= extra

    s = " " + " ".join(mapped) + " "
    for k, v in ALIASES.items():
        if k in mapped or (" " + k + " ") in s:
            mapped.add(v)
    return mapped


def detect_skills(text: str) -> Set[str]:
    if not text:
        return set()
    tlow = text.lower()
    toks = apply_aliases_to_tokens(tokenize(text))
    hits = set()
    for s in ALL_SKILLS:
        if " " in s and s in tlow:
            hits.add(s)
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

    now_year = datetime.now().year
    total = 0.0
    for m in RANGE_RE.finditer(text):
        start = int(m.group("start"))
        end_raw = m.group("end").lower()
        end = now_year if end_raw in {"present", "current", "now"} else int(end_raw)
        if end >= start:
            total += min(end - start, 10)
            if total >= 15:
                break
    return int(round(total))


def embed(texts: List[str]) -> np.ndarray:
    return sbert_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


# ----------------
# JD parsing
# ----------------
def _section_kind(header_text: str) -> str:
    ht = header_text.lower()
    for p in MUST_HEADERS:
        if re.search(p, ht):
            return "must"
    for p in NICE_HEADERS:
        if re.search(p, ht):
            return "nice"
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
        if mode not in {"must", "nice"}:
            continue
        b = BULLET_LINE.match(line)
        captured = b.group(1) if b else line
        chunks[mode].append(captured)

    return {
        "must": "\n".join(chunks["must"]).strip(),
        "nice": "\n".join(chunks["nice"]).strip(),
    }


def jd_profile(jd: str) -> Tuple[Set[str], Set[str], int]:
    sec = parse_jd_sections_full(jd)
    must_raw, nice_raw = sec["must"], sec["nice"]

    must_sk = detect_skills(must_raw) if must_raw else set()
    nice_sk = detect_skills(nice_raw) if nice_raw else set()

    if not must_sk and not nice_sk:
        detected = detect_skills(jd)
        must_sk, nice_sk = detected, set()

    nice_sk -= (must_sk & nice_sk)

    min_years = 0
    for y, _, _ in EXPERIENCE_RE.findall(jd):
        min_years = max(min_years, int(y))

    return {s.lower() for s in must_sk}, {s.lower() for s in nice_sk}, min_years


def coverage_counts(
    cv_sk: Set[str],
    must: Set[str],
    nice: Set[str],
) -> Tuple[float, int, int, int, int, Optional[float]]:
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
    if p >= 0.92:
        return "Perfect fit"
    if p >= 0.85:
        return "Excellent fit"
    if p >= 0.75:
        return "Great fit"
    if p >= 0.65:
        return "Strong fit"
    if p >= 0.50:
        return "Fair fit"
    return "Developing"


# ----------------
# Core scoring (used by APIs)
# ----------------
def compute_matches_from_text(jd_text: str, cvs: List[Dict[str, str]]) -> List[Dict]:
    jd_text = jd_text or ""
    jd_text_norm = normalize_text(jd_text)

    must_sk, nice_sk, min_years = jd_profile(jd_text_norm)
    req_all = must_sk | nice_sk

    jd_vec = embed([jd_text_norm])[0]

    WEIGHTS = {"sem": 0.25, "jac": 0.10, "exp": 0.25, "cov": 0.40}
    rows = []

    for idx, cv in enumerate(cvs, start=1):
        name = cv.get("name") or f"Candidate {idx}"
        raw = normalize_text(cv.get("text", ""))

        txt_for_embed = raw[:4000] if len(raw) > 4000 else raw
        cv_vec = embed([txt_for_embed])[0] if raw else embed([""])[0]

        cv_skills_sorted = sorted(detect_skills(raw))
        cv_sk_set = {s.lower() for s in cv_skills_sorted}
        cv_years = extract_years(raw)

        sem = cosine_sim(cv_vec, jd_vec)

        weighted_cov, mh, mt, nh, nt, nice_cov = coverage_counts(
            cv_sk_set, must_sk, nice_sk
        )

        matched = sorted(list(cv_sk_set & req_all))
        missing_must = sorted(list(must_sk - cv_sk_set))
        missing_nice = sorted(list(nice_sk - cv_sk_set))
        extra = sorted(list(cv_sk_set - req_all))

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
            "score": round(final * 100, 1),
            "fit": fit_label(final),
            "years": cv_years,
            "cv_skills": cv_skills_sorted,
            "matched_skills": matched,
            "missing_skills": sorted(list(req_all - cv_sk_set)),
            "missing_must": missing_must,
            "missing_nice": missing_nice,
            "extra_skills": extra,
            "suggestions": suggestions,
            "min_years": min_years,
            "components": {
                "semantic": round(sem, 3),
                "skill_overlap": round(jaccard, 3),
                "experience_score": round(exp, 3),
                "coverage": round(weighted_cov, 3),
                "must_coverage": round((mh / mt), 3) if mt else 0.0,
                "nice_coverage": (round(nice_cov, 3) if nice_cov is not None else None),
                "must_counts": f"{mh}/{mt}" if mt else "0/0",
                "nice_counts": (f"{nh}/{nt}" if nt else "N/A"),
            },
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
        reverse=True,
    )

    return [rows[i] for i in order]
