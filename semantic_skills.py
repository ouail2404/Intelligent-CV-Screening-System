# semantic_skills.py
import re
from sentence_transformers import SentenceTransformer, util

SENT_SPLIT = re.compile(r'(?<=[.!?])\s+')

class SemanticSkillAssessor:
    def __init__(self, model: SentenceTransformer):
        self.model = model

    def sentences(self, text: str):
        return [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]

    def skill_coverage(self, cv_text: str, skills: list[str], threshold: float = 0.5):
        """
        Pure semantic check: no keyword rules.
        For each skill concept, find the most similar sentence in the CV.
        If similarity >= threshold → mark as present.
        """
        sents = self.sentences(cv_text)
        if not sents:
            return {sk: {"present": False, "score": 0.0, "evidence": ""} for sk in skills}

        sent_emb = self.model.encode(sents, convert_to_tensor=True, normalize_embeddings=True)
        skill_emb = self.model.encode(skills, convert_to_tensor=True, normalize_embeddings=True)

        sims = util.cos_sim(skill_emb, sent_emb).cpu().numpy()  # (len(skills), len(sents))

        out = {}
        for i, sk in enumerate(skills):
            j = sims[i].argmax()
            best = float(sims[i][j])
            out[sk] = {
                "present": best >= threshold,
                "score": round(best, 3),
                "evidence": sents[j]
            }
        return out
