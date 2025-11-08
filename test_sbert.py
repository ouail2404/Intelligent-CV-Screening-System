
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

e1 = model.encode("We need someone skilled in React and REST APIs.", convert_to_tensor=True)
e2 = model.encode("Looking for a frontend developer with SPA experience and API integration.", convert_to_tensor=True)

print(float(util.cos_sim(e1, e2)))
# Expect a positive similarity (e.g., 0.6–0.9)
