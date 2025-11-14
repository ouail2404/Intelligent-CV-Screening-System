import React, { useState } from "react";
import auilogo from "./assets/aui_logo.png"

const API_BASE = "http://127.0.0.1:5000";

function App() {
  const [jd, setJd] = useState("");
  const [files, setFiles] = useState([]);
  const [results, setResults] = useState([]);
  const [openId, setOpenId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [hasAttempted, setHasAttempted] = useState(false);

  const handleFileChange = (e) => {
    const selected = Array.from(e.target.files || []);
    setFiles(selected);
    setError("");
  };

  const handleMatch = async () => {
    setHasAttempted(true);
    setError("");
    setResults([]);
    setOpenId(null);

    if (!jd.trim()) {
      setError("Please paste the Job Description.");
      return;
    }

    if (!files.length) {
      setError("Please upload at least one CV file (PDF, DOCX, or TXT).");
      return;
    }

    try {
      setLoading(true);

      const formData = new FormData();
      formData.append("job_description", jd);
      files.forEach((file) => {
        formData.append("files", file);
      });

      const res = await fetch(`${API_BASE}/api/match_files`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (!res.ok) {
        setError(data.error || "An error occurred while matching candidates.");
        return;
      }

      setResults(data.results || []);
      if ((data.results || []).length === 0) {
        setError("No candidates produced valid content.");
      }
    } catch (err) {
      setError("Could not reach the backend. Is Flask running on port 5000?");
    } finally {
      setLoading(false);
    }
  };

  const toggleRow = (id) => {
    setOpenId((prev) => (prev === id ? null : id));
  };

  const primaryGreenBg = "bg-emerald-700";

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      {/* Top bar */}
      <header className="w-full border-b border-slate-200 bg-white/95 backdrop-blur">
        <div className="max-w-6xl mx-auto flex items-center justify-between px-4 py-3 gap-4">
          {/* Left: Title only */}
          <div className="flex items-center gap-3">
            <div>
              <h1 className="text-2xl font-semibold tracking-tight text-emerald-800">
                Intelligent CV Screening System
              </h1>
            </div>
          </div>

          {/* Right: AUI logo */}
          <div className="flex items-center">
            <img
              src={auilogo}
              alt="Al Akhawayn University Logo"
              className="h-14 w-32"
              onError={(e) => {
                e.currentTarget.style.display = "none";
              }}
            />
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-6xl mx-auto px-4 py-6 space-y-6">
        {/* Input Card */}
        <section className="bg-white border border-slate-200 rounded-2xl shadow-sm">
          <div className="px-5 py-4 border-b border-slate-100">
            <h2 className="text-lg font-semibold text-slate-900">
              Job Description & Candidate CVs
            </h2>
            <p className="text-xs text-slate-500 mt-1">
              Paste the official JD on the left, upload one or more CV files on the right,
              then run the semantic matcher. Designed for HR and faculty reviewers.
            </p>
          </div>

          <div className="px-5 py-4 grid gap-6 md:grid-cols-2">
            {/* JD column */}
            <div className="space-y-2">
              <label
                htmlFor="jd"
                className="block text-sm font-medium text-slate-800"
              >
                Job Description
              </label>
              <p className="text-[10px] text-slate-500">
                The system automatically detects MUST / NICE sections if you use headings
                such as <span className="font-semibold">"Must Have"</span> or{" "}
                <span className="font-semibold">"Preferred"</span>.
              </p>
              <textarea
                id="jd"
                value={jd}
                onChange={(e) => setJd(e.target.value)}
                className="w-full h-60 text-sm rounded-xl border border-slate-200 bg-slate-50 focus:outline-none focus:ring-2 focus:ring-emerald-500/80 focus:border-emerald-500/80 px-3 py-2 resize-vertical"
                placeholder={`Follow this format please

Role: Frontend Developer

Must Have:
- React, JavaScript, HTML, CSS
- 2+ years experience

Nice to Have:
- TypeScript, Next.js, Tailwind CSS
- Testing (Jest / Cypress)
`}
              />
            </div>

            {/* CV upload column */}
            <div className="space-y-2">
              <span className="block text-sm font-medium text-slate-800">
                Candidate CVs
              </span>
              <p className="text-[10px] text-slate-500">
                Upload one or more CVs in <strong>PDF, DOCX, or TXT</strong> format.
                The system parses each CV, extracts skills & experience, and ranks candidates.
              </p>

              {/* Dropzone / input */}
              <label
                htmlFor="cvFiles"
                className="flex flex-col items-center justify-center gap-1 border-2 border-dashed border-emerald-200 rounded-xl bg-emerald-50/60 px-4 py-6 cursor-pointer hover:border-emerald-400 hover:bg-emerald-50 transition"
              >
                <span className="text-xs font-semibold text-emerald-800">
                  Click to choose CV files
                </span>
                <span className="text-[10px] text-emerald-600">
                  or drag & drop them into this area
                </span>
                <span className="text-[9px] text-emerald-500">
                  Accepted: .pdf, .docx, .txt
                </span>
                <input
                  id="cvFiles"
                  type="file"
                  multiple
                  accept=".pdf,.docx,.txt"
                  onChange={handleFileChange}
                  className="hidden"
                />
              </label>

              {/* File list */}
              {files.length > 0 && (
                <div className="mt-1 flex flex-wrap gap-1">
                  {files.map((f, idx) => (
                    <span
                      key={idx}
                      className="px-2 py-0.5 rounded-full bg-emerald-50 border border-emerald-200 text-[9px] text-emerald-800"
                    >
                      {f.name}
                    </span>
                  ))}
                </div>
              )}

              {/* Run button + errors */}
              <div className="pt-2">
                <button
                  onClick={handleMatch}
                  disabled={loading}
                  className={`w-full md:w-auto inline-flex items-center justify-center px-5 py-2 text-sm font-semibold rounded-full ${primaryGreenBg} text-white shadow-sm hover:bg-emerald-600 transition disabled:opacity-60 disabled:cursor-not-allowed`}
                >
                  {loading ? "Running semantic matching..." : "Run Matching"}
                </button>
                {error && (
                  <p className="mt-1 text-[10px] text-red-500">{error}</p>
                )}
                {!error && !loading && !hasAttempted && (
                  <p className="mt-1 text-[9px] text-slate-400">
                    Tip: start with 3–5 CVs to showcase the system’s ranking and explanations.
                  </p>
                )}
              </div>
            </div>
          </div>
        </section>

        {/* Results: hidden until first Run Matching click */}
        {hasAttempted && (
          <section className="space-y-2">
            <div className="flex items-baseline gap-3">
              <h2 className="text-lg font-semibold text-slate-900">
                Results
              </h2>
              <p className="text-[10px] text-slate-500">
                Candidates are ranked using semantic similarity (Sentence-BERT),
                MUST/NICE coverage, skill overlap, and experience score.
                Click a row to inspect the reasoning.
              </p>
            </div>

            {loading && (
              <p className="text-[10px] text-emerald-700">
                Analyzing CVs and computing semantic matches...
              </p>
            )}

            {!loading && results.length === 0 && !error && (
              <p className="text-[10px] text-slate-400">
                No valid candidates were produced for this run.
              </p>
            )}

            <div className="bg-white border border-slate-200 rounded-2xl shadow-sm divide-y divide-slate-100">
              {results.map((r, index) => {
                const isOpen = openId === r.id;
                return (
                  <div key={r.id || index} className="px-4 py-3">
                    {/* Row header */}
                    <button
                      type="button"
                      onClick={() => toggleRow(r.id)}
                      className="w-full flex items-center justify-between gap-3"
                    >
                      <div className="flex items-center gap-3 text-left">
                        <div className="w-6 h-6 flex items-center justify-center rounded-full bg-emerald-50 text-[10px] text-emerald-800 font-semibold">
                          {index + 1}
                        </div>
                        <div>
                          <div className="text-sm font-semibold text-slate-900">
                            {r.name}
                          </div>
                          <div className="text-[10px] text-slate-500">
                            {r.fit} — {r.years || 0} years experience
                            {typeof r.min_years === "number" &&
                              ` (JD min: ${r.min_years || 0} yrs)`}
                          </div>
                        </div>
                      </div>

                      {/* Score pill */}
                      <div className="flex items-center gap-2">
                        <div className="flex flex-col items-end">
                          <span className="text-xs font-semibold text-emerald-700">
                            {r.score}%
                          </span>
                          <span className="text-[9px] text-slate-400">
                            overall match
                          </span>
                        </div>
                        <div
                          className={`w-7 h-7 rounded-full border-2 flex items-center justify-center ${
                            isOpen
                              ? "border-emerald-600 text-emerald-700"
                              : "border-slate-300 text-slate-400"
                          } text-[9px]`}
                        >
                          {isOpen ? "−" : "+"}
                        </div>
                      </div>
                    </button>

                    {/* Expanded details */}
                    {isOpen && (
                      <div className="mt-3 pl-9 space-y-2 text-[10px]">
                        {/* Suggestions */}
                        {r.suggestions && r.suggestions.length > 0 && (
                          <div>
                            <div className="font-semibold text-slate-800">
                              Suggestions to improve fit
                            </div>
                            <div className="mt-1 flex flex-wrap gap-1">
                              {r.suggestions.map((s, i) => (
                                <span
                                  key={i}
                                  className="px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-800 border border-emerald-100"
                                >
                                  {s}
                                </span>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Matched skills */}
                        <div>
                          <div className="font-semibold text-slate-800">
                            Matched skills
                          </div>
                          <div className="mt-1 flex flex-wrap gap-1">
                            {r.matched_skills && r.matched_skills.length ? (
                              r.matched_skills.slice(0, 40).map((s, i) => (
                                <span
                                  key={i}
                                  className="px-2 py-0.5 rounded-full bg-emerald-50 text-emerald-800 border border-emerald-100"
                                >
                                  {s}
                                </span>
                              ))
                            ) : (
                              <span className="text-slate-400">
                                No matched skills detected.
                              </span>
                            )}
                          </div>
                        </div>

                        {/* Missing MUST */}
                        <div className="grid gap-2 md:grid-cols-2">
                          <div>
                            <div className="font-semibold text-red-600">
                              Missing MUST-have skills
                            </div>
                            <div className="mt-1 flex flex-wrap gap-1">
                              {r.missing_must && r.missing_must.length ? (
                                r.missing_must.map((s, i) => (
                                  <span
                                    key={i}
                                    className="px-2 py-0.5 rounded-full bg-red-50 text-red-700 border border-red-100"
                                  >
                                    {s}
                                  </span>
                                ))
                              ) : (
                                <span className="text-slate-400">
                                  None (all MUST covered).
                                </span>
                              )}
                            </div>
                          </div>
                        </div>

                        {/* Extra skills */}
                        <div>
                          <div className="font-semibold text-slate-800">
                            Extra skills (beyond JD)
                          </div>
                          <div className="mt-1 flex flex-wrap gap-1">
                            {r.extra_skills && r.extra_skills.length ? (
                              r.extra_skills.slice(0, 40).map((s, i) => (
                                <span
                                  key={i}
                                  className="px-2 py-0.5 rounded-full bg-slate-50 text-slate-700 border border-slate-200"
                                >
                                  {s}
                                </span>
                              ))
                            ) : (
                              <span className="text-slate-400">None.</span>
                            )}
                          </div>
                        </div>

                        {/* Score breakdown */}
                        {r.components && (
                          <div className="pt-1">
                            <div className="font-semibold text-slate-800">
                              Score breakdown
                            </div>
                            <div className="mt-1 flex flex-wrap gap-1">
                              <span className="px-2 py-0.5 rounded-full bg-slate-50 border border-slate-200 text-slate-700">
                                semantic: {r.components.semantic}
                              </span>
                              <span className="px-2 py-0.5 rounded-full bg-slate-50 border border-slate-200 text-slate-700">
                                overlap: {r.components.skill_overlap}
                              </span>
                              <span className="px-2 py-0.5 rounded-full bg-slate-50 border border-slate-200 text-slate-700">
                                experience: {r.components.experience_score}
                              </span>
                              <span className="px-2 py-0.5 rounded-full bg-slate-50 border border-slate-200 text-slate-700">
                                coverage: {r.components.coverage}
                              </span>
                              <span className="px-2 py-0.5 rounded-full bg-slate-50 border border-slate-200 text-slate-700">
                                must coverage: {r.components.must_coverage} (
                                {r.components.must_counts})
                              </span>
                              <span className="px-2 py-0.5 rounded-full bg-slate-50 border border-slate-200 text-slate-700">
                                nice coverage:{" "}
                                {r.components.nice_coverage === null
                                  ? "N/A"
                                  : `${r.components.nice_coverage} (${r.components.nice_counts})`}
                              </span>
                            </div>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}

              {results.length === 0 && !loading && (
                <div className="px-4 py-3 text-[10px] text-slate-400">
                  Results will appear here after running a match.
                </div>
              )}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
