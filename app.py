import re
import io
from pathlib import Path
import pandas as pd
import streamlit as st
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

st.set_page_config(page_title="ResumeMatch | CS547",page_icon="💼",layout="wide")
st.title("💼 ResumeMatch")
st.caption("CS547 Information Retrieval · Work-in-progress deployment")
DATA=Path(__file__).parent/"data"/"postings.csv"

@st.cache_data
def load_jobs():
    if not DATA.exists():
        return pd.DataFrame()
    cols=["job_id","title","company_name","description","location","job_posting_url"]
    jobs=pd.read_csv(DATA,usecols=lambda x:x in cols,nrows=20000)
    return jobs.dropna(subset=["title","description"]).drop_duplicates("job_id").reset_index(drop=True)

def extract_text(upload):
    if upload is None: return ""
    data=upload.getvalue()
    if upload.name.lower().endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages)
    if upload.name.lower().endswith(".docx"):
        from docx import Document
        return "\n".join(p.text for p in Document(io.BytesIO(data)).paragraphs)
    return data.decode("utf-8",errors="replace")

def tokenize(s):
    return re.findall(r"[a-z0-9+#.]+",str(s).lower())

SKILLS=["python","java","javascript","react","sql","aws","docker","kubernetes","git","linux","html","css","mongodb","pytorch","tensorflow","machine learning","excel","tableau","flask","django","fastapi","c++","azure"]
def skills(text):
    s=str(text).lower()
    return {x for x in SKILLS if re.search(r"(?<![a-z0-9])"+re.escape(x)+r"(?![a-z0-9])",s)}

def evidence_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\\s+|\\n+", str(text)) if s.strip()]

def eligibility_checks(text):
    checks = [
        ("Visa sponsorship", [
            ("Potential restriction", r"\\b(?:no|not offering|does not offer|unable to provide|cannot provide|without)\\s+(?:visa\\s+)?sponsorship\\b|\\b(?:sponsorship|visa sponsorship)\\s+(?:is\\s+)?(?:not available|unavailable)\\b"),
            ("Sponsorship mentioned", r"\\b(?:visa sponsorship|sponsor(?:ship)?\\s+(?:available|provided)|will sponsor)\\b"),
        ]),
        ("Citizenship", [("Requirement found", r"\\b(?:must be|requires?|required to be)\\s+(?:a\\s+)?(?:u\\.?s\\.?|united states)\\s+citizen\\b|\\b(?:u\\.?s\\.?|united states)\\s+citizenship\\s+(?:is\\s+)?required\\b")]),
        ("Security clearance", [("Requirement found", r"\\b(?:must (?:have|hold|obtain)|requires?|required|active|current)\\b[^.!?\\n]{0,65}\\b(?:security clearance|secret clearance|top secret clearance)\\b")]),
        ("Experience", [("Requirement found", r"\\b(?:at least|minimum of|minimum|\\d+\\s*[-–]\\s*\\d+|\\d+\\+?)\\s*\\d*\\s*(?:years|yrs)\\s+(?:of\\s+)?(?:relevant\\s+|professional\\s+|work\\s+)?experience\\b|\\b\\d+\\+?\\s*(?:years|yrs)\\s+(?:of\\s+)?(?:relevant\\s+|professional\\s+|work\\s+)?experience\\b")]),
    ]
    sentences=evidence_sentences(text)
    output=[]
    for name, patterns in checks:
        status, proof = "Not specified", None
        for candidate_status, pattern in patterns:
            proof=next((line[:320] for line in sentences if re.search(pattern,line,re.I)),None)
            if proof:
                status=candidate_status
                break
        output.append((name,status,proof))
    return output

SKILL_GUIDES={
    "python":"Practice Python scripting, APIs, and a small portfolio project.",
    "sql":"Practice SQL joins, aggregation, and query optimization.",
    "aws":"Deploy a small application using AWS services.",
    "docker":"Containerize a Python or web application with Docker.",
    "kubernetes":"Learn deployments, services, and container orchestration basics.",
    "react":"Build a React interface that consumes a REST API.",
    "java":"Build and test a Java backend application.",
    "machine learning":"Train and evaluate a model on a public dataset.",
    "pytorch":"Implement and train a small neural network in PyTorch.",
    "tensorflow":"Build a simple model with TensorFlow.",
    "azure":"Deploy a small application on Azure.",
    "git":"Practice branching, pull requests, and resolving merge conflicts.",
    "linux":"Practice command-line tools, permissions, and shell scripts.",
    "mongodb":"Build a CRUD API using MongoDB.",
    "javascript":"Practice modern JavaScript and asynchronous APIs.",
    "excel":"Practice pivot tables, formulas, and data cleaning.",
    "tableau":"Create an interactive dashboard using a sample dataset.",
}

def skill_advice(resume_text, job_text):
    candidate=skills(resume_text)
    mentioned=skills(job_text)
    matched=sorted(candidate & mentioned)
    missing=sorted(mentioned - candidate)
    return matched, missing

@st.cache_resource(show_spinner="Building search index (first run only)...")
def build_index(data_path, modified_time):
    frame=pd.read_csv(data_path,usecols=lambda x:x in ["job_id","title","company_name","description","location","job_posting_url"],nrows=20000)
    frame=frame.dropna(subset=["title","description"]).drop_duplicates("job_id").reset_index(drop=True)
    texts=(frame.title.fillna("")+" "+frame.description.fillna("").str.slice(0,1200)).tolist()
    tokens=[tokenize(t) for t in texts]
    bm25=BM25Okapi(tokens)
    vectorizer=TfidfVectorizer(stop_words="english",max_features=8000,dtype=np.float32)
    matrix=vectorizer.fit_transform(texts)
    return frame,bm25,vectorizer,matrix

jobs=load_jobs()
if jobs.empty:
    st.warning("The real LinkedIn dataset has not been published to this repository yet. This is a code-preview deployment, not the completed project.")
    st.info("To enable search, add a reduced data/postings.csv containing job_id, title, company_name, description, location and job_posting_url. Do not commit personal resumes.")
    st.stop()

with st.sidebar:
    st.header("Search")
    uploaded=st.file_uploader("Upload resume",type=["pdf","docx","txt"])
    query=st.text_area("Or enter skills / experience")
    location=st.text_input("Location filter")
    count=st.slider("Number of results",5,20,10)
    mode=st.selectbox("Ranking method",["Hybrid RRF (BM25 + TF-IDF)","BM25 only"])
    run=st.button("Find matching jobs",type="primary")
st.metric("Indexed job postings",f"{len(jobs):,}")
st.caption("BM25 or BM25 + TF-IDF RRF baseline. Neural dense retrieval, MinHash and formal evaluation remain planned; eligibility is heuristic.")
if not run: st.stop()
query=extract_text(uploaded)+"\n"+query
if not query.strip():
    st.error("Please upload a resume or enter a query.")
    st.stop()
candidates=jobs
if location:
    candidates=candidates[candidates.location.fillna("").str.contains(location,case=False,regex=False)]
if candidates.empty:
    st.info("No jobs matched your location filter.")
    st.stop()
with st.spinner("Finding matching jobs..."):
    indexed,bm25,vectorizer,matrix=build_index(str(DATA),DATA.stat().st_mtime_ns)
    selected=indexed.index[indexed.index.isin(candidates.index)].to_numpy()
    scores_all=bm25.get_scores(tokenize(query))
    scores=scores_all[selected]
    ranked=indexed.iloc[selected].copy().assign(score=scores)
if mode.startswith("Hybrid"):
    # Lightweight lexical vector baseline; NOT a dense neural embedding model.
    semantic=np.asarray((matrix[selected] @ vectorizer.transform([query]).T).toarray()).ravel()
    bm_order=np.argsort(-scores,kind="stable")
    vec_order=np.argsort(-semantic,kind="stable")
    bm_rank={idx:r+1 for r,idx in enumerate(bm_order)}
    vec_rank={idx:r+1 for r,idx in enumerate(vec_order)}
    fused=[1/(60+bm_rank[i])+1/(60+vec_rank[i]) for i in range(len(scores))]
    ranked=ranked.assign(rrf_score=fused).sort_values("rrf_score",ascending=False).head(count)
else:
    ranked=ranked.sort_values("score",ascending=False).head(count)
for _,job in ranked.iterrows():
    with st.container(border=True):
        st.subheader(str(job.title))
        st.caption(f"{job.get('company_name','Unknown')} · {job.get('location','Unknown')}")
        matched,missing=skill_advice(query,job.description)
        st.markdown("**Skill-gap advisor**")
        st.write("**Matched skills:**", ", ".join(matched) or "None detected")
        st.write("**Skills not found in your input:**", ", ".join(missing) or "None detected")
        if matched or missing:
            st.progress(len(matched)/max(1,len(matched)+len(missing)),text=f"Keyword coverage: {len(matched)}/{len(matched)+len(missing)} skills mentioned in this posting")
        if missing:
            with st.expander("Suggested learning priorities",expanded=True):
                for skill in missing[:5]:
                    st.write(f"**{skill.title()}** — {SKILL_GUIDES.get(skill,'Find an introductory tutorial and build a small project using this skill.')}")
        st.caption("Keyword-based suggestions only: skills mentioned in a posting are not necessarily mandatory. Missing means not found in the supplied resume/query.")
        st.caption(f"BM25 score: {job.score:.2f}" + (f" · RRF: {job.rrf_score:.4f}" if "rrf_score" in ranked.columns else ""))
        with st.expander("Eligibility and requirements",expanded=True):
            for label,status,proof in eligibility_checks(job.description):
                st.write(f"**{label}: {status}**")
                if proof:
                    st.caption(f"Evidence from posting: {proof}")
            st.caption("This is automated text detection, not a determination of eligibility. 'Not specified' does not mean eligible; confirm with the employer.")
        with st.expander("Job description"): st.write(str(job.description)[:4000])
        url=job.get("job_posting_url")
        if isinstance(url,str) and url.startswith(("https://","http://")):
            st.link_button("View job",url)
