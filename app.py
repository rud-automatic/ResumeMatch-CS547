import re
import io
from pathlib import Path
import pandas as pd
import streamlit as st
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

def eligibility(text):
    rules={
        "Potential sponsorship restriction":r"(?i)(?:no|without|cannot provide|unable to provide)\s+(?:visa\s+)?sponsorship",
        "Potential citizenship requirement":r"(?i)(?:u\.?s\.?|united states)\s+citizenship\s+required|must be (?:a )?(?:u\.?s\.?|united states) citizen",
        "Potential clearance requirement":r"(?i)(?:active|current|requires?)\s+(?:secret|top secret|security)\s+clearance",
    }
    sentences=re.split(r"(?<=[.!?])\s+|\n+",str(text))
    return [(name,next((s.strip()[:300] for s in sentences if re.search(pattern,s)),None)) for name,pattern in rules.items()]

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
texts=(candidates.title.fillna("")+" "+candidates.description.fillna("").str.slice(0,2500)).tolist()
bm25=BM25Okapi([tokenize(t) for t in texts])
scores=bm25.get_scores(tokenize(query))
ranked=candidates.assign(score=scores)
if mode.startswith("Hybrid"):
    # Lightweight lexical vector baseline; NOT a dense neural embedding model.
    matrix=TfidfVectorizer(stop_words="english",max_features=12000).fit_transform(texts+[query])
    semantic=cosine_similarity(matrix[-1],matrix[:-1]).ravel()
    bm_order=sorted(range(len(scores)),key=lambda i:(-scores[i],i))
    vec_order=sorted(range(len(semantic)),key=lambda i:(-semantic[i],i))
    bm_rank={idx:r+1 for r,idx in enumerate(bm_order)}
    vec_rank={idx:r+1 for r,idx in enumerate(vec_order)}
    fused=[1/(60+bm_rank[i])+1/(60+vec_rank[i]) for i in range(len(scores))]
    ranked=ranked.assign(rrf_score=fused).sort_values("rrf_score",ascending=False).head(count)
else:
    ranked=ranked.sort_values("score",ascending=False).head(count)
mine=skills(query)
for _,job in ranked.iterrows():
    with st.container(border=True):
        st.subheader(str(job.title))
        st.caption(f"{job.get('company_name','Unknown')} · {job.get('location','Unknown')}")
        required=skills(job.description)
        st.write("**Matched skills:**",", ".join(sorted(mine&required)) or "None detected")
        st.write("**Missing skills:**",", ".join(sorted(required-mine)) or "None detected")
        st.caption(f"BM25 score: {job.score:.2f}" + (f" · RRF: {job.rrf_score:.4f}" if "rrf_score" in ranked.columns else ""))
        for label,proof in eligibility(job.description):
            if proof: st.warning(f"{label}: {proof}")
        with st.expander("Job description"): st.write(str(job.description)[:4000])
        url=job.get("job_posting_url")
        if isinstance(url,str) and url.startswith(("https://","http://")):
            st.link_button("View job",url)
