# ResumeMatch — CS547 Information Retrieval

Streamlit application for resume-to-job retrieval using the LinkedIn job postings dataset.

## Current status
**Work in progress.** This initial GitHub commit provides a deployable Streamlit interface and a BM25 retrieval baseline. The large LinkedIn dataset is **not** included in GitHub; until a compact dataset is added, the app displays a clear setup notice. Full dense retrieval, RRF, MinHash, skill-gap analysis, and labeled evaluation remain to be completed.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data
Place a compact `data/postings.csv` with columns `job_id,title,company_name,description,location,job_posting_url`. The app indexes up to 20,000 rows. Do not commit personal resumes or the original 493 MB CSV.

## Deploy
1. Open https://share.streamlit.io/
2. Select repository `rud-automatic/ResumeMatch-CS547`, branch `main`, entrypoint `app.py`.
3. Deploy. Search becomes available when a prepared compact postings CSV is included.

## Academic integrity / limitations
Skill matching uses a small keyword dictionary. Eligibility detection is heuristic and must not be treated as legal advice. Search quality has not yet been evaluated.
