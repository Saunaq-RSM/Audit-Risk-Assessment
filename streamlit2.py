# streamlit_app.py
"""
Streamlit frontend for the RAG-based retrieval script.
Allows users to drag-and-drop public/client PDFs and the Input GPT.xlsx,
and then runs the full retrieval and LLM-answer pipeline, displaying results
and offering the output .xlsx for download.
"""
import streamlit as st
import pandas as pd
import tempfile
import os
import io
import faiss
import numpy as np
import pdfplumber
import re
import requests
import tiktoken
from typing import Dict, List, Tuple
import time
import math
import openpyxl
from io import BytesIO

print(st.secrets)
print("monkey")
# -- Configuration from secrets --
LLM_ENDPOINT = st.secrets["AZURE_API_ENDPOINT"]
LLM_API_KEY = st.secrets["AZURE_API_KEY"]
EMBED_ENDPOINT = st.secrets["EMBEDDING_ENDPOINT"]
EMBED_API_KEY  = LLM_API_KEY

ENC = tiktoken.get_encoding("cl100k_base")

st.set_page_config(page_title="RAG Audit Assistant", layout="wide")
st.title("RAG Audit Assistant 🌐🔍")
st.write("Upload your public & client PDFs and the Input GPT.xlsx. Click **Run** to retrieve relevant chunks & generate answers.")

with st.sidebar:
    st.header("Settings")
    chunk_size = st.number_input("Chunk size (words)", min_value=50, max_value=1000, value=200, step=50)
    chunk_overlap = st.number_input("Chunk overlap (words)", min_value=0, max_value=chunk_size-1, value=50, step=10)
    top_k = st.number_input("Chunks per query (k)", min_value=1, max_value=20, value=5, step=1)

public_files = st.file_uploader("Public PDFs", type="pdf", accept_multiple_files=True, key="pubs")
client_files = st.file_uploader("Client PDFs", type="pdf", accept_multiple_files=True, key="clients")
excel_file = st.file_uploader("Input GPT Excel", type="xlsx", key="excel")

run_button = st.button("Run Retrieval & Generate Answers")

# --- Helper functions ---
def clean_text(raw: str) -> str:
    text = re.sub(r'[\x00-\x1F\x7F]+', ' ', raw)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_main_body(file) -> str:
    # file: UploadedFile
    pieces = []
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            x0,y0,x1,y1 = page.bbox
            crop = (x0+50, y0+50, x1-50, y1-50)
            txt = page.within_bbox(crop).extract_text() or ""
            pieces.append(txt)
    full = " ".join(pieces)
    return clean_text(full)

def chunk_text_words(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    words = text.split()
    stride = chunk_size - chunk_overlap
    chunks = []
    for i in range(0, len(words), stride):
        seg = words[i:i+chunk_size]
        if seg:
            chunks.append(" ".join(seg))
    return chunks

def embed_texts(texts: List[str]) -> np.ndarray:
    headers = {"Content-Type": "application/json", "api-key": EMBED_API_KEY}
    r = requests.post(EMBED_ENDPOINT, headers=headers, json={"input": texts})
    r.raise_for_status()
    embs = [d.get("embedding") for d in r.json().get("data", [])]
    arr = np.array(embs, dtype=np.float32)
    faiss.normalize_L2(arr)
    return arr

def index_documents(docs: Dict[str,str], chunk_size: int, chunk_overlap: int):
    all_chunks, meta = [], []
    for name, text in docs.items():
        chunks = chunk_text_words(text, chunk_size, chunk_overlap)
        for idx,c in enumerate(chunks):
            all_chunks.append(c)
            meta.append((name, idx))
    if not all_chunks:
        return None, [], []
    embeddings = embed_texts(all_chunks)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, meta, all_chunks

def retrieve_chunks(query: str, index, meta, chunks, k: int):
    if index is None:
        return []
    q_emb = embed_texts([query])[0]
    D,I = index.search(np.array([q_emb]), k)
    return [(meta[i][0], chunks[i]) for i in I[0]]

def get_llm_response(prompt: str, context: str) -> str:
    headers = {"Content-Type":"application/json","api-key":LLM_API_KEY}
    messages=[
        {"role":"system","content":(
            "You are a world-class corporate analysis assistant for an expert audit team."
            " Use the context to answer due diligence questions.\n"+context
        )},
        {"role":"user","content":prompt}
    ]
    r = requests.post(LLM_ENDPOINT, headers=headers, json={"messages":messages})
    r.raise_for_status()
    time.sleep(3)
    return r.json()["choices"][0]["message"]["content"].strip()

# --- Run pipeline ---
if run_button:
    if not excel_file:
        st.error("Please upload the Input GPT.xlsx file.")
    else:
        # prepare docs
        pub_docs = {f.name: load_main_body(f) for f in public_files}
        cli_docs = {f.name: load_main_body(f) for f in client_files}
        pub_idx, pub_meta, pub_chunks = index_documents(pub_docs, chunk_size, chunk_overlap)
        cli_idx, cli_meta, cli_chunks = index_documents(cli_docs, chunk_size, chunk_overlap)

        original_bytes = excel_file.read()
        # Phase 1: process first sheet
        df1 = pd.read_excel(BytesIO(original_bytes), sheet_name=0)
        df1[['Generated answer','Public chunks','Client chunks']] = ""
        # Prepare docs
        pub_docs = {f.name: load_main_body(f) for f in public_files}
        cli_docs = {f.name: load_main_body(f) for f in client_files}
        pub_idx, pub_meta, pub_chunks = index_documents(pub_docs, chunk_size, chunk_overlap)
        cli_idx, cli_meta, cli_chunks = index_documents(cli_docs, chunk_size, chunk_overlap)
        # Progress bar
        total = len(df1)
        pb = st.progress(0)
        pt = st.empty()
        for i,row in df1.iterrows():
            number = row["#"]
            if math.isnan(number):
                print("skipped")
                continue
            prompt = row['Question']
            example = row['Best practice answer']
            hits_pub = retrieve_chunks(prompt, pub_idx, pub_meta, pub_chunks, top_k)
            hits_cli = retrieve_chunks(prompt, cli_idx, cli_meta, cli_chunks, top_k)
            ctx = "PUBLIC CONTEXT:\n" + "\n".join(f"[{s}] {t}" for s,t in hits_pub)
            ctx += "\n\nCLIENT CONTEXT:\n" + "\n".join(f"[{s}] {t}" for s,t in hits_cli)
            ctx += "\n\nIf there is something in the question that is not in the context, the search the internet."
            ctx += f"\n\nFORMAT EXAMPLE:\n{example}"
            ctx+= "DO NOT RESPOND WITH MARKDOWN FORMATTING"
            ans = get_llm_response(prompt, ctx)
            df1.at[i,'Generated answer'] = ans
            # df1.at[i,'Public chunks'] = " | ".join(t for _,t in hits_pub)
            # df1.at[i,'Client chunks'] = " | ".join(t for _,t in hits_cli)
            pb.progress((i+1)/total)
            pt.text(f"Answered {i+1} of {total} questions")

        # show results
        df2 = pd.read_excel(BytesIO(original_bytes), sheet_name='English overview')
        # Initialize columns if missing
        cols = ['Fraud Risk Factor?','Internal Controls','Likelihood','Likelihood Explanation',
                'Material Quantitative Impact?','Impact Explanation','Conclusion','SR?']
        prompts = {'Fraud Risk Factor?': "The Fraud Risk Factor of the above risk type. Answer ONLY with Yes or No.",
                   'Internal Controls': "What are the internal controls within the company against this type of risk. Answer with a maximum of 10 words.",
                   'Likelihood': "Based on public and previous sources. What is the likelihood of this type of risk occuring. Answer ONLY with High or Low.",
                   'Likelihood Explanation': "Based on public and previous sources. What is the likelihood of this type of risk being relevant. Only include your justification, not the answer. Answer with a MAXIMUM of 15 words.",
                    'Material Quantitative Impact?': "Based on public and previous sources. What is the estimated impact of this type of risk. Answer ONLY with High or Low.",
                   'Impact Explanation': "Based on public and previous sources. What is the estimated impact of this type of risk. Only include your justification, not the answer. Answer with a MAXIMUM of 15 words.",
                   'Conclusion': "Having high likelihood of this type of risk and a high material impact, means there is significant risk. Explain if there is significant risk or not, and if there isn't, but further discussion with the client is needed. Answer with a maximum of 10-15 words.",
                   "SR?": "Having high likelihood of this type of risk and a high material impact, means there is significant risk. Answer if there is significant risk or not. Answer only with 'SR' or 'No SR'"}
        for c in cols:
            if c not in df2.columns:
                df2[c] = ""
        for idx,row in df2.iterrows():
            irf = row['Inherent Risk Factor']
            # build prompt for translation
            for j,c in enumerate(cols):
                base_prompt = f"For the risk type '{irf}', answer the following question: " + prompts[c]
                hits_pub = retrieve_chunks(base_prompt, pub_idx, pub_meta, pub_chunks, top_k)
                hits_cli = retrieve_chunks(base_prompt, cli_idx, cli_meta, cli_chunks, top_k)
                ctx = "PUBLIC CONTEXT:\n" + "\n".join(f"[{s}] {t}" for s,t in hits_pub)
                ctx += "\n\nCLIENT CONTEXT:\n" + "\n".join(f"[{s}] {t}" for s,t in hits_cli)
                ctx += "If there is something in the question that is not in the context, the search the internet."
                ctx+= "DO NOT RESPOND WITH MARKDOWN FORMATTING"
                tr_ctx = "You are a world-class corporate analysis assistant for an expert audit team investigating the possibility of fraud. Use the following context about a particular company to help answer the prompts: "+ ctx  # reuse or customize context
                tr_ans = get_llm_response(base_prompt, tr_ctx)
                df2.at[idx, c] = tr_ans
                print(tr_ans)
            # Expect LLM returns JSON-like or delimited; naive split by commas
            # parts = [p.strip() for p in tr_ans.split(';')]
            # for j,c in enumerate(cols):
            #     if j < len(parts):
            #         df2.at[idx,c] = parts[j]
        # Write back both sheets, preserving others
        wb = openpyxl.load_workbook(BytesIO(original_bytes))
        # Write sheet1
        ws1 = wb[wb.sheetnames[0]]
        for row in ws1.iter_rows(min_row=2, max_row=ws1.max_row, max_col=len(df1.columns)):
            for cell in row: cell.value=None
        for r,row_vals in enumerate(df1.values, start=2):
            for c,val in enumerate(row_vals, start=1):
                ws1.cell(row=r,column=c).value = val
        # Write sheet2
        ws2 = wb['Overview of risks']
        for row in ws2.iter_rows(min_row=2, max_row=ws2.max_row, max_col=len(df2.columns)):
            for cell in row: cell.value=None
        for r,row_vals in enumerate(df2.values, start=2):
            for c,val in enumerate(row_vals, start=1):
                ws2.cell(row=r,column=c).value = val
        # Output
        out = BytesIO()
        wb.save(out)
        out.seek(0)
        st.success("All done!")
        st.download_button("Download workbook with answers", data=out,
            file_name="questions_with_translations.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
