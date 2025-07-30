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
print(st.secrets)

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
    time.sleep(2)
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

        excel_bytes = excel_file.read()
        df = pd.read_excel(io.BytesIO(excel_bytes))
        df[['Generated answer','Public chunks','Client chunks']] = ""

        progress = st.progress(0)
        total = len(df)
        results = []
        for i, row in df.iterrows():
            number = row["#"]
            print(number)
            if math.isnan(number):
                print("skipped")
                continue
            q = row['Question']
            example = row['Best practice answer']
            pub_hits = retrieve_chunks(q, pub_idx, pub_meta, pub_chunks, top_k)
            cli_hits = retrieve_chunks(q, cli_idx, cli_meta, cli_chunks, top_k)

            ctx = "PUBLIC CONTEXT:\n" + "\n".join(f"[{src}] {txt}" for src,txt in pub_hits)
            ctx += "\n\nCLIENT CONTEXT:\n" + "\n".join(f"[{src}] {txt}" for src,txt in cli_hits)
            ctx += f"\n\nUSE THE FOLLOWING TEXT AS EXAMPLE FOR FORMATTING (DO NOT EXCEED LEN):\n{example}"

            ans = get_llm_response(prompt=q, context=ctx)
            df.at[i,'Generated answer'] = ans
            df.at[i,'Public chunks']    = " | ".join(txt for _,txt in pub_hits)
            df.at[i,'Client chunks']    = " | ".join(txt for _,txt in cli_hits)

            progress.progress((i+1)/total)

        # show results
        st.success("Done! Here are the answers:")
        st.dataframe(df)

        # save into the original workbook to preserve other sheets/formatting
        import openpyxl
        # load workbook from uploaded bytes
        wb = openpyxl.load_workbook(io.BytesIO(excel_bytes))
        # target first sheet
        sheet_name = wb.sheetnames[0]
        ws = wb[sheet_name]
        # clear existing rows except header
        for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
            for cell in row:
                cell.value = None
        # write header row
        for col_idx, col_name in enumerate(df.columns, start=1):
            ws.cell(row=1, column=col_idx, value=col_name)
        # write dataframe rows
        for row_idx, row_data in enumerate(df.values, start=2):
            for col_idx, value in enumerate(row_data, start=1):
                ws.cell(row=row_idx, column=col_idx, value=value)
        # output modified workbook
        output = io.BytesIO()
        wb.save(output)
        output.seek(0)
        st.download_button(
            "Download results as Excel",
            data=output,
            file_name="questions_with_answers.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
