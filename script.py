# client_public_retrieval.py
"""
Script to generate LLM answers for a set of questions using RAG with token-based chunk-level retrieval.
Chunks are created by splitting text into groups of N tokens, and newlines are removed in cleaning.
"""
import os
import re
import requests
import pandas as pd
import pdfplumber
import faiss
import numpy as np
import tiktoken
from typing import Dict, List, Tuple

# Azure endpoint & key configurations
LLM_ENDPOINT = (
    "https://ai-sieffersprojecthub149512100299.openai.azure.com/"
    "openai/deployments/RSMNL-ICS-TP-Functional-Analysis-o4-mini/chat/completions?api-version=2025-01-01-preview"
)
LLM_API_KEY = ""
EMBED_ENDPOINT = (
    "https://ai-sieffersprojecthub149512100299.openai.azure.com/"
    "openai/deployments/text-embedding-ada-002-test/embeddings?api-version=2023-05-15"
)
EMBED_API_KEY = ""

# --- Tokenizer ---
ENC = tiktoken.get_encoding("cl100k_base")

# --- Cleaning ---
def clean_text(raw: str) -> str:
    """
    Remove newlines and control characters, collapse whitespace to single space.
    """
    text = re.sub(r'[\x00-\x1F\x7F]+', ' ', raw)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- PDF extraction ---
def load_main_body(path: str, margin: int = 50) -> str:
    """
    Extract central body text via pdfplumber, then clean.
    """
    pieces = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                x0, y0, x1, y1 = page.bbox
                crop = (x0+margin, y0+margin, x1-margin, y1-margin)
                txt = page.within_bbox(crop).extract_text() or ""
                pieces.append(txt)
    except FileNotFoundError:
        return ""
    full = " ".join(pieces)
    return clean_text(full)

# --- Chunking by tokens ---
def chunk_text_tokens(text: str, chunk_size: int = 200) -> List[str]:
    """
    Split `text` into chunks of `chunk_size` tokens each, using tiktoken.
    """
    token_ids = ENC.encode(text)
    chunks = []
    for i in range(0, len(token_ids), chunk_size):
        slice_ids = token_ids[i:i+chunk_size]
        chunk = ENC.decode(slice_ids)
        chunks.append(chunk)
    return chunks

# --- Embeddings ---
def embed_texts(texts: List[str]) -> np.ndarray:
    headers = {"Content-Type": "application/json", "api-key": EMBED_API_KEY}
    resp = requests.post(EMBED_ENDPOINT, headers=headers, json={"input": texts})
    resp.raise_for_status()
    embs = [d["embedding"] for d in resp.json().get("data", [])]
    arr = np.array(embs, dtype=np.float32)
    faiss.normalize_L2(arr)
    return arr

# --- Indexing ---
def index_documents(
    docs: Dict[str, str],
    chunk_size: int = 500
) -> Tuple[faiss.IndexFlatIP, List[Tuple[str,int]], List[str]]:
    """
    Create FAISS index over token-based chunks.
    Returns index, metadata, and chunk texts.
    """
    all_chunks, meta = [], []
    for source, text in docs.items():
        if not text:
            continue
        for idx, chunk in enumerate(chunk_text_tokens(text, chunk_size)):
            all_chunks.append(chunk)
            meta.append((source, idx))
    if not all_chunks:
        return None, [], []
    embeddings = embed_texts(all_chunks)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, meta, all_chunks

# --- Retrieval ---
def retrieve_chunks(
    query: str,
    index: faiss.IndexFlatIP,
    meta: List[Tuple[str,int]],
    chunks: List[str],
    k: int = 10
) -> List[Tuple[str, str]]:
    """
    Return top-k (source_name, chunk_text) for the query.
    """
    if index is None or not chunks:
        return []
    q_emb = embed_texts([query])[0]
    D, I = index.search(np.array([q_emb]), k)
    return [(meta[i][0], chunks[i]) for i in I[0]]

# --- Chat call ---
def get_llm_response(prompt: str, context: str) -> str:
    headers = {"Content-Type": "application/json", "api-key": LLM_API_KEY}
    messages = [
        {"role": "system", "content": (
            "You are a world-class corporate analysis assistant for an expert audit team."
            " Use the context to answer due diligence questions.\n" + context
        )},
        {"role": "user", "content": prompt}
    ]
    r = requests.post(LLM_ENDPOINT, headers=headers, json={"messages": messages})
    r.raise_for_status()
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(prompt+ context)
    print("num_tokens: ", len(tokens))

    return r.json()["choices"][0]["message"]["content"].strip()

# --- Main workflow ---
if __name__ == '__main__':
    public_docs = {
        'Business Register': 'documents/register.pdf',
        'Financial Statement': 'documents/financial_statement.pdf'
    }
    client_docs = {
        'Interview Transcript': 'documents/interview_transcript.pdf',
        'Audit File': 'documents/audit_file.pdf'
    }

    # Load and index
    pub_texts = {n: load_main_body(p) for n, p in public_docs.items()}
    cli_texts = {n: load_main_body(p) for n, p in client_docs.items()}
    pub_idx, pub_meta, pub_chunks = index_documents(pub_texts, chunk_size=750)
    cli_idx, cli_meta, cli_chunks = index_documents(cli_texts, chunk_size=750)

    df = pd.read_excel('documents/Input GPT.xlsx')
    df['Generated answer'] = ''
    df['Public chunks'] = ''
    df['Client chunks'] = ''

    for i, row in df.iterrows():
        if i == 0:
            continue
        q = row['Question']
        pub_hits = retrieve_chunks(q, pub_idx, pub_meta, pub_chunks, k=10)
        cli_hits = retrieve_chunks(q, cli_idx, cli_meta, cli_chunks, k=10)

        # Log retrieved chunks
        print(f"Q{i+1}: {q}")
        # for src, txt in pub_hits:
        #     print(f" PUBLIC [{src}]: {txt[:100]}...")
        # for src, txt in cli_hits:
        #     print(f" CLIENT [{src}]: {txt[:100]}...")

        # Assemble context including chunk texts
        ctx_parts = ["PUBLIC CONTEXT:"]
        ctx_parts += [f"[{src}] {txt}" for src, txt in pub_hits]
        ctx_parts.append("CLIENT CONTEXT:")
        ctx_parts += [f"[{src}] {txt}" for src, txt in cli_hits]
        full_ctx = "\n\n".join(ctx_parts) + f"\n\nUSE THE FOLLOWING TEXT AS EXAMPLE FOR FORMATTING YOUR RESPONSE. DO NOT LET YOUR RESPONSE EXCEED THE LENGTH OF THE FOLLOWING TEXT:\n{row['Best practice answer']}\n"

        ans = get_llm_response(prompt=q, context=full_ctx)
        df.at[i, 'Generated answer'] = ans
        df.at[i, 'Public chunks'] = ' | '.join(txt for _, txt in pub_hits)
        df.at[i, 'Client chunks'] = ' | '.join(txt for _, txt in cli_hits)

        print("Answer:", ans)
        print("---")

    df.to_excel('questions_with_answers.xlsx', index=False)
    print("Completed: questions_with_answers.xlsx")
