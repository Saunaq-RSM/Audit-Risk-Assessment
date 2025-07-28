# client_public_retrieval.py
"""
Script to generate LLM answers for a set of questions by including full source texts directly in the prompt,
using dedicated Azure OpenAI endpoint for chat.

Inputs:
 - Public PDFs: Business Register, Financial Statement
 - Client PDFs: Interview Transcript, Audit File
 - Excel with two columns: Question, Best practice answer

Outputs:
 - New Excel with columns: Question, Best practice answer, Generated answer

Usage:
 python client_public_retrieval.py
"""
import os
import re
import requests
import pandas as pd
import PyPDF2
from typing import Dict
import tiktoken

# Azure endpoint & key configurations
LLM_ENDPOINT = (
    "https://ai-sieffersprojecthub149512100299.openai.azure.com/"
    "openai/deployments/RSMNL-ICS-TP-Functional-Analysis-o4-mini/chat/completions?api-version=2025-01-01-preview"
)
LLM_API_KEY = "ACMAvytOGdUVjjqbAepgJa1QsirxDkHnTbB2p2syPDPls580uF8uJQQJ99BFACfhMk5XJ3w3AAAAACOGOQ4Z"

# Utility: clean and preserve paragraphs
def clean_text(raw: str) -> str:
    # remove control chars except newline
    cleaned = re.sub(r'[\x00-\x09\x0B-\x1F\x7F]+', ' ', raw)
    # collapse multiple spaces/tabs
    cleaned = re.sub(r'[ \t]{2,}', ' ', cleaned)
    # split into paragraphs on existing newlines
    paras = [p.strip() for p in cleaned.splitlines() if p.strip()]
    # filter out very short garbage paragraphs
    paras = [p for p in paras if len(re.findall(r'\w', p)) > 5]
    # join with extra newline between paragraphs
    return '\n\n'.join(paras)

# Extract main body text via pdfplumber
import pdfplumber

def load_main_body(path: str, margin: int = 50) -> str:
    texts = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                x0, y0, x1, y1 = page.bbox
                crop = (x0+margin, y0+margin, x1-margin, y1-margin)
                body = page.within_bbox(crop).extract_text() or ""
                texts.append(body)
    except FileNotFoundError:
        return ""
    return clean_text("\n".join(texts))

# Chat completion via Azure endpoint
def get_llm_response(prompt: str, context: str) -> str:
    headers = {"Content-Type": "application/json", "api-key": LLM_API_KEY}
    messages = [
        {"role": "system", "content": (
            "You are a world-class corporate analysis assistant for an expert audit team. "
            "Your task is to answer a predefined set of due diligence questions about the company. "
            "Use the following context to help you answer the prompt:\n" + context
        )},
        {"role": "user", "content": prompt}
    ]
    resp = requests.post(LLM_ENDPOINT, headers=headers, json={"messages": messages})
    resp.raise_for_status()
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(prompt+ context)
    print("num_tokens: ", len(tokens))

    return resp.json()["choices"][0]["message"]["content"].strip()

# Main workflow
if __name__ == '__main__':
    public_docs = {
        'Business Register': 'documents/register.pdf',
        'Financial Statement': 'documents/financial_statement.pdf'
    }
    client_docs = {
        'Interview Transcript': 'documents/interview_transcript.pdf',
        'Audit File': 'documents/audit_file.pdf'
    }

    # load and clean sources
    pub_texts: Dict[str, str] = {name: load_main_body(path) for name, path in public_docs.items()}
    cli_texts: Dict[str, str] = {name: load_main_body(path) for name, path in client_docs.items()}

    # build context string
    context_parts = ["PUBLIC SOURCES:"]
    for name, txt in pub_texts.items():
        if txt:
            context_parts.append(f"=== {name} ===\n{txt}")
    context_parts.append("CLIENT SOURCES:")
    for name, txt in cli_texts.items():
        if txt:
            context_parts.append(f"=== {name} ===\n{txt}")
    base_context = "\n\n".join(context_parts)

    # read questions
    df = pd.read_excel('documents/Input GPT.xlsx')
    df['Generated answer'] = ''

    for idx, row in df.iterrows():
        prompt = row['Question']
        example = row['Best practice answer']
        full_ctx = base_context + f"\n\nUSE THE FOLLOWING STYLE & LENGTH AS EXAMPLE. ONLY USE BULLET POINTS IF IT'S IN THIS EXAMPLE:\n{example}\n"
        answer = get_llm_response(prompt=prompt, context=full_ctx)
        df.at[idx, 'Generated answer'] = answer
        print(answer)
        print(f"Q{idx+1} done")

    # save results
    df.to_excel('questions_with_answers.xlsx', index=False)
    print("Completed. Output saved to questions_with_answers.xlsx")

# test_endpoints.py remains unchanged
