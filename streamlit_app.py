import streamlit as st
import pandas as pd
import tempfile
import os
import io
from script_backup import load_main_body, clean_text, get_llm_response

st.set_page_config(page_title="Audit Due Diligence Assistant", layout="wide")

# -- Load secrets --
LLM_ENDPOINT = st.secrets["AZURE_API_ENDPOINT"]
LLM_API_KEY = st.secrets["AZURE_API_KEY"]

# override backend constants
import script_backup as backend
backend.LLM_ENDPOINT = LLM_ENDPOINT
backend.LLM_API_KEY = LLM_API_KEY

st.title("Audit Due Diligence Assistant")
st.write("Upload your public and client PDFs plus the Input GPT.xlsx to generate answers.")

# File uploaders
with st.expander("Upload PDFs and Excel"):  # collapsed by default
    public_files = st.file_uploader(
        "Public PDFs (Business Register, Financial Statement)",
        type=["pdf"], accept_multiple_files=True, key="public"
    )
    client_files = st.file_uploader(
        "Client PDFs (Interview Transcript, Audit File)",
        type=["pdf"], accept_multiple_files=True, key="client"
    )
    excel_file = st.file_uploader(
        "Input GPT Excel (questions & best practice answers)",
        type=["xlsx"], key="excel"
    )

if st.button("Generate Answers"):
    if not excel_file:
        st.error("Please upload the Input GPT.xlsx file.")
    else:
        # Save uploaded PDFs to temp files
        temp_dir = tempfile.mkdtemp()
        public_docs = {}
        for f in public_files:
            path = os.path.join(temp_dir, f.name)
            with open(path, 'wb') as out:
                out.write(f.read())
            public_docs[f.name] = path
        client_docs = {}
        for f in client_files:
            path = os.path.join(temp_dir, f.name)
            with open(path, 'wb') as out:
                out.write(f.read())
            client_docs[f.name] = path

        # Read questions
        df = pd.read_excel(excel_file)
        df['Generated answer'] = ''
        df['Public sources used'] = ''
        df['Client sources used'] = ''

        # Process each question
        progress = st.progress(0)
        total = len(df)
        for i, row in df.iterrows():
            q = row['Question']
            bp = row['Best practice answer']
            # build context from uploaded docs
            # load & clean
            pub_texts = {name: clean_text(load_main_body(path)) for name, path in public_docs.items()}
            cli_texts = {name: clean_text(load_main_body(path)) for name, path in client_docs.items()}
            # simple retrieval: list keys as sources
            pub_used = [name for name, txt in pub_texts.items() if txt]
            cli_used = [name for name, txt in cli_texts.items() if txt]
            # assemble prompt context
            ctx_parts = ["PUBLIC SOURCES:"]
            for name, txt in pub_texts.items():
                if txt:
                    ctx_parts.append(f"=== {name} ===\n{txt}")
            ctx_parts.append("CLIENT SOURCES:")
            for name, txt in cli_texts.items():
                if txt:
                    ctx_parts.append(f"=== {name} ===\n{txt}")
            base_ctx = "\n\n".join(ctx_parts)
            full_ctx = base_ctx + f"\n\nUSE THE FOLLOWING STYLE & LENGTH AS EXAMPLE:\n{bp}\n"

            # call backend LLM
            answer = get_llm_response(prompt=q, context=full_ctx)
            df.at[i, 'Generated answer'] = answer
            df.at[i, 'Public sources used'] = "; ".join(pub_used)
            df.at[i, 'Client sources used'] = "; ".join(cli_used)
            progress.progress((i+1)/total)

        # Output excel for download
        output = io.BytesIO()
        df.to_excel(output, index=False)
        output.seek(0)
        st.success("Generation complete!")
        st.download_button(
            label="Download answers Excel",
            data=output,
            file_name="questions_with_answers.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
