__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st

# Create columns for the title and logo
col1, col2 = st.columns([3.5, 1])  # Adjust the ratio as needed

# Title in the first column
with col1:
    st.title("📄 DFA Q&A")
    st.write(
        "This app answers questions based on FAQs found [here](https://consular.dfa.gov.ph/faqs-menu?). "
    )
# Logo and "Developed by CAIR" text in the second column
with col2:
    st.image("images/CAIR_cropped.png", use_column_width=True)
    st.markdown(
        """
        <div style="text-align: center; margin-top: -10px;">
            Developed by CAIR
        </div>
        """, 
        unsafe_allow_html=True)
    
question = st.text_area(
    "Enter your email text here!",
    placeholder="""Dear DFA,

My name is Juan Dela Cruz. How can I apply for a passport?

Thank you!
    """,
    height=200
)

# Adding the sidebar for selecting the repo_id
st.sidebar.title("Model Selection")
repo_id = st.sidebar.selectbox(
    "Select the HuggingFace model:",
    options=[
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.2",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "microsoft/Phi-3-mini-4k-instruct",
        "google/gemma-2-2b-it"
    ],
    index=0  # Default selection
)

n_retrieved_docs = st.sidebar.number_input(
    "Number of documents to retrieve:",
    min_value=1,
    max_value=20,
    value=5,  # Default value
    step=1
)

from langchain_huggingface import HuggingFaceEndpoint
from langchain_chroma.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableAssign
from langchain_core.output_parsers import StrOutputParser

if question:
    llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.1)

    from dotenv import load_dotenv
    load_dotenv()

    from prompts import prompt, dfa_rag_prompt

    # RETRIEVER 
    CHROMA_PATH = "chroma"

    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    retriever =  db.as_retriever(search_kwargs={'k': n_retrieved_docs})

    def format_docs(docs):
        return f"\n\n".join(f"[FAQ]" + doc.page_content.replace("\n", " ") for n, doc in enumerate(docs, start=1))

    chain = prompt | llm | {"context": retriever | format_docs, "question": RunnablePassthrough()} | dfa_rag_prompt | llm

    input_dict = {"question": question}

    response = chain.invoke(input_dict)

    st.write(response)
