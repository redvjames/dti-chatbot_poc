__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st

# Create columns for the title and logo
col1, col2 = st.columns([3.5, 1])  # Adjust the ratio as needed

# Title in the first column
with col1:
    st.title("📄 DTI Chatbot POC")
    st.write(
        "This app answers questions based on FAQs found [here](https://podrs.dti.gov.ph/#/user-control/user-faqs). "
    )
# Logo and "Developed by CAIR" text in the second column
with col2:
    st.image("images/CAIR_cropped.png", use_column_width=True)
    st.markdown(
        """
        <div style="text-align: right; margin-top: -10px;">
            Developed by CAIR
        </div>
        """, 
        unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

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
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableAssign

from dotenv import load_dotenv
load_dotenv()

from prompts import query_extract_prompt, dti_rag_prompt

llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.1)

# RETRIEVER 
CHROMA_PATH = "chroma"

embedding_function = OpenAIEmbeddings()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
retriever =  db.as_retriever(search_kwargs={'k': n_retrieved_docs})

def format_docs(docs):
    return f"\n\n".join(f"[FAQ]" + doc.page_content.replace("\n", " ") for n, doc in enumerate(docs, start=1))

chain = query_extract_prompt | llm | {"context": retriever | format_docs, "question": RunnablePassthrough()} | dti_rag_prompt | llm

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container

    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = chain.invoke({"query": prompt})
    # response = f"Echo: {prompt.upper()}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# st.write(message)
