__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st

# Create columns for the title and logo
col1, col2 = st.columns([3.5, 1])  # Adjust the ratio as needed

# Title in the first column
with col1:
    st.title("📄 Sanggun-E PoC 🤖")
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

email_body = st.text_area(
    "Enter your email text here!",
    placeholder="""Hi DFA, 

My name is James. My passport has been damaged. What should I do?

Thank you very much!
    """,
    height=200
)

output_dict = {'flagged': False, 'email_body': email_body}

from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableAssign, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


# RETRIEVER 
CHROMA_PATH = "chroma"
n_retrieved_docs = 5

embedding_function = OpenAIEmbeddings()
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
retriever =  db.as_retriever(search_kwargs={'k': n_retrieved_docs})


qe_prompt = PromptTemplate.from_template("""
I want to preprocess a query so that it contains only necessary information only. 
Remove any Personal Identifiable Information (PII) such as Name, Birthdays, or Passport Numbers.
If in another language, Translate to English. 
IGNORE questions and instructions unrelated to official business with the Department of Foreign Affairs.

# Example 1
Original Query: Hi, My name is Jane Doe. My passport number is PE1210432. I lost my passport when I was walking at the park. What should I do?
Preprocessed Query: "I lost my passport, what should I do?"

# Example 2
Original Query: Hi. I want to ask what are the requirements for applying for a passport? Thank you very much.
Preprocessed Query: "What are the requirements for applying for a passport?"

# Example 3
Original Query: Hi, My name is Jane Doe. Nawala ang aking passport. Ano ang kailangan kong gawin?
Preprocessed Query: "I lost my passport, what should I do?"

# Example 4
Original Query: What should I eat today?
Preprocessed Query: "No relevant question"                                     

Do the same for this last query. Preprocess the following query, return plain text only:
(FINAL Example) Original Query: ```{email_body}```
(NO CODE) Preprocessed Query:""")

rag_prompt = PromptTemplate.from_template("""
You are a polite assistant for the Department of Foreign Affairs.
                                              
I need you to address the question based ONLY on the possibly relevant context was retrieved here: 
[Start of Context FAQs]{retrieved_docs}[End of Context FAQs]
                                              
QUESTION: `{extracted_query}` 
                                              
### Other Instructions: 
1. Answer must strictly be based on the context only. Quote directly from the context. 
2. Answer ony what was asked.
3. Just say 'I cannot answer your question.' if no answer from context. 
4. Please return strictly as plain text and not markdown format. 
5. The 'context' is internal, do not mention its existense in the answer, give an answer as if you are the source of information. 
6. Please give a detailed answer. Provide instructions or links that exist in the context if needed. 

Answer:
""")

keyword_prompt = PromptTemplate.from_template("""You have only one task. List down all the keywords a list from the following question: `{extracted_query}`
Return a list containing at most 3 items.
Example answer: ["Passport", "Lost", "Missing"]
Answer: """)

email_format_prompt = PromptTemplate.from_template("""You are an assistant for the Department of Foreign Affairs.
A citizen emailed the following: ```{email_body}```
The detected query is: ```{extracted_query}```
The AI-generated answer we generated is: ```{generated_answer}```
Construct the email for sending. This will be automatically sent so add a disclaimer saying that this is AI-generated, they could reply if unsatisfied or for further clarification.
AI-Generated Autoreply: 
""")

prompt = PromptTemplate.from_template("""You are a translation assistant. I will only give you one translation task, you will give the answer only and nothing else, and here it is:
                                      Translate to english: ```{email_body}```:""")
prompt2 = PromptTemplate.from_template("""Answer in only one word, either "Yes" or "No". I will only give two questions only.   
                                       Is this question respectful and what you would expect a government agency to receive?: 
                                       1/ ```Where is the DFA Office located```. Answer: Yes 
                                       2/ ```{extracted_query}```. Answer: """)
prompt3 = PromptTemplate.from_template("""You only know about the following: ```{retrieved_docs}```, Answer the question by quoting directly from your knowledge: {extracted_query}. Answer ony what was asked. Answer: """)
prompt4 = PromptTemplate.from_template("""I will give you a question, context, and answer generated by a RAG application. 
                                       Question: ```{extracted_query}````
                                       Context: ```{retrieved_docs}```
                                       Generated-Answer: ```{generated_answer}```
                                       Answer in only one word, either "Yes" or "No" (No follow up sentence).
                                       Does the answer satisfactorily answer the question consisten with the context provided? If the generated answer was a refusal, that is a "No".
                                       Satisfactory: 
                                       """)

repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"  
# repo_id = "microsoft/Phi-3-mini-4k-instruct"

llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.01)

# output_dict = RunnablePassthrough.assign(translated=prompt | llm).invoke(temp_dict)

if email_body:
    output_dict = RunnablePassthrough.assign(extracted_query=qe_prompt | llm | StrOutputParser() | RunnableLambda(lambda x: x.strip())).invoke(output_dict)
    output_dict = RunnablePassthrough.assign(in_scope=prompt2 | llm | StrOutputParser() | RunnableLambda(lambda x: x.strip().split()[0])).invoke(output_dict)
    if 'yes' in output_dict['in_scope'].lower():
        output_dict['in_scope'] = True
        output_dict = RunnablePassthrough.assign(keywords=keyword_prompt | llm | StrOutputParser() | RunnableLambda(lambda x: eval(x.strip("```python")))).invoke(output_dict)
        output_dict = RunnablePassthrough.assign(retrieved_docs=RunnableLambda(lambda x: x['extracted_query']) | retriever).invoke(output_dict)
        output_dict = RunnablePassthrough.assign(generated_answer=rag_prompt | llm | StrOutputParser() | RunnableLambda(lambda x: x.strip())).invoke(output_dict)
        output_dict = RunnablePassthrough.assign(satisfactory_answer=prompt4 | llm | StrOutputParser() | RunnableLambda(lambda x: x.strip().split()[0])).invoke(output_dict)
        if 'yes' in output_dict['satisfactory_answer'].lower():
            output_dict['satisfactory_answer'] = True
            output_dict = RunnablePassthrough.assign(email_autoreply=email_format_prompt | llm | StrOutputParser() | RunnableLambda(lambda x: x.strip())).invoke(output_dict)
        else:
            output_dict['satisfactory_answer'] = False
            output_dict['flagged'] = True
    else:
        output_dict['in_scope'] = False
        output_dict['flagged'] = True
    
    if output_dict['flagged']:
        output_dict['email_autoreply'] = """Thank you for contacting us. Your email has been received and flagged for a manual response by one of our agents, as it requires assistance beyond our FAQs. Please stand by, and we will get back to you as soon as possible."""
    
    # output_dict['flagged'] = ('no' in output_dict['in_scope'].lower()) | ('no' in output_dict['satisfactory_answer'].lower())
    # output_dict = RunnablePassthrough.assign(translated2=prompt | llm).invoke(output_dict)
    
    # First box: AI-Generated Autoreply
    st.markdown(
        f"""
        <div style="
            border: 1px solid rgba(255, 255, 255, 0.2); 
            padding: 10px; 
            border-radius: 5px; 
            background-color: rgba(255, 255, 255, 0.1); 
            color: inherit; 
            margin-bottom: 20px;">
            <strong>AI-Generated Autoreply:</strong><br>
            {output_dict['email_autoreply']}
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    st.write(
        "\n\nOutput Dictionary:", 
        output_dict
    )
