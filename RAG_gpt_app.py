from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
#import chromadb
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
import pprint
import os
import streamlit as st

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

####Enivironment settings for openai API key and Vector Embeddings############

API_KEY = st.secrets["OPENAI_API_KEY"]
from openai import OpenAI
client = OpenAI(api_key=API_KEY)
persist_directory = './embeddings/db/'
#"https://github.com/naren579/CAPSTONE_EMPLOBOT/tree/main/embeddings/db"

#Initialize the Chroma DB client
store = Chroma(persist_directory=persist_directory,collection_name="Capgemini_policy_embeddings")

# Get all embeddings
embeddings = store.get(include=['embeddings'])
embed_prompt = OpenAIEmbeddings()

###############################################################################


######################Getting Similar Vector Embeddings for a given prompt#####

def retrieve_vector_db(query, n_results=3):
    similar_embeddings = store.similarity_search_by_vector_with_relevance_scores(embedding = embed_prompt.embed_query(query),k=n_results)
    results=[]
    prev_embedding = []
    for embedding in similar_embeddings:
      if embedding not in prev_embedding:
        results.append(embedding)

      prev_embedding =  embedding
    return results

###############################################################################

############### Function to generate response for a given Prompt###############

def question_to_response(query,temperature=0,max_tokens=200,top_n=10):
    retrieved_results=retrieve_vector_db(query, n_results=top_n)
  #print(retrieved_results)
    if len(retrieved_results) < 1:
        context =''
    else:
        context = ''.join(retrieved_results[0][0].page_content)
        context=context+''.join(retrieved_results[1][0].page_content)
  #print(context)
    prompt = f'''
    [INST]
    You are an expert in Capgemini policies.Generate response atleast 400 tokens.

    Question: {query}

    Context : {context}
    [/INST]
    '''
    completion = client.chat.completions.create( temperature=temperature, max_tokens=max_tokens,
      model="ft:gpt-3.5-turbo-0125:personal:fine-tune-gpt3-5-1:9AFEVLdj",
      messages=[
        {"role": "system", "content": "You are an expert in capgemini Policies."},
        {"role": "user", "content": prompt}
      ]
    )
    return completion.choices[0].message.content
###############################################################################


#################Initialize session state to store history####################
st.set_page_config(layout="wide")
st.title("Emplochat")

# if 'history' not in st.session_state:
#     st.session_state.history = []
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "ft:gpt-3.5-turbo-0125:personal:fine-tune-gpt3-5-1:9AFEVLdj"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Enter your query here?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)   

with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[{"role": "system", "content": "You are an expert in Capgemini policies.Generate response atleast 400 tokens"+m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
st.session_state.messages.append({"role": "assistant", "content": response})
# # Display history
# # for pair in st.session_state.history:
# #     with st.container():
# #         with col1:
# #             st.text(f"Question: {pair['question']}")
# #         with col2:
# #             st.text(f"Answer: {pair['response']}")

# prompt = st.chat_input("Enter your question here:")



# if prompt:
#     # Backend Processing
#     response = question_to_response(prompt)  # Your function to process input and generate response

#     # Update history
#     #st.session_state.history.append({"question": user_input, "response": response})

#     #Display current response
#     st.write(f"Emplobot 🤖: {response}")

# # for pair in st.session_state.history:
# #     st.write(f"Question❓: {pair['question']}")
# #     st.write(f"Emplobot 🤖: {pair['response']}")

# ###############################################################################











