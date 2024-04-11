from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
import streamlit as st
from openai import OpenAI

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Initializing OpenAI client
API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=API_KEY)

# Initialize Chroma DB client
persist_directory = './embeddings/db/'
store = Chroma(persist_directory=persist_directory, collection_name="Capgemini_policy_embeddings")

# Initialize OpenAI Embeddings
embed_prompt = OpenAIEmbeddings()

def retrieve_vector_db(query, n_results=3):
    similar_embeddings = store.similarity_search_by_vector_with_relevance_scores(embedding=embed_prompt.embed_query(query), k=n_results)
    results = []
    prev_embedding = []
    for embedding in similar_embeddings:
        if embedding not in prev_embedding:
            results.append(embedding)
        prev_embedding = embedding
    return results

def question_to_response(query, temperature=0, max_tokens=200, top_n=10):
    retrieved_results = retrieve_vector_db(query, n_results=top_n)
    context = ''
    if len(retrieved_results) >= 2:
        context = ''.join(retrieved_results[0][0].page_content)
        context += ''.join(retrieved_results[1][0].page_content)

    prompt = f'''
    [INST]
    You are an expert in Capgemini policies. Generate response at least 400 tokens.

    Question: {query}

    Context : {context}
    [/INST]
    '''

    # Prepare messages for chat completion
    messages = [{"role": "system", "content": "You are an expert in Capgemini Policies."}]
    if st.session_state.messages:
        messages += [{"role": "user", "content": m["content"]} for m in st.session_state.messages]

    completion = client.chat.completions.create(
        temperature=temperature,
        max_tokens=max_tokens,
        model="ft:gpt-3.5-turbo-0125:personal:fine-tune-gpt3-5-1:9AFEVLdj",
        messages=messages
    )
    return completion.choices[0].message.content

# Set page configuration
st.set_page_config(layout="wide")
st.title("Emplochat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
user_input = st.text_input("Enter your query here?")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get response from the assistant
    response = question_to_response(user_input)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
