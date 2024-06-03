import streamlit as st
import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv, find_dotenv

# Getting API key from .env file
_ = load_dotenv(find_dotenv())

# client = OpenAI(
#     # This is the default and can be omitted
#     api_key=os.environ.get("OPENAI_API_KEY"),
# )

# OpenAI llm model to be used
llm_model = "gpt-3.5-turbo"

st.title("Practice Chatbot!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def getResponse(query, chat_history):
    template_string = """
    You are a helpful assistant. Answer the following queries to the best of your ability and your answers should be concise. \
    If you do not know the answer, just say that you do not know. Do not make up an answer.

    Current chat history: {chat_history}
    
    Here is a query to answer: {user_query}
    """
    
    prompt_template = ChatPromptTemplate.from_template(template_string)
    llm = ChatOpenAI(temperature=0.0, model=llm_model, api_key=os.environ.get('OPEN_API_KEY'))
    chain = prompt_template | llm | StrOutputParser()

    return chain.invoke({"chat_history": chat_history, "user_query": query})

#conversation UI
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)


#user input
user_query = st.chat_input("Your messsage")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        ai_response = getResponse(user_query, st.session_state.chat_history)
        st.markdown(ai_response)

    st.session_state.chat_history.append(AIMessage(ai_response))

import requests
from langchain import wikipedia
from pydantic import BaseModel, Field

@tool
def search_wikipedia(query: str) -> str:
    """Run Wikipedia search and get page summaries."""
    page_titles = wikipedia.search(query)
    summaries = []
    for page_title in page_titles[: 3]:
        try:
            wiki_page =  wikipedia.page(title=page_title, auto_suggest=False)
            summaries.append(f"Page: {page_title}\nSummary: {wiki_page.summary}")
        except (
            self.wiki_client.exceptions.PageError,
            self.wiki_client.exceptions.DisambiguationError,
        ):
            pass
    if not summaries:
        return "No good Wikipedia Search Result was found"
    return "\n\n".join(summaries)

class DogBreedInput(BaseModel):
    name: str = Field(..., description="Name of the dog breed to send to the api")

@tool
def get_rand_dog_img() -> str:
    """Get a random dog image."""

    URL = "https://dog.ceo/api/breeds/image/random"

    response = requests.get(URL)
    
    if response.status_code == 200:
        results = response.json()
    else:
        raise Exception(f"API Request failed with status code: {response.status_code}")
    
    return results.message

tools = [search_wikipedia, get_rand_dog_img]

