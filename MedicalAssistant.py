import os
import streamlit as st
import openai
from dotenv import load_dotenv, find_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI as LangchainOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import Tool
from langchain.chat_models import ChatOpenAI

# Load environment variables
#Input your OpenAI API key in an .env
_ = load_dotenv(find_dotenv())
openai.api_key = os.environ['OPENAI_API_KEY']

# Set up Streamlit app
st.title("Medical LLM Assistant")
st.write("Note This is a LLM model and there's a chance the product will be wrong")
prompt = st.text_input('What Would you liked Explained or')

llm = LangchainOpenAI(temperature=0)

search = DuckDuckGoSearchRun()

# Defining use of the Google SerperAPI Wrapper tool
tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description="Useful for when you need to ask with search."
    )
]

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Set up ChatOpenAI model
chat = ChatOpenAI(temperature=0)


# Defining investment-related system messages
medical_system_messages = [
    SystemMessage(content="You are a highly experienced medical doctor that provides detailed medical explanations."),
    SystemMessage(content="It's important to do thorough research and consider professional advice."),
    SystemMessage(content="You provide general and detailed information about medical and pharmaceutical terms."),
    SystemMessage(content="You are a friendly medical doctor")

]

# Processing user input and display results
if st.button("Submit"):
    if prompt:
        user_input = prompt  # Directly use the user's input
        conversation = [
            *medical_system_messages,
            HumanMessage(content=user_input)
        ]
        result = chat(conversation)
        response_text = result.content
        st.write(response_text)
    else:
        st.write("Please enter a question.")
