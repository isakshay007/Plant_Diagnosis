import streamlit as st
from lyzr_automata.ai_models.openai import OpenAIModel
from lyzr_automata import Agent, Task
from lyzr_automata.pipelines.linear_sync_pipeline import LinearSyncPipeline
from PIL import Image
from lyzr_automata.tasks.task_literals import InputType, OutputType
import os

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["apikey"]

st.markdown(
    """
    <style>
    .app-header { visibility: hidden; }
    .css-18e3th9 { padding-top: 0; padding-bottom: 0; }
    .css-1d391kg { padding-top: 1rem; padding-right: 1rem; padding-bottom: 1rem; padding-left: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

image = Image.open("./logo/lyzr-logo.png")
st.image(image, width=150)

# App title and introduction
st.title("Plant Diagnosis Assistant")
st.markdown("Welcome to your Plant Health Companion, your go-to resource for diagnosing and remedying plant issues powered by Lyzr Automata. What can I assist you with today? ")
input = st.text_input("Please enter your problems or concerns",placeholder=f"""Type here""")

open_ai_text_completion_model = OpenAIModel(
    api_key=st.secrets["apikey"],
    parameters={
        "model": "gpt-4-turbo-preview",
        "temperature": 0.2,
        "max_tokens": 1500,
    },
)


def generation(input):
    generator_agent = Agent(
        role=" Expert PLANT DIAGNOSIS ASSISTANT ",
        prompt_persona=f"Your task is to CAREFULLY ANALYZE the user's input regarding their plant's type, visible symptoms, and any other relevant details they provide. You MUST use your EXPERTISE to pinpoint possible issues and offer GUIDANCE on both IMMEDIATE and LONG-TERM care for their plant.")
    prompt = f"""
You are an Expert PLANT DIAGNOSIS ASSISTANT. Your task is to CAREFULLY ANALYZE the user's input regarding their plant's type, visible symptoms, and any other relevant details they provide. You MUST use your EXPERTISE to pinpoint possible issues and offer GUIDANCE on both IMMEDIATE and LONG-TERM care for their plant.

Here's how you should approach the task:

1. Based on the information provided by the user, IDENTIFY 2-3 POTENTIAL PROBLEMS that could be affecting the health of the plant.

2. For each identified problem, DELIVER a comprehensive ISSUE DESCRIPTION so that the user can UNDERSTAND what might be wrong with their plant.

3. PROVIDE CLEAR INSTRUCTIONS on IMMEDIATE ACTIONS that the user should take to mitigate harm and begin recovering their plant’s health.

4. ADVISE on LONG-TERM CARE strategies tailored specifically for the type of plant in question to help PREVENT future issues similar to those identified.

Your step-by-step guidance is crucial for users looking to nurse their plants back to health or maintain their well-being.

 """

    generator_agent_task = Task(
        name="Generation",
        model=open_ai_text_completion_model,
        agent=generator_agent,
        instructions=prompt,
        default_input=input,
        output_type=OutputType.TEXT,
        input_type=InputType.TEXT,
    ).execute()

    return generator_agent_task 
   
if st.button("Diagnose"):
    solution = generation(input)
    st.markdown(solution)

with st.expander("ℹ️ - About this App"):
    st.markdown("""
    This app uses Lyzr Automata Agent . For any inquiries or issues, please contact Lyzr.

    """)
    st.link_button("Lyzr", url='https://www.lyzr.ai/', use_container_width=True)
    st.link_button("Book a Demo", url='https://www.lyzr.ai/book-demo/', use_container_width=True)
    st.link_button("Discord", url='https://discord.gg/nm7zSyEFA2', use_container_width=True)
    st.link_button("Slack",
                   url='https://join.slack.com/t/genaiforenterprise/shared_invite/zt-2a7fr38f7-_QDOY1W1WSlSiYNAEncLGw',
                   use_container_width=True)