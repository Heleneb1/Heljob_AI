import streamlit as st
from openai import OpenAI

st.title("Heljob-Bot – Version Cloud")

# Initialisation du client OpenAI
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Prompt utilisateur
prompt = st.text_input("Pose ta question sur le CV, le Job ou les ATS :")

if prompt:
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # tu peux aussi tester gpt-4o ou gpt-5-mini
        messages=[
            {"role": "system", "content": "You are Heljob-Bot, a very smart assistant who knows everything about jobs, CVs, and ATS. Be succinct and direct."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1
    )
    st.write(response.choices[0].message["content"])
