# en local

import ollama

modelfile = """
FROM llama3.2
SYSTEM You are a very smart assistant, your name is Heljob-Bot who knows everything about Job, ATS, CV writing. You are very succinct and direct.
PARAMETER temperature 0.1
"""

ollama.create(model="heljob-bot", modelfile=modelfile)

# # Test the model
# res = ollama.generate(model="heljob-bot", prompt="How to create an efficient CV and cover letter?")
# print(res['response'])

# run python file en haut à droite ▶️
# ollama list, ollama --help, supprimer un model ollama rm NomDuModel
# ollama run Heljob-Bot 
# sortir /bye

