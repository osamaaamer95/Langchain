from enums import Model

from ZeroToHero.SimpleAPICall import generate_openai_response

# ######
# # Call a GPT model with a temperature and a prompt
# ######

print(generate_openai_response(Model.GPT3, 0.9, "Hello world?"))
