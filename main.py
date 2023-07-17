from enums import OpenAIModel

from ZeroToHero.SimpleAPICall import generate_openai_response

# ######
# # Call a GPT model with a temperature and a prompt
# ######

print(generate_openai_response(OpenAIModel.DAVINCI_3, 0.9, "Hello world?"))
