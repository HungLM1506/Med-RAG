from utils.chatbot import * 

user_description = "There is a female patient has a BMI of 29 at the age of 27. What is Outcome ?"
response = LLM_MED().rag_chatbot(user_description)

print(response)