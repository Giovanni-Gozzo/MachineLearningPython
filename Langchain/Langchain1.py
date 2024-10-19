from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(
    model="davinci",
    temperature=0,

)
response=llm.invoke("Hello, how are you?")
print(response)

#Invoke permet d'envoyer une requête à l'API OpenAI et de recevoir une réponse.

#Batch permet d'envoyer plusieurs requêtes à l'API OpenAI et de recevoir plusieurs réponses.
#On les recoit sous forme de liste.



