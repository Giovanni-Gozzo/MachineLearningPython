from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
llm = ChatOpenAI(
    temperature=0.7,
    model="gpt-3.5-turbo",

)

#Prompt template
prompt = ChatPromptTemplate.from_template("Tell me a joke about a {subject}")

# create LLM Chain
chain = prompt | llm

response= chain.invoke({"subject": "dog"})
print(response)

#Prompt template 2
prompt = ChatPromptTemplate.from_messages([
    ("system", "Tell me a recipe unique to your culture"),
    ("human", "{input}")]
)

chain = prompt | llm
response = chain.invoke({"input": "indian curry"})
print(response)


