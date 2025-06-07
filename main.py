from langchain_core.messages import HumanMessage #langchain é o framework py que trabalha com IAs
from langchain_openai import ChatOpenAI #framework importando openAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv #importar dotenv pra não expor chave da api da OpenAI

load_dotenv()

def main():
    model = ChatOpenAI(temperature=0)#definindo uma variável para qual será seu modelo de IA, que no caso será o modelo de Chat da OpenAI. Temperature 0 significa que as respostas serão frias e diretas, sem muito espaço pra criatividade.

    tools=[]
    agent_executor = create_react_agent(model, tools) #basicamente essa é a variável que será a nossa "pessoa" que responderá nossos comandos. Ela tem dois parametros passados = o modelo que ela se baseará de IA e as ferramentas que queremos que ela use.