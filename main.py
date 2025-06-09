from langchain_core.messages import HumanMessage #langchain é o framework py que trabalha com IAs
from langchain_community.chat_models import ChatOllama #framework importando o modelo da nossa IA
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv #importar dotenv pra não expor chave da api da OpenAI

load_dotenv()

def main():
    model = ChatOllama(model='mistral')#definindo uma variável para qual será seu modelo de IA, que no caso será o modelo de Chat da OpenAI. Temperature 0 significa que as respostas serão frias e diretas, sem muito espaço pra criatividade.

    tools=[]
    agent_executor = create_react_agent(model, tools) #basicamente essa é a variável que será a nossa "pessoa" que responderá nossos comandos. Ela tem dois parametros passados = o modelo que ela se baseará de IA e as ferramentas que queremos que ela use.

    while True: #o chat irá continuar funcionando até o usuário não querer mais, por isso o while True.

        msg_usuario = input('\nVocê: ').strip() # .strip só pra remover espaços antes e depois da mensagem do usuário.

        if msg_usuario == 'sair':
            break
            
        print('\nMarIA: ', end='') # end='' pra msg que a IA responder não pular pra linha de baixo

        for chunk in agent_executor.stream(
            {'messages': [HumanMessage(content=msg_usuario)]} # for chunk = pra cada "chunk" da resposta (cada "parte"), que o agent_executor der, a pergunta ou 'message' é pra ser interpretada como vinda de uma pessoa (HumanMessage) e o conteúdo é a msg_usuario. O .stream em agent_executor é pra ele me dar a resposta conforme ele for analisando, poderia ser o .invoke que ele faria toda a resposta e só me enviaria ela depois.
        ):
            if "agent" in chunk and "messages" in chunk['agent']: #burocracia
                for message in chunk['agent']['messages']: #pra cada mensagem - palavra que ele usou pra responder meu input - da resposta, printe a palavra (print(message.content, end=''))
                    print(message.content, end='')

if __name__ == '__main__':
    main()