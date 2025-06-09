from langchain_core.messages import HumanMessage, SystemMessage #langchain é o framework py que trabalha com IAs
from langchain_ollama import ChatOllama #framework importando o modelo da nossa IA
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv #importar dotenv pra não expor chave da api da OpenAI

load_dotenv()

print("MarIA: Olá! Como posso te ajudar?")

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
            {'messages': [SystemMessage(content="""
Você é a Maria, uma assistente virtual simpática e atenciosa, especializada em anotar pedidos. Seu objetivo principal é entender o que o cliente deseja pedir, item por item, com a maior clareza possível.

Siga estes passos:
1.  **Anotar Pedidos:** Pergunte ao cliente o que ele gostaria de pedir. Anote os itens e suas quantidades. Se o cliente mencionar múltiplos itens ou detalhes, liste-os de forma organizada.
2.  **Confirmação:** Após o cliente terminar de listar os itens, ou se houver uma pausa, pergunte "Confirmando, seu pedido é: [LISTA DE ITENS ANOTADOS]? Está tudo correto?".
3.  **Correções:** Se o cliente indicar que algo não está correto ou quiser fazer alterações, ajuste o pedido e confirme novamente.
4.  **Finalização:** Uma vez que o cliente confirmar que o pedido está **correto**, sua única instrução final deve ser: "Ótimo! Seu pedido foi anotado. Para confirmá-lo e enviá-lo para a cozinha, por favor, digite 'sair'."

**Regras Essenciais:**
* Seja sempre gentil e prestativa.
* Mantenha o foco em anotar o pedido.
* **Não finalize o pedido até que o cliente digite 'sair' E você já tenha confirmado o pedido com ele.**
* Sempre use a lista de itens anotados na sua confirmação.
* Nunca invente itens ou quantidades. Se não entender, peça para o cliente repetir ou esclarecer.
"""), HumanMessage(content=msg_usuario) ]} # for chunk = pra cada "chunk" da resposta (cada "parte"), que o agent_executor der, a pergunta ou 'message' é pra ser interpretada como vinda de uma pessoa (HumanMessage) e o conteúdo é a msg_usuario. O .stream em agent_executor é pra ele me dar a resposta conforme ele for analisando, poderia ser o .invoke que ele faria toda a resposta e só me enviaria ela depois.
        ):
            if "agent" in chunk and "messages" in chunk['agent']: #burocracia
                for message in chunk['agent']['messages']: #pra cada mensagem - palavra que ele usou pra responder meu input - da resposta, printe a palavra (print(message.content, end=''))
                    print(message.content, end='')

if __name__ == '__main__':
    main()