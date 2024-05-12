from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import BaseConversationalRetrievalChain
from langchain_community.llms import OpenAI
from langchain.agents import AgentExecutor, load_tools, ZeroShotAgent

from dotenv import load_dotenv
from PdfKnowledge import *

load_dotenv()
def conversational_chat_with_memory(qa: BaseConversationalRetrievalChain, question: str, memory: ConversationBufferMemory) -> str:
    response_text = qa(question)
    print(f'{memory}')
    print(f'{response_text["answer"]}')
    return response_text["answer"]

def conversational_chat_with_react(question: str, memory: ConversationBufferMemory) -> str:

    prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
    suffix = """Begin!"


    Question: {input}
    Thought: {agent_scratchpad}

    Answer in Bahasa Indonesia
    """

    llm = OpenAI(temperature=0)
    tools = load_tools(["llm-math"], llm=llm)
    tools.append(PdfKnowledge())

    prompt = ZeroShotAgent.create_prompt(
        tools=tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"]
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors=True
    )

    result = agent_chain.run(input=question)



    print(result)

    return result
