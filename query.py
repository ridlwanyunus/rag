import argparse
from dataclasses import dataclass
from langchain.vectorstores.chroma import  Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import BaseConversationalRetrievalChain
from langchain.tools.render import  render_text_description
from langchain.llms import OpenAI
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents import tool, AgentExecutor, load_tools, ZeroShotAgent
from langchain.tools import Tool
from langchain.chains import LLMMathChain

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

    {chat_history}
    Question: {input}
    {agent_scratchpad}
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
        agent=agent, tools=tools, verbose=True, memory=memory
    )

    result = agent_chain.run(input=question)



    print(result)

    return result
