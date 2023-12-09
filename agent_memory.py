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
from langchain.agents import tool, AgentExecutor
from langchain.tools import Tool
from langchain.chains import LLMMathChain

from dotenv import load_dotenv
from PdfKnowledge import *

def get_agent_memory() -> any:
    template = """
            Answer the following questions as best you can. 

            You have access to the following tools:

            {tools}
            Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}] 
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question in Bahasa Indonesia

            Begin!

            Question: {input}
            Thought:{agent_scratchpad}

            """

    llm = OpenAI(temperature=0)
    llm_math_chain = LLMMathChain(llm=llm, verbose=True)

    tools = [PdfKnowledge()]

    # tools.append(
    #     Tool(
    #         name="Calculator",
    #         func=llm_math_chain.run,
    #         description="useful for when you need to answer questions about math",
    #     ),
    # )

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    llm_with_stop = llm.bind(stop=["\nObservation"])

    agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
            }
            | prompt
            | llm_with_stop
            | ReActSingleInputOutputParser()
    )

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)