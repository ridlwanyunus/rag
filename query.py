import argparse
from dataclasses import dataclass
from langchain.vectorstores.chroma import  Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import BaseConversationalRetrievalChain

from dotenv import load_dotenv

load_dotenv()
def conversational_chat_with_memory(qa: BaseConversationalRetrievalChain, question: str, memory: ConversationBufferMemory) -> str:
    response_text = qa(question)
    print(f'{memory}')
    print(f'{response_text["answer"]}')
    return response_text["answer"]