from dataclasses import dataclass
from langchain.vectorstores.chroma import  Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import BaseConversationalRetrievalChain

CHROMA_PATH = "chroma"

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def qa_memory() -> any:


    custom_template = """
        Answer the question based only on the following context:

        {context}

        ---

        Answer the question based on the above context: {question}
        Answer like a nice bot please
        """

    sys_prompt = PromptTemplate(
        template=custom_template,
        input_variables=["question"]
    )
    # LLM
    llm = ChatOpenAI()



    db = Chroma(persist_directory=CHROMA_PATH,
                embedding_function=OpenAIEmbeddings()
                )

    retriever = db.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
    qa.combine_docs_chain.llm_chain.prompt.messages[0] = SystemMessagePromptTemplate.from_template(custom_template)

    return qa, memory


def clear_memory() -> any:
    memory.clear()