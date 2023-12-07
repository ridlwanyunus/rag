import argparse
from dataclasses import dataclass
from langchain.vectorstores.chroma import  Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferMemory

from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def question(question) -> any:
    print(f'{question}')
    embedding_function = OpenAIEmbeddings()
    db = Chroma(persist_directory=CHROMA_PATH,
                embedding_function=embedding_function
                )

    results = db.similarity_search_with_relevance_scores(question, k=3)
    if len(results) == 0 or results[0][1] < 0.7:
        print(f'Unable to find matching results.')
        return

    context_text = "\n\n--\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=question)
    # print(prompt)

    model = ChatOpenAI()
    response_text = model.predict(prompt)

    return response_text

def conversational_chat(question) -> str:
    print(f'{question}')


    # LLM
    llm = ChatOpenAI()

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    db = Chroma(persist_directory=CHROMA_PATH,
                embedding_function=OpenAIEmbeddings()
            )

    retriever = db.as_retriever()
    qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

    response_text = qa(question)
    print(f'{response_text["answer"]}')
    return response_text["answer"]

