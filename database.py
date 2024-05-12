from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader

import os
import shutil
from dotenv import load_dotenv
from minio_util import delete_data, get_file

CHROMA_PATH = "chroma"
DATA_PATH = "data/books"

load_dotenv()

def renew():
    delete_data()
    get_file()
    generate_data_store()
def main():
    generate_data_store()
def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    pdf_folder_path = DATA_PATH
    documents = []

    for file in os.listdir(pdf_folder_path):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder_path, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    return documents

def split_text(documents: list[Document]):
    text_splitter = CharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=1000
    )
    chunks = text_splitter.split_documents(documents)
    print(f'Split {len(documents)} documents into {len(chunks)} chunks.')


    return chunks

def save_to_chroma(chunks: list[Document]):
    # if os.path.exists(CHROMA_PATH):
    #     shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(),
        persist_directory=CHROMA_PATH
    )

    db.persist()
    print(f'Saved {len(chunks)} chunks to {CHROMA_PATH}')

if __name__ == "__main__":
    main()