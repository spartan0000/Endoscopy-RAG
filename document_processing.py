#run the document processing here


import openai
from openai import OpenAI
import chromadb
import os
from chromadb.config import Settings
from typing import List
import json
from dotenv import load_dotenv

import langchain 
from langchain_text_splitters import RecursiveCharacterTextSplitter


from functions import document_chunker, cache_protocol
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


openai_client = OpenAI(api_key = OPENAI_API_KEY)

chroma_client = chromadb.PersistentClient(
    path = './chroma_db'
)

protocol_collection = chroma_client.get_or_create_collection(name = 'endoscopy_protocol')

filenames = os.listdir('documents')

def process_documents(filenames: List[str]): #this should do take care of all the documents in the folder
    for filename in filenames:
        with open(os.path.join('documents', filename), 'r', encoding = 'utf-8') as f:
            content = f.read()
            chunks = document_chunker(content, 800, 100)
            cache_protocol(chunks, protocol_collection, filename.split('.')[0])


if __name__ == '__main__':
    process_documents(filenames)
