#run the document processing here


import openai
from openai import OpenAI
import chromadb
import os
from chromadb.config import Settings
from typing import List
import json
from dotenv import load_dotenv


from langchain.text_splitter import RecursiveCharacterTextSplitter


from functions import document_chunker, cache_protocol


filenames = os.listdir('documents')

def process_documents(filenames: List[str]): #this should do take care of all the documents in the folder
    for filename in filenames:
        with open(os.path.join('documents', filename), 'r', encoding = 'utf-8') as f:
            content = f.read()
            chunks = document_chunker(content, 800, 100)
            cache_protocol(chunks, filename.split('.')[0])