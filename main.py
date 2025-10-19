import openai
from openai import OpenAI
import chromadb
import os
from chromadb.config import Settings
from typing import List
import json
from dotenv import load_dotenv


from langchain.text_splitter import RecursiveCharacterTextSplitter


from functions import format_query_json, get_embedding, document_chunker, cache_protocol, query_protocol_collection, generate_recommendation


load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


openai_client = OpenAI(api_key = OPENAI_API_KEY)

chroma_client = chromadb.PersistentClient(
    path = './chroma_db'
)

protocol_collection = chroma_client.get_or_create_collection(name = 'endoscopy_protocol')


def main():
    user_query = """ """
    query_embedding = get_embedding(user_query)
    results = query_protocol_collection(query_embedding, n_results = 10)
    output = generate_recommendation(results, user_query)
    print(output)
    

if __name__ == "__main__":
    main()
