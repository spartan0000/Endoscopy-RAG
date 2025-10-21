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

import logging
from datetime import datetime

from functions import format_query_json, get_embedding, document_chunker, cache_protocol, query_protocol_collection, generate_recommendation


load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


openai_client = OpenAI(api_key = OPENAI_API_KEY)

chroma_client = chromadb.PersistentClient(
    path = './chroma_db'
)

protocol_collection = chroma_client.get_or_create_collection(name = 'endoscopy_protocol')

logging.basicConfig(
    filename = 'logs/audit_logs.jsonl',
    level = logging.INFO,
    format = '%(message)s'
    )

def log_entry(entry: dict):
    logging.info(json.dumps(entry), ensure_ascii=False)



def main():
    #user input or the medical information to be processed
    user_query = """"""

    #format the user query into structured json
    formatted_query = format_query_json(user_query)
    
    #convert user query to an embedding
    query_embedding = get_embedding(user_query)

    #query the vector database for relevant protocols
    results = query_protocol_collection(query_embedding, protocol_collection, n_results = 10)

    #generate recommendation based on the retrieved protocols and the user query
    output = generate_recommendation(results, user_query)

    #log the interaction
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'user_query': user_query,
        'formatted_query': formatted_query,
        'database_results': results, #this will have metadatas that includes the source name and the chunk id
        'document_contents': [r['document'][:200] for r in results], #log a snippet of the documents contents for future reference
        'recommendation': output,

    }

    log_entry(log_data)

    print(output)

    return formatted_query #leave this here for now.  Later we can send this to a file or database to store and retrieve as needed.

if __name__ == "__main__":
    main()
