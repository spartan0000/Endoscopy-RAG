#all the functions we need to run in main.py

import openai
from openai import OpenAI
import chromadb
import os
from chromadb.config import Settings
from typing import List
import json
from dotenv import load_dotenv


from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


openai_client = OpenAI(api_key = OPENAI_API_KEY)

chroma_client = chromadb.PersistentClient(
    path = './chroma_db'
)

protocol_collection = chroma_client.get_or_create_collection(name = 'endoscopy_protocol')


def format_query_json(user_query: str) -> json: 
    system_prompt = """
    summarize the user input that includes medical data on a person's history of colonoscopy procedures and the pathology reports from the polyps that were removed.
    key information includes the dates of the procedure including month, day, and year.
    Other key information includes the number of polyps, the size of the polyps which is generally reported in millimeters, and the histology of the polyps.  The number, size and histology all needs
    to be carefully collated into ***JSON output*** with the following schema for each colonoscopy procedure.  The number of polyps, the size of the polyps and the histology can be obtained from the pathology report
    that is dated on or within 1-2 days of the procedure date.  The ***JSON schema*** is as follows:
    {'patient name': patient's name,
    'patient NHI': patient's National Health Identification number,
    
        'colonoscopy': {
                            {'date': date of procedure}, 
                            {'number of polyps': number of polyps}, 
                            {'polyp size': {'less than 10mm': number of polyps > 10mm}, {'>= 10mm': number of polyps >= 10mm}},
                            {'histology': {
                                {'adenoma': number of adenomas},
                                {'adenoma size': size of largest adenoma},
                                {'high grade dysplasia': yes or no},
                                {'tubulovillous or villous adenoma': yes or no},
                                {'sessile serrated polyps': number of sessile serrated polyps},
                                {'sessile serrated polyp size': size of largest sessile serrated polyp'},
                                {'dysplasia in the sessile serrated polyp': yes or no},
                                {'hyperplastic polyp greater or equal to 10mm in size: yes or no},
                            }
                        }
                    }
                }
        """

    user_prompt = f'This is the user prompt - {user_query}'

    response1 = openai_client.responses.create(
        model = 'gpt-4o-mini',
        response_format = {'type':'json_object'},
        input = [
            {
                'role':'system',
                'content': system_prompt,
            },
            {
                'role': 'user',
                'content': user_prompt,
            }
        ],
        temperature = 0.5

    )

    return response1.output_text

def format_query_summary(user_query: str) -> str:
    system_prompt = """
    you are a helpful medical assistant who is tasked with providing a detailed summary of the pertinent details of the user input data that references recent colonoscopy procedures,
    the details from the procedure notes themselves, as well as the histological report from any polyps that were removed during that procedure.  Pertinent information that must be included
    in the summary are the number of polyps, the types of polyps (such as adenoma (and whether the adenoma is tubulovillous or villous) , sessile serrated polyps, hyperplastic polyps as well as their sizes. 
    This information must be summarized in significant detail so that it can be used as input for an LLM to query a vector database of follow up protocols to make a determination on when the specific
    patient will next need a colonscopy based on published guidelines
    """

    #still working this out
    
    
    
    
          
    
def get_embedding(text: str) -> List[float]:
    response = openai_client.embeddings.create(
        input = text,
        model = 'text-embedding-3-small',
    )
    return response.data[0].embedding

def document_chunker(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        separators = ['\n\n', '\n', ' ', '.'],
    )

    return text_splitter.split_text(text)

def semantic_chunker():
    pass

    #come back and add this later - might work better for some documents



def cache_protocol(chunks, collection, source_name):
    documents = []
    embeddings = []
    ids = []
    metadatas = []
    
    for i, chunk in enumerate(chunks):
        try:
            embedding = get_embedding(chunk)
            if not embedding or not isinstance(embedding, list):
                print('Invalid embedding')
                continue
        except Exception as e:
            print('Invalid embedding')
            continue

        documents.append(chunk)
        embeddings.append(embedding)
        ids.append(f'Chunk_{i}')
        metadatas.append(
            {'source': source_name,
             'chunk index': i,
             'length': len(chunk),
             
            }
        )
                

    if documents:
        protocol_collection.add(
            documents = documents,
            embeddings = embeddings,
            ids = ids,
            metadatas = metadatas,
        )
        
        
def query_protocol_collection(query_embedding: List[float], collection, n_results: int) -> List[dict]:
    results = collection.query(query_embeddings = [query_embedding], n_results = n_results)

    if results['documents']:
        db_output = [
            {'document': doc,
                'metadatas': meta,
            }
            for doc, meta in zip(results['documents'][0], results['metadatas'][0])
        ]
    else:
        db_output = []
    
    
    return db_output

def generate_recommendation(db_results: List[dict], user_query: str) -> str:
    system_prompt = """You are a helpful medical assistant who is tasked with making evidence based recommendations for follow up after a colonoscopy
        You will take the user query which includes medical information as well as the protocol that is pulled from the vector database, provide recommendations for follow up that meet the guidelines
        as stated in the documents that are provided.  If it is unclear what the recommendations should be or if the patient's data does not meet any of the criteria in the documents, simply output 
        'review by surgeon'.  If there is uncertainty between two possible time intervals, choose the shorter one - for example if the choice is between 6 months or 12 months, choose 6 months"""
    
    db_docs = ''.join(db_results[i]['document'] for i in range(len(db_results)))

    response = openai_client.responses.create(
        model = 'gpt-40-mini',

        input = [
            {'role': 'system',
             'content': system_prompt
            },
            {'role': 'user',
              'content': user_query,
            },
            {'role': 'user',
               'content': db_docs,
            }
        ],
        temperature = 1.0,
    )

    return response.output_text
     
    