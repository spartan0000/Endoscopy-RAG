#all the functions we need to run in main.py

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


load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


openai_client = OpenAI(api_key = OPENAI_API_KEY)

chroma_client = chromadb.PersistentClient(
    path = './chroma_db'
)

protocol_collection = chroma_client.get_or_create_collection(name = 'endoscopy_protocol')

###Things to consider for future improvements:
### 1. Add a function to summarize user query to make it more concise for embedding generation
### 2. Add a function to do semantic chunking of documents instead of just fixed size chunking




########################################################################################################################################################
####IMPORTANT####IMPORTANT####IMPORTANT####IMPORTANT####IMPORTANT####IMPORTANT####IMPORTANT####IMPORTANT####IMPORTANT####IMPORTANT####
#this is a prototyping project so while the text below would be confidential and protected health information, when prototyping, we will use fake data
#once we are ready to progress to testing on real patient data, we will have appropriate approvals in place and will use a locally hosted model so that
#patient data never leaves the hospital network
#########################################################################################################################################################



def format_query_json(user_query: str) -> dict: 
    system_prompt = """
    summarize the user input that includes medical data on a person's history of colonoscopy procedures and the pathology reports from the polyps that were removed.
    key information includes the following:
    - dates of the procedure including month, day, and year.
    - the number of polyps
    - the size of the polyps which is generally reported in millimeters - less than 10mm or greater than or equal to 10mm is a useful cutoff
    - the histology of the polyps.  
    - for adenomas - the number of polyps and the size of the largest adenoma, whether there is high grade dysplasia (yes/no), whether the adenoma is tubulovillous or villous (yes/no)
    - for sessile serrated polyps - the number of sessile serrated polyps, the size of the largest sessile serrated polyp, whether there is dysplasia (yes/no)
    - for hyperplastic polyps - size greater than or equal to 10mm (yes/no)
    format the output as ***JSON output*** with the following schema for each colonoscopy procedure.  

    {'patient_name': '',
    'patient_NHI': '',
    
        'colonoscopy': [
                            {'date': '', 
                            'number of polyps': 0, 
                            },
        'histology': {
            {'adenoma': number of adenomas,
            'adenoma_size': 'largest adenoma size in mm',
            'high_grade_dysplasia_in the adenoma': 'yes' or 'no',
            'tubulovillous_or_villous_adenoma': 'yes' or 'no',
            'sessile_serrated_polyps': number of sessile serrated polyps,
            'sessile_serrated_polyp_size': 'size of largest sessile serrated polyp',
            'dysplasia_in_the_sessile_serrated_polyp': 'yes' or 'no',
            'hyperplastic_polyp_greater_or_equal_to_10mm_in_size: 'yes' or 'no',
        }
    }
    ]
}
    make sure the JSON is properly formatted and can be parsed by a standard JSON parser.
"""


    user_prompt = f'Please format this medical text into structured JSON output - {user_query}'

    response1 = openai_client.responses.create(
        model = 'gpt-4o-mini',
        text = {'format': {'type': 'json_object'}},
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
        temperature = 0.8

    )

    try:
        raw_output = response1.output_text
        result_json = json.loads(raw_output)
        return result_json
    except json.JSONDecodeError:
        return {'error': 'Failed to parse JSON', 'raw_output': response1.output_text}

    
    

def format_query_summary(user_query: str) -> str:
    system_prompt = """
    you are a helpful medical assistant who is tasked with providing a detailed summary of only the pertinent details of the user input data that references recent colonoscopy procedures,
    the details from the procedure notes themselves, as well as the histological report from any polyps that were removed during that procedure.  Pertinent information that must be included
    in the summary are the number of polyps, the types of polyps (such as adenoma and whether the adenoma is tubulovillous or villous) , sessile serrated polyps, hyperplastic polyps as well as their sizes. 
    Regarding the procedure details, the significant findings include the BBPS score and where the scope was advanced to.  Regarding the polyps that are noted, please summarize and reconcile the information
    on the polyps that is contained within the procedure note as well as the histology report.  DO NOT make any clinical diagnoses or recommendations.
    """

    response = openai_client.responses.create(
        model = 'gpt-4o-mini',
        input = [
            {'role': 'system',
             'content': system_prompt,
             },
             {'role': 'user',
              'content': user_query}
              
        ],
        temperature = 0.9
    )

    return response.output_text
    
    
    
          
    
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
        'review by surgeon'. If there is uncertainty between two possible time intervals, choose the shorter one - for example if the choice is between 6 months or 12 months, choose 6 months.
        Provide a detailed explanation for your recommendation based on the protocols and data provided.
        """
    
    db_docs = ''.join(db_results[i]['document'] for i in range(len(db_results)))

    response = openai_client.responses.create(
        model = 'gpt-4o-mini',

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
     
