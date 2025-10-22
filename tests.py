#tests

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

import unittest

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


from functions import format_query_json, format_query_summary, get_embedding, document_chunker, cache_protocol, query_protocol_collection, generate_recommendation

chroma_client = chromadb.PersistentClient(path = './chroma_db')
protocol_collection = chroma_client.get_or_create_collection(name = 'endoscopy_protocol')

class TestEndoscopyRAG(unittest.TestCase):

    def test_embedding(self):
        test_text = 'This is test text'
        embedding = get_embedding(test_text)
        self.assertIsInstance(embedding, list)
        self.assertGreater(len(embedding), 0)

    def test_format_query_json(self):
        user_query = 'Patient Bob Thebuilder whose NHI is ABC1234 had a colonoscopy on 23-1-2020 with 2 polyps, one was a 5mm adenoma and the other a 12mm sessile serrated polyp.'
        formatted_json = format_query_json(user_query)
        self.assertIsInstance(formatted_json, dict)
        self.assertIn('patient_name', formatted_json)
        self.assertIn('colonoscopy', formatted_json)
    
    def test_document_chunker(self):
        test_text = 'This is a long text. ' * 100
        chunks = document_chunker(text = test_text, chunk_size = 50, chunk_overlap = 10)
        self.assertIsInstance(chunks, list)
        self.assertGreater(len(chunks), 1)
        self.assertLessEqual(len(chunks[0]), 50)

    def test_query_protocol_collection(self):
        test_embedding = get_embedding('Test query text')
        results = query_protocol_collection(test_embedding, protocol_collection, n_results = 5)
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 5)
    
    
    def test_format_query_summary(self): 
        user_query = 'Patient Bob Thebuilder whose NHI is ABC1234 had a colonoscopy on 22-1-2022 and had 8 polyps removed including a 15mm tubulovillous adenoma with high grade dysplasia' 
        summary = format_query_summary(user_query)
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)
          

        

if __name__ == '__main__':
    unittest.main()
