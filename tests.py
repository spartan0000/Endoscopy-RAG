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


from functions import format_query_json, get_embedding, document_chunker, cache_protocol, query_protocol_collection, generate_recommendation

class TestEndoscopyRAG(unittest.TestCase):

    def test_embedding(self):
        test_text = 'This is test text'
        embedding = get_embedding(test_text)
        self.assertIsInstance(embedding, list)
        self.assertGreater(len(embedding), 0)

    def test_format_query_json(self):
        user_query = 'Patient Bob Thebuilder had a colonoscopy on 23-1-2020 with 2 polyps, one was a 5mm adenoma and the other a 12mm sessile serrated polyp.'
        formatted_json = format_query_json(user_query)
        self.assertIsInstance(formatted_json, dict)
        self.assertIn('patient name', formatted_json)
        self.assertIn('colonoscopy', formatted_json)
        

if __name__ == '__main__':
    unittest.main()
