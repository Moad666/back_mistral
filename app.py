from flask import Flask, request, jsonify
import requests
import json
from flask_cors import CORS

#used for building and training deep learning models
import torch
#library is specifically designed for natural language processing (NLP) tasks, such as generating sentence embeddings.
from sentence_transformers import SentenceTransformer

# to compute the similarity between text documents
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# library is used for natural language processing (NLP) tasks.
import spacy

#library is used for numerical computations
import numpy as np
#used for working with regular expressions
import re



app = Flask(__name__)
CORS(app)

OLLAMA_MISTRAL_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"

def transform_text(input_text):
    # Replace '\n' with a space
    transformed_text = input_text.replace('\n', ' ')
    # Replace '\' with a space
    transformed_text = transformed_text.replace('\\', '')
    # Replace '  ' with a single space
    #transformed_text = ' '.join(transformed_text.split())
    # Split the text into words, remove any empty strings, and join them back together
    #transformed_text = ' '.join(filter(None, transformed_text.split()))
    
    # Use regular expression to replace any occurrences of words with spaces with a single word
    transformed_text = re.sub(r'([A-Za-z]+)\s([A-Za-z]+)', r'\1\2', transformed_text)
    return transformed_text

@app.route('/generate', methods=['POST'])
def generate_text():
    try:
        data = request.json
        prompt = data.get('prompt')

        if not prompt:
            return jsonify({'error': 'Missing prompt'}), 400

        payload = {
            'model': MODEL_NAME,
            'prompt': prompt
        }

        response = requests.post(OLLAMA_MISTRAL_API_URL, json=payload, stream=True)

        if response.status_code != 200:
            return jsonify({'error': f'API request failed with status code {response.status_code}'}), 500

        # Combine all responses into a single string
        combined_response = ''
        for line in response.iter_lines():
            if line:
                response_data = json.loads(line)
                combined_response += response_data['response'] + ' '
        transformed_response = transform_text(combined_response)
        print('after transformed : ' + transformed_response)
        return jsonify({'response': transformed_response.strip()}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

'''
#loads a pre-trained spaCy model, which is trained on web text and includes word vectors.
nlp = spacy.load("en_core_web_md")

@app.route('/analyze_context', methods=['POST'])
def analyze_context():
    try:
        data = request.json

        # Retrieve text1 and text2 from the request body
        text1 = data.get('text1')
        text2 = data.get('text2')

        if not text1 or not text2:
            return jsonify({'error': 'Missing text1 or text2 in the request body'}), 400

        #Tokenization and Preprocessing 
        #Tokenization is the process of breaking down a text into smaller units, called tokens. These tokens are typically words, punctuation marks, or other meaningful elements of the text. 
        doc1 = nlp(text1)
        doc2 = nlp(text2)

        # Compute document embeddings
        # These vectors capture the semantic meaning or context of the document
        doc1_embedding = doc1.vector
        doc2_embedding = doc2.vector

        # Compute similarity
        similarity_score = calculate_similarity(doc1_embedding, doc2_embedding)

        # Define a threshold for similarity
        # If the computed similarity score is greater than or equal to this threshold, it's considered that the texts are similar
        similarity_threshold = 0.85

        if similarity_score >= similarity_threshold:
            similarity_response = "The texts are similar."
        else:
            similarity_response = "The texts are not similar."

        return jsonify({'similarity_response': similarity_response, 'similarity_score': float(similarity_score)}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def calculate_similarity(embedding1, embedding2):
    # Compute cosine similarity between embeddings
    # is a mathematical operation that takes two equal-length sequences of numbers (vectors) and returns a single number. 
    similarity_score = embedding1.dot(embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return similarity_score'''


# Load a pre-trained transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

@app.route('/analyze_context', methods=['POST'])
def analyze_context():
    try:
        data = request.json

        # Retrieve text1 and text2 from the request body
        text1 = data.get('text1')
        text2 = data.get('text2')

        if not text1 or not text2:
            return jsonify({'error': 'Missing text1 or text2 in the request body'}), 400

        # Compute document embeddings
        doc1_embedding = model.encode(text1, convert_to_tensor=True)
        doc2_embedding = model.encode(text2, convert_to_tensor=True)

        # Compute cosine similarity between embeddings
        similarity_score = calculate_similarity(doc1_embedding, doc2_embedding)

        # Define a threshold for similarity
        similarity_threshold = 0.8

        if similarity_score >= similarity_threshold:
            similarity_response = "The texts are similar."
        else:
            similarity_response = "The texts are not similar."

        return jsonify({'similarity_response': similarity_response, 'similarity_score': float(similarity_score)}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def calculate_similarity(embedding1, embedding2):
    # Compute cosine similarity between embeddings
    similarity_score = torch.nn.functional.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0)).item()
    return similarity_score



if __name__ == '__main__':
    app.run(debug=True)
