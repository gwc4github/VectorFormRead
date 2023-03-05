#!/usr/bin/env python
# coding: utf-8
import os

# <a href="https://colab.research.google.com/github/gwc4github/VectorFormRead/blob/main/VectorFormRead.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>




# This is the new version that can use either an S3 file OR a local file
###
# In the modified code, the get_file_contents() function checks if the input path starts with 's3://'. 
# If it does, the function reads the contents of the S3 object specified by the path. If not, the function assumes 
# the input path is a local file path and reads the contents of the file using Python's built-in open() function.
# In this program, the run_layoutlmft_ser() function takes a local or S3 path to a document as input and
# returns the predicted tags and all embeddings as a JSON object. 
# https://guillaumejaume.github.io/FUNSD/download/
# https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md
###

import boto3
import json
import torch
# from langchain.vectorstores import faiss
import faiss

from transformers import AutoTokenizer, AutoModelForTokenClassification

# Function to get S3 object or local file contents
def get_file_contents(path):
    if path.startswith('s3://'):
        # If the path starts with 's3://', assume it's an S3 object path
        obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=path[5:])
        retval = obj['Body'].read().decode('utf-8')
    else:
        # Otherwise, assume it's a local file path
        with open(path, 'r') as f:
            retval = f.read()
    return json.loads(retval)  # Just return the document's content json w/o metadata.

# Function to preprocess text
def preprocess_text(text):
    # Tokenize text and create input IDs
    encoded_input = tokenizer(text, return_tensors='pt')
    return encoded_input

# Function to run LayoutLMFT SER model and return results and embeddings
def run_layoutlmft_ser(path):
    # Load document from S3 or local file
    document_text = get_file_contents(path)['form']

    # Preprocess text and get input IDs
    input_ids = preprocess_text(document_text)

    # Run model and get results
    outputs = model(**input_ids)
    predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()
    predicted_tags = [model.config.id2label[tag] for tag in predictions]

    # Get all embeddings
    all_embeddings = outputs.last_hidden_state.tolist()

    # Return results and embeddings as JSON
    results = {'predicted_tags': predicted_tags, 'embeddings': all_embeddings}
    return json.dumps(results)


def get_question(question_id, p_form_data):
    for form_item in p_form_data:
        if form_item['id'] == question_id:
            return form_item
    return None  # We should only get here if there is an answer w/o a question


if __name__ == '__main__':
    # ! nvcc --version
    # !python --version
    print(torch.__version__)

    # Set up AWS credentials and S3 bucket information
    ACCESS_KEY = 'your_access_key'
    SECRET_KEY = 'your_secret_key'
    BUCKET_NAME = 'your_bucket_name'

    # Set up S3 client and resource
    s3_client = boto3.client('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)
    s3_resource = boto3.resource('s3', aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY)

    # Load LayoutLMFT SER model and tokenizer
    model_name = "microsoft/layoutlmv2-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)

    test_file = '/Users/greggwcasey/Google Drive/PycharmProjectsLocal/FUNSD_layoutlmv2/dataset/training_data/annotations/00040534.json'
    # read embeddings and load vector DB with them
    document_data = get_file_contents(test_file)
    form_data = document_data['form']
    i = 0 # TODO MAKE THIS A LOOP
    element_data = form_data[i]
    # tokenized_text = tokenizer('one')
    tokenized_text = tokenizer(['one','two','three'], boxes=[[84, 109, 136, 119],[84, 109, 136, 119],[84, 109, 136, 119]])

    tokenized_text = tokenizer(element_data['text'].split(), boxes=[word['box'] for word in element_data['words']])
    # NO: We want the upper left corner of both Q&A boxes.
    # NO: Then we want the relative X and Y differences.
    # YES: do we just want to load the q/A together as rows (not list links) so that the Q&A is in the same
    # row.  Then we care about tokens, box coordinates (all or just top-left?) Q is always first.
    import numpy as np
    # vector_db = np.array([])
    max_tokens: int = 30
    max_element_tokens: int = int(max_tokens/2)-10  # 10 is for the box tokens
    vector_db = []
    for element_data in form_data:
        if element_data['label'] == 'answer':  # ignore questions- we will put every answer with it's question.
            for link in element_data['linking']: # Ans could link to >1 Q
                question_id = link[0]  # for q is the first ID
                question_data = get_question(question_id, form_data)  # Get the row for the question
                # Create the vector DB row for this A/Q pair
                if question_data['text'] == '' or element_data['text'] == '':
                    continue  # if these are empty, we don't want to add them to the vector DB
                # vector_db_row = {
                #     'question_box': question_data['box'],
                #     'question_embeddings': tokenizer([word['text'] for word in question_data['words']], boxes=[word['box'] for word in question_data['words']]),  # Tokenize the text TODO need if for empty word
                #     'answer_box': element_data['box'],
                #     'answer_embeddings': tokenizer([word['text'] for word in element_data['words']], boxes=[word['box'] for word in element_data['words']]),   # Tokenize the text
                # }
                vector_db_row = [question_data['box'][0],question_data['box'][1],  # Q box top-left
                                 question_data['box'][0], question_data['box'][3],  # Q box bottom-left
                                 element_data['box'][0],element_data['box'][1],   # A box top-left
                                 element_data['box'][0], element_data['box'][3]]  # A box bottom-left
                words_list = [word['text'] for word in question_data['words']]
                box_list = [word['box'] for word in question_data['words']]

                vector_db_row = vector_db_row + tokenizer(words_list, boxes=box_list)['input_ids'][:max_element_tokens]  # Tokenize the text keeping only max number\
                words_list = [word['text'] for word in element_data['words']]
                box_list = [word['box'] for word in element_data['words']]
                vector_db_row = vector_db_row + tokenizer(words_list, boxes=box_list)['input_ids'][:max_element_tokens]  # Tokenize the text

                print('pause')
                vector_db.append(vector_db_row + [0] * (40 - len(vector_db_row)))  # pad with 0s
                # vector_db = np.append(vector_db, vector_db_row)
    print('Vector rows created')

    # https://www.pinecone.io/learn/faiss-tutorial/
    nlist = 10  # how many partitions (Voronoi cells) we’d like our index to have.
    vector_db = np.array(vector_db)
    d = vector_db.shape[1]  # dimension of the embeddings
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist)

    index.train(vector_db)
    index.is_trained  # check if index is now trained
    index.add(vector_db)
    index.ntotal  # number of embeddings indexed

    # test the index
    k = 3  # number of nearest neighbors we want to retrieve
    D, I = index.search(vector_db, k)
    print(I)  # neighbors
    print(D)  # distances

