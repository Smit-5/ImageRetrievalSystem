import clip
import os
import skimage
import IPython.display
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torch
import model

import json

# from collections import OrderedDict

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

model1, preprocess = clip.load("ViT-B/32")
model1.cuda().eval()
input_resolution = model1.visual.input_resolution
context_length = model1.context_length
vocab_size = model1.vocab_size


def search_similar_vectors(database, query_vector, top_k=5):
    """
    Function to search for similar vectors in a dataframe.

    Args:
    - database: DataFrame containing vectors to search through.
    - query_vector: Vector to compare against the database.
    - top_k: Number of most similar vectors to retrieve.

    Returns:
    - top_k_similar: DataFrame containing top k similar vectors.
    """

    # Calculate cosine similarity between query_vector and vectors in the database
    similarity_scores = cosine_similarity(loaded_embeddings, query_vector)

    # Create a DataFrame to store similarity scores along with their indices
    similarity_df = pd.DataFrame({'similarity': similarity_scores.flatten()})

    # Combine similarity scores with the original dataframe
    database_with_similarity = pd.concat([database, similarity_df], axis=1)

    # Sort the dataframe by similarity scores in descending order and retrieve top k
    top_k_similar = database_with_similarity.sort_values(by='similarity', ascending=False).head(top_k)

    return top_k_similar

# Example usage:
# Assuming you have a DataFrame named 'database' with a column named 'vector'
# and a query vector 'query_vector'

# Call the function to retrieve top 5 similar vectors
# Replace 'database' and 'query_vector' with your actual data
# and adjust 'top_k' as needed

# images = []
# images.append(preprocess(Image.open(f'imgtest.jpg').convert("RGB")))

database = pd.read_csv('dataIRS.csv')

loaded_embeddings = [model.deserialize_embedding(emb) for emb in database['Serialized_Embeddings']]

# query_vector = model.create_image_embedding(images)
# query_vector = query_vector.cpu()

# top_k_similar = search_similar_vectors(database, query_vector, top_k=5)

# Display the top k similar vectors
# print(top_k_similar)


# Incomplete code from here
# def search_image(query_img):
#     query_img=create_image_embedding(query_img)
#     filename=''
#     database=pd.read_csv(filename)