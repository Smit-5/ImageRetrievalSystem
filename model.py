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

import json

# from collections import OrderedDict

# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

model, preprocess = clip.load("ViT-B/32")
model.cuda().eval()
input_resolution = model.visual.input_resolution
context_length = model.context_length
vocab_size = model.vocab_size


def create_image_embedding(images):  # Ensure that image is kind of [ img1 ] 
    
    image_input = torch.tensor(np.stack(images)).cuda()

    with torch.no_grad():
        image_embeddings = model.encode_image(image_input).float() 
        image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
    return image_embeddings

def input_img_embedding(img):
    images = []
    images.append(preprocess(img.convert("RGB")))

    query_vector = create_image_embedding(images)
    query_vector = query_vector.cpu()
    return query_vector

def input_text_embedding(txt):
    text = []
    text.append(txt)
    query_text=clip.tokenize(text)
    # texts=clip.tokenize(texts)
    with torch.no_grad():
        # text_embeddings = model.encode_text(texts.cuda()).float()
        query_text_embedding = model.encode_text(query_text.cuda()).float()
        query_text_embedding /= query_text_embedding.norm(dim=-1, keepdim=True)
    query_text_embedding = query_text_embedding.cpu()
    return query_text_embedding

def serialize_embedding(embedding):
    return json.dumps(embedding.tolist())  # Convert to a list and then to JSON string

def deserialize_embedding(serialized_embedding):
    return np.array(json.loads(serialized_embedding))  # Convert back to NumPy array