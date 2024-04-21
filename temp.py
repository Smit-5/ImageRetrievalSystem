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
from PIL import Image

import json

model1, preprocess = clip.load("ViT-B/32")
model1.cuda().eval()
input_resolution = model1.visual.input_resolution
context_length = model1.context_length
vocab_size = model1.vocab_size



# path="\train"
format_='jpg'
num_images=301
images=[];names=[]
for i in range(1,num_images):
    # images.append(Image.open(f'{i}.{format_}'))
    images.append(preprocess(Image.open(f'img ({i}).jpg').convert("RGB")))
    names.append(f'img ({i}).{format_}')

embeddings = model.create_image_embedding(images)
embeddings = embeddings.cpu()  # Convert to NumPy array on the CPU 

data={
    'Name':names,
#     'Embeddings':embeddings
}

database = pd.DataFrame(data)

serialized_embeddings = [model.serialize_embedding(emb) for emb in embeddings]
database['Serialized_Embeddings'] = serialized_embeddings

path_to_csv = r'C:\Users\smitr\Downloads\train-test\dataIRS.csv'

database.to_csv(path_to_csv)

# loaded_embeddings = [model.deserialize_embedding(emb) for emb in database['Serialized_Embeddings']]
# to load embeddings from .csv