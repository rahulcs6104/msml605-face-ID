import numpy as np
from PIL import Image

from src.pairs import load_pairs
from src.similarity import cosine_similarity_vectorized, euclidean_distance_vectorized


def load_image_as_vector(path: str,image_size:tuple):
    img = Image.open(path).convert("L").resize(image_size,Image.LANCZOS)
    return np.asarray(img,dtype=np.float32).flatten()


def score_pairs(pairs: list,metric:str="cosine",image_size:tuple=(50,50)):
    n= len(pairs)
    dim = image_size[0] * image_size[1]
    L= np.zeros((n,dim), dtype=np.float32)
    R=np.zeros((n,dim), dtype=np.float32)
    labels= np.zeros(n, dtype=np.int32)
    for i, pair in enumerate(pairs):
        L[i]= load_image_as_vector(pair["left_path"],  image_size)
        R[i]= load_image_as_vector(pair["right_path"], image_size)
        labels[i]= int(pair["label"])
    if metric == "euclidean":
        scores=-euclidean_distance_vectorized(L, R)
    elif metric == "cosine":
        scores=cosine_similarity_vectorized(L, R)
    else:
        raise ValueError(f"Unknown distance metric: '{metric}'. Please use euclidean or cosine")
    return scores,labels


def apply_threshold(scores,threshold):
    ret=[]
    for i in scores:
        if i>=threshold:
            ret.append(1)
        else:
            ret.append(0)
    return np.array(ret, dtype=np.int32)