import numpy as np

#loop implmnetations
def cosine_similarity_loop(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    Number_of_rows = a.shape[0]
    ret = np.empty(Number_of_rows,dtype=np.float64)
    for i in range(Number_of_rows):
        dot = np.dot(a[i], b[i])
        norm_a = np.linalg.norm(a[i])
        norm_b = np.linalg.norm(b[i])
        denominator = norm_a*norm_b

        if denominator>1e-10:
            ret[i] = dot /denominator
        else:
            ret[i]=0.0

    return ret


def euclidean_distance_loop(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    Number_of_rows = a.shape[0]
    ret = np.empty(Number_of_rows,dtype=np.float64)
    for i in range(Number_of_rows):
        ret[i] =np.linalg.norm(a[i]- b[i])


    return ret




#vectorized implementation
def cosine_similarity_vectorized(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dot_products = np.sum(a * b, axis=1)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    denominator = norm_a * norm_b
    denominator = np.where(denominator ==0,1e-10,denominator)

    return dot_products /denominator


def euclidean_distance_vectorized(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    difference =a-b
    return np.linalg.norm(difference,axis=1)




