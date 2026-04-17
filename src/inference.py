import time
import numpy as np
from src.embeddings import extract_embedding
from src.similarity import cosine_similarity_vectorized


def compute_confidence(score: float, threshold: float,min_score: float = -1.0,max_score:float = 1.0):
    margin = abs(score-threshold)
    max_margin = max(abs(max_score-threshold),abs(threshold-min_score))
    confidence = 0.5+0.5*(margin/max(max_margin, 1e-10))
    return round(min(confidence, 1.0), 4)


def verify_pair(image_path_a: str, image_path_b: str,threshold: float):
    timings = {}
    t0 = time.perf_counter()
    emb_a = extract_embedding(image_path_a)
    timings["embed_a_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    t0 = time.perf_counter()
    emb_b = extract_embedding(image_path_b)
    timings["embed_b_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    t0 = time.perf_counter()
    score = float(cosine_similarity_vectorized(emb_a.reshape(1, -1), emb_b.reshape(1, -1))[0])
    timings["score_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    t0 = time.perf_counter()
    if score >= threshold:
        decision="same"
    else:
        decision="different"

    confidence = compute_confidence(score, threshold)
    t1=time.perf_counter()
    elapsed_seconds=t1-t0
    elapsed_ms = elapsed_seconds*1000
    timings["decision_ms"] =round(elapsed_ms, 2)
    timings["total_ms"] = round(
        timings["embed_a_ms"]+timings["embed_b_ms"]+timings["score_ms"]+timings["decision_ms"],2)
    return {"image_a":image_path_a,"image_b":image_path_b,"score":round(score,6),"threshold":threshold,"decision":decision,"confidence": confidence,"latency":timings,}