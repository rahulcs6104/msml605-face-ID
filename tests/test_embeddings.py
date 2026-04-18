import os
import numpy as np
import pytest
from PIL import Image

from src.embeddings import extract_embedding, preprocess_face
class TestPreprocessFace:

    @pytest.fixture
    def dummy_image(self, tmp_path):
        rng = np.random.default_rng(42)
        img = Image.fromarray(rng.integers(0, 255, (250, 250, 3), dtype=np.uint8))
        path = str(tmp_path / "test_face.jpg")
        img.save(path)
        return path

    def test_preprocess_returns_tensor(self, dummy_image):
        import torch
        face = preprocess_face(dummy_image)
        assert isinstance(face, torch.Tensor)

    def test_preprocess_shape(self, dummy_image):
        face = preprocess_face(dummy_image)
        assert face.shape == (3, 160, 160), f"Expected (3,160,160), got {face.shape}"

class TestExtractEmbedding:
    @pytest.fixture
    def dummy_image(self, tmp_path):
        rng = np.random.default_rng(42)
        img = Image.fromarray(rng.integers(0, 255, (250, 250, 3), dtype=np.uint8))
        path = str(tmp_path / "test_face.jpg")
        img.save(path)
        return path

    def test_embedding_shape_is_512(self, dummy_image):
        emb = extract_embedding(dummy_image)
        assert emb.shape == (512,),f"Expected (512,), got {emb.shape}"

    def test_embedding_dtype(self, dummy_image):
        emb = extract_embedding(dummy_image)
        assert emb.dtype in(np.float32, np.float64)

    def test_embedding_not_all_zeros(self, dummy_image):
        emb = extract_embedding(dummy_image)
        assert np.any(emb != 0),"Embedding should not be all zeros"

    def test_same_image_gives_same_embedding(self, dummy_image):
        emb1 = extract_embedding(dummy_image)
        emb2 = extract_embedding(dummy_image)
        np.testing.assert_allclose(emb1, emb2, atol=1e-6)

    def test_embedding_has_reasonable_norm(self, dummy_image):
        emb = extract_embedding(dummy_image)
        norm = np.linalg.norm(emb)
        assert 0.1 < norm < 50.0,f"Unexpected embedding norm: {norm}"