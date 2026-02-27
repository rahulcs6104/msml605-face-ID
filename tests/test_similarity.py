import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.similarity import (cosine_similarity_loop, cosine_similarity_vectorized, euclidean_distance_loop, euclidean_distance_vectorized)
TOLERANCE = 1e-10

class TestCosine:
    def test_same_vector_should_be_one(self):
        a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result =cosine_similarity_vectorized(a, a)
        np.testing.assert_allclose(result, np.ones(2), atol=1e-9)

    def test_perpendicular_vectors_should_be_zero(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[0.0, 1.0]])
        result =cosine_similarity_vectorized(a, b)
        np.testing.assert_allclose(result, np.array([0.0]), atol=1e-9)
    
    def test_opposite_vectors_should_be_minus_one(self):
        a = np.array([[1.0, 0.0]])
        b = np.array([[-1.0, 0.0]])
        result =cosine_similarity_vectorized(a, b)
        np.testing.assert_allclose(result, np.array([-1.0]), atol=1e-9)



    def test_output_is_right_shape(self):
        rng = np.random.default_rng(0)
        a = rng.standard_normal((100, 64))
        b = rng.standard_normal((100, 64))
        result =cosine_similarity_vectorized(a, b)

        assert result.shape == (100,)

    def test_loop_and_vectorized_give_same_answer(self):
        rng = np.random.default_rng(42)
        a =rng.standard_normal((200, 32))
        b = rng.standard_normal((200, 32))
        loop_result =cosine_similarity_loop(a, b)
        vectorized_result  =cosine_similarity_vectorized(a, b)
        max_diff =np.max(np.abs(loop_result - vectorized_result))
        assert max_diff < TOLERANCE, f"expected both methods to agree but got a difference of {max_diff}, max allowed is {TOLERANCE}"


    def test_result_is_between_minus_one_and_one(self):
        rng = np.random.default_rng(7)
        a = rng.standard_normal((500, 16))
        b = rng.standard_normal((500, 16))
        result =cosine_similarity_vectorized(a, b)
        assert np.all(result >= -1.0 - 1e-9) and np.all(result <= 1.0 + 1e-9)


class TestEuclidean:
    def test_same_vector_should_be_zero(self):
        a = np.array([[1.0, 2.0, 3.0]])
        result =euclidean_distance_vectorized(a, a)
        np.testing.assert_allclose(result, np.zeros(1), atol=1e-9)

    def test_simple_3_4_5_triangle(self):
        a = np.array([[0.0, 0.0]])
        b = np.array([[3.0, 4.0]])
        result =euclidean_distance_vectorized(a, b)
        np.testing.assert_allclose(result, np.array([5.0]), atol=1e-9)

    def test_output_is_right_shape(self):
        rng = np.random.default_rng(1)
        a = rng.standard_normal((100, 64))
        b = rng.standard_normal((100, 64))
        result =euclidean_distance_vectorized(a, b)
        assert result.shape == (100,)


    def test_distance_cant_be_negative(self):
        rng = np.random.default_rng(3)
        a = rng.standard_normal((500, 16))
        b = rng.standard_normal((500, 16))
        result =euclidean_distance_vectorized(a, b)
        assert np.all(result >= 0)

    def test_loop_and_vectorized_give_same_answer(self):
        rng = np.random.default_rng(99)
        a = rng.standard_normal((200, 32))
        b = rng.standard_normal((200, 32))
        loop_result =euclidean_distance_loop(a, b)
        vectorized_result  =euclidean_distance_vectorized(a, b)
        max_diff = np.max(np.abs(loop_result - vectorized_result))
        assert max_diff < TOLERANCE, f"expected both methods to agree but got a difference of {max_diff}, max allowed is {TOLERANCE}"
    def test_distance_a_to_b_equals_b_to_a(self):
        rng = np.random.default_rng(5)
        a = rng.standard_normal((50, 8))
        b = rng.standard_normal((50, 8))
        assert np.allclose(
            euclidean_distance_vectorized(a, b),
            euclidean_distance_vectorized(b, a),
        )