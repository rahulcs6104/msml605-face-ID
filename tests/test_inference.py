import os
import numpy as np
import pytest
from PIL import Image
from src.inference import verify_pair,compute_confidence
class TestComputeConfidence:
    def test_at_threshold_gives_half(self):
        c = compute_confidence(0.5, 0.5)
        assert c== 0.5, f"Expected 0.5,got {c}"
    def test_far_above_threshold_gives_high(self):
        c = compute_confidence(1.0, 0.5)
        assert c> 0.6, f"Expected >0.6,got {c}"
    def test_far_below_threshold_gives_high(self):
        c = compute_confidence(-1.0, 0.5)
        assert c> 0.6, f"Expected >0.6,got {c}"
    def test_range_is_always_valid(self):
        for score in np.linspace(-1, 1, 50):
            c= compute_confidence(score, 0.3)
            assert 0.5<= c<=1.0, f"confidence{c} out of [0.5,1.0] for score {score}"

    def test_equal_distance_gives_equal_confidence(self):
        """Score equidistant above and below threshold → same confidence."""
        c_above =compute_confidence(0.7, 0.5)
        c_below =compute_confidence(0.3, 0.5)
        assert abs(c_above-c_below) < 1e-6
    def test_closer_to_threshold_means_lower_confidence(self):
        c_near =compute_confidence(0.51, 0.5)
        c_far  =compute_confidence(0.9, 0.5)
        assert c_near< c_far


class TestVerifyPair:

    @pytest.fixture
    def two_images(self, tmp_path):
        rng =np.random.default_rng(42)
        paths =[]
        for i in range(2):
            img = Image.fromarray(rng.integers(0, 255, (250, 250, 3), dtype=np.uint8))
            p =str(tmp_path/f"face_{i}.jpg")
            img.save(p)
            paths.append(p)
        return paths

    def test_returns_all_required_keys(self,two_images):
        result =verify_pair(two_images[0],two_images[1], 0.5)
        for key in("score", "decision", "confidence", "latency", "threshold"):
            assert key in result, f"Missing required key: {key}"

    def test_decision_is_same_or_different(self,two_images):
        result= verify_pair(two_images[0],two_images[1], 0.5)
        assert result["decision"] in ("same", "different")

    def test_latency_is_positive(self,two_images):
        result= verify_pair(two_images[0],two_images[1], 0.5)
        assert result["latency"]["total_ms"] > 0

    def test_score_in_valid_range(self,two_images):
        result =verify_pair(two_images[0],two_images[1], 0.5)
        assert -1.0<=result["score"]<= 1.0

    def test_latency_breakdown_keys(self,two_images):
        result =verify_pair(two_images[0],two_images[1], 0.5)
        for key in ("embed_a_ms", "embed_b_ms","score_ms","decision_ms", "total_ms"):
            assert key in result["latency"],f"Missing latency key:{key}"