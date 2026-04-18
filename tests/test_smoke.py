import json
import os
import subprocess
import numpy as np
import pytest
from PIL import Image


class TestSmoke:
    @pytest.fixture
    def sample_images(self, tmp_path):
        rng = np.random.default_rng(99)
        paths = []
        for i in range(2):
            img = Image.fromarray(rng.integers(0, 255, (250, 250, 3), dtype=np.uint8))
            p=str(tmp_path/f"smoke_{i}.jpg")
            img.save(p)
            paths.append(p)
        return paths

    def test_cli_single_pair_completes(self, sample_images, tmp_path):
        output_json=str(tmp_path/"smoke_output.json")
        result=subprocess.run(["python", "scripts/cli.py","--config", "configs/m3.yaml","--image-a", sample_images[0],"--image-b", sample_images[1],"--output-json", output_json,],capture_output=True,text=True,timeout=120,)
        assert result.returncode == 0, (
            f"CLI exited with code {result.returncode}.\n"
            f"STDERR:\n{result.stderr}"
        )
        assert os.path.exists(output_json), "Output JSON was not created"
        with open(output_json) as f:
            data = json.load(f)
        assert isinstance(data, list) and len(data)==1
        entry=data[0]
        assert "score" in entry,"mising 'score' in the output"
        assert "decision" in entry,"mising 'decision' in the output"
        assert "confidence" in entry,"mising 'confidence' in the output"
        assert "latency" in entry,"mising 'latency' in the output"
        assert "threshold" in entry,"mising 'threshold' in the output"
        assert entry["decision"] in ("same", "different")
        assert 0.5 <= entry["confidence"] <= 1.0
        assert entry["latency"]["total_ms"] > 0