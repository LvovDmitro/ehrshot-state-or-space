import importlib.util
from pathlib import Path
import re
import shutil

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("public_package", ROOT / "paper/reproduce_public.py")
public = importlib.util.module_from_spec(spec)
spec.loader.exec_module(public)


def test_readme_local_links_and_images_exist():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    links = re.findall(r"\]\(([^)]+)\)", text) + re.findall(r'src="([^"]+)"', text)
    for link in links:
        if not link.startswith(("https://", "http://", "#")):
            assert (ROOT / link.split("#", 1)[0]).exists(), link


def test_public_aggregates_validate():
    result = public.verify_aggregates(ROOT / "paper/aggregate_outputs")
    assert result["primary_comparisons"] == 10
    assert result["matched_size_summaries"] == 12


def test_invalid_published_metric_is_rejected(tmp_path):
    inputs = tmp_path / "aggregates"
    shutil.copytree(ROOT / "paper/aggregate_outputs", inputs)
    path = inputs / "recomputed_ensemble_metrics.csv"
    metrics = pd.read_csv(path)
    metrics.loc[0, "auroc"] = 1.1
    metrics.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Invalid published metric"):
        public.verify_aggregates(inputs)
