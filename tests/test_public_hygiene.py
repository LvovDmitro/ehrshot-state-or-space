"""Check public inputs without credentials or external services."""
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DIRS = ["configs", "docs", "final_exps", "paper", "scripts", "tests", "tools", ".github"]


def public_files():
    for path in ROOT.iterdir():
        if path.is_file():
            yield path
    for name in PUBLIC_DIRS:
        yield from (path for path in (ROOT / name).rglob("*")
                    if path.is_file() and "__pycache__" not in path.parts
                    and "reproduced" not in path.parts)


def test_dataset_defaults_are_local():
    config = json.loads((ROOT / "configs/state_or_space_sequence_datasets.json").read_text())
    assert config["clearml"]["enabled"] is False
    assert config["build"]["upload"] is False
    assert config["paths"]["output_s3_prefix"] == ""


def test_public_files_have_no_literal_credentials_or_private_addresses():
    patterns = [
        r"github_pat_[A-Za-z0-9_]{30,}", r"gh[pousr]_[A-Za-z0-9]{30,}",
        r"AKIA[A-Z0-9]{16}", r"hf_[A-Za-z0-9]{30,}",
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        r"https?://(?:10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+)",
    ]
    for path in public_files():
        if path.suffix.lower() in {".png", ".pdf"}:
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in patterns:
            assert re.search(pattern, text) is None, str(path.relative_to(ROOT))


def test_public_files_exclude_models_records_and_checksum_reports():
    restricted = {".parquet", ".arrow", ".feather", ".pt", ".pth", ".ckpt", ".npz",
                  ".npy", ".pkl", ".pickle", ".joblib", ".zip"}
    for path in public_files():
        assert path.suffix.lower() not in restricted, str(path.relative_to(ROOT))
    assert not (ROOT / "RELEASE_MANIFEST.csv").exists()


def test_readme_does_not_link_to_itself():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "[Code](https://github.com/LvovDmitro/ehrshot-state-or-space)" not in text
