import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_example_environment_has_portable_newlines():
    release = load_module("prepare_release", ROOT / "tools/prepare_release.py")
    assert ".example" in release.TEXT
    assert b"\r" not in (ROOT / "env.example").read_bytes()


def test_release_manifest_matches_public_files():
    public = load_module("reproduce_public", ROOT / "paper/reproduce_public.py")
    assert public.verify_manifest(ROOT) > 0
