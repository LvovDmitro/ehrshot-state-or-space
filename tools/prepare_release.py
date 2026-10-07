"""Normalize public text, audit secrets/data, and rebuild release checksums."""
from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
TEXT = {".py", ".sh", ".csv", ".json", ".md", ".txt", ".cff", ".tex", ".bib", ".yaml", ".yml"}
EXCLUDED = {"RELEASE_MANIFEST.csv", "PROJECT_MANIFEST.json", "SHA256SUMS.txt",
            "public_results/SHA256SUMS.txt"}
TOKENS = [re.compile(r"github_pat_[A-Za-z0-9_]{30,}"),
          re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
          re.compile(r"AKIA[A-Z0-9]{16}"),
          re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")]
RESTRICTED = {".parquet", ".arrow", ".feather", ".pt", ".pth", ".ckpt", ".npz",
              ".npy", ".pkl", ".pickle", ".joblib", ".zip"}


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def check_payload(path: str, data: bytes) -> None:
    if Path(path).suffix.lower() in RESTRICTED:
        raise ValueError(f"Restricted/binary model-data file tracked: {path}")
    text = data.decode("utf-8", errors="replace")
    if any(pattern.search(text) for pattern in TOKENS):
        raise ValueError(f"Credential candidate in file (value redacted): {path}")
    if path.endswith(".csv"):
        header = next(csv.reader(io.StringIO(text)), [])
        if set(header) & {"row_id", "subject_id", "patient_id", "candidate_code", "prediction_time"}:
            raise ValueError(f"Restricted row-level CSV schema: {path}")


def main():
    names = sorted(set(git("ls-files", "--cached", "--others", "--exclude-standard")
                       .decode("utf-8").splitlines()))
    entries = []
    for name in names:
        path = ROOT / name
        if not path.is_file() or name in EXCLUDED:
            continue
        data = path.read_bytes()
        check_payload(name, data)
        if path.suffix in TEXT or path.name in {".gitattributes", ".gitignore", ".python-version"}:
            data = data.replace(b"\r\n", b"\n")
            path.write_bytes(data)
        entries.append({"relative_path": name, "sha256": hashlib.sha256(data).hexdigest(),
                        "bytes": len(data)})
    # Historical scan prints no values or clinical content, and reads only Git blobs.
    revisions = git("rev-list", "--all").decode().splitlines()
    checked = set()
    for revision in revisions:
        tree = git("ls-tree", "-r", revision).decode("utf-8").splitlines()
        for line in tree:
            metadata, name = line.split("\t", 1)
            blob = metadata.split()[2]
            if blob in checked:
                continue
            data = git("cat-file", "blob", blob)
            # Historical public task metadata is not a credential or clinical row.
            if any(pattern.search(data.decode("utf-8", errors="replace")) for pattern in TOKENS):
                raise ValueError("Credential candidate in Git history (value redacted)")
            if Path(name).suffix.lower() in RESTRICTED:
                raise ValueError("Restricted/model binary in Git history")
            if name.endswith(".csv") and set(next(csv.reader(io.StringIO(
                    data.decode("utf-8", errors="replace"))), [])) & {
                        "row_id", "subject_id", "patient_id", "candidate_code"}:
                raise ValueError("Restricted CSV schema in Git history")
            checked.add(blob)
    with (ROOT / "RELEASE_MANIFEST.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "sha256", "bytes"])
        writer.writeheader()
        writer.writerows(entries)
    (ROOT / "PROJECT_MANIFEST.json").write_text(json.dumps({
        "release": "camera-ready-2026-10-07",
        "original_commit": "9831154370cde4b4ae173420b57704f55abd0c68",
        "files": entries}, indent=2) + "\n", encoding="utf-8")
    (ROOT / "SHA256SUMS.txt").write_text("".join(
        f"{entry['sha256']}  {entry['relative_path']}\n" for entry in entries), encoding="utf-8")
    legacy = [entry for entry in entries if entry["relative_path"].startswith("public_results/")]
    (ROOT / "public_results" / "SHA256SUMS.txt").write_text("".join(
        f"{entry['sha256']}  {entry['relative_path'][15:]}\n" for entry in legacy), encoding="utf-8")
    print(json.dumps({"public_files": len(entries), "history_commits": len(revisions),
                     "history_blobs": len(checked), "literal_token_candidates": 0,
                     "restricted_csv_or_data_candidates": 0,
                     "note": "Pattern/schema scan; not proof that every possible secret is absent."}))


if __name__ == "__main__":
    main()
