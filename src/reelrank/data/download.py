"""Download and extract MovieLens datasets (idempotent)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

MIRRORS = {
    "ml-latest-small": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",
    "ml-25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
}


def download_movielens(dataset: str, raw_dir: str | Path) -> Path:
    """Download and extract a MovieLens dataset.

    Returns the extracted folder (which contains ratings.csv, movies.csv, ...).
    Skips the download when the data is already present.
    """
    if dataset not in MIRRORS:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {list(MIRRORS)}")

    raw_dir = Path(raw_dir)
    target = raw_dir / dataset
    if (target / "ratings.csv").exists():
        return target

    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / f"{dataset}.zip"
    if not zip_path.exists():
        _stream_download(MIRRORS[dataset], zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)

    if not (target / "ratings.csv").exists():
        raise RuntimeError(f"extraction did not produce {target / 'ratings.csv'}")
    return target


def _stream_download(url: str, dest: Path) -> None:
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
