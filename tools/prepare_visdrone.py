from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image

ASSETS_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0"
SPLITS = {
    "train": ("VisDrone2019-DET-train.zip", "VisDrone2019-DET-train"),
    "val": ("VisDrone2019-DET-val.zip", "VisDrone2019-DET-val"),
    "test": ("VisDrone2019-DET-test-dev.zip", "VisDrone2019-DET-test-dev"),
}
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_dataset_root() -> Path:
    return (repo_root() / "datasets" / "VisDrone").resolve()


def progress(blocks: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    downloaded = min(blocks * block_size, total_size)
    percent = downloaded * 100 / total_size
    size_mb = total_size / 1024 / 1024
    downloaded_mb = downloaded / 1024 / 1024
    print(f"\r  {downloaded_mb:7.1f}/{size_mb:7.1f} MB  {percent:5.1f}%", end="", flush=True)
    if downloaded >= total_size:
        print()


def download(url: str, destination: Path, force: bool = False) -> None:
    use_curl = bool(shutil.which("curl.exe"))
    if destination.exists() and not force and destination.stat().st_size > 0 and not use_curl:
        print(f"[skip] archive exists: {destination}")
        return
    if force and destination.exists():
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    action = "resume" if destination.exists() and destination.stat().st_size > 0 and not force else "download"
    print(f"[{action}] {url}")
    if use_curl:
        command = ["curl.exe", "-L", "-C", "-", "-o", str(destination), url]
        subprocess.run(command, check=True)
        return
    urllib.request.urlretrieve(url, destination, reporthook=progress)


def unzip(archive: Path, destination: Path, force: bool = False) -> None:
    source_name = archive.stem
    source_dir = destination / source_name
    if source_dir.exists() and not force:
        print(f"[skip] extracted folder exists: {source_dir}")
        return
    print(f"[unzip] {archive.name} -> {destination}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(destination)


def find_image(images_dir: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def move_images(source_dir: Path, target_dir: Path) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for image_path in sorted(source_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        destination = target_dir / image_path.name
        if destination.exists():
            image_path.unlink()
        else:
            shutil.move(str(image_path), str(destination))
            moved += 1
    return moved


def convert_annotations(source_dir: Path, images_dir: Path, labels_dir: Path) -> int:
    labels_dir.mkdir(parents=True, exist_ok=True)
    if not source_dir.exists():
        return 0

    converted = 0
    for annotation_file in sorted(source_dir.glob("*.txt")):
        image_path = find_image(images_dir, annotation_file.stem)
        if image_path is None:
            raise FileNotFoundError(f"image for annotation not found: {annotation_file.name}")

        with Image.open(image_path) as image:
            width, height = image.size

        dw, dh = 1.0 / width, 1.0 / height
        lines = []
        for row in annotation_file.read_text(encoding="utf-8").splitlines():
            fields = row.split(",")
            if len(fields) < 6 or fields[4] == "0":
                continue
            x, y, w, h = map(int, fields[:4])
            cls = int(fields[5]) - 1
            x_center = (x + w / 2) * dw
            y_center = (y + h / 2) * dh
            w_norm = w * dw
            h_norm = h * dh
            lines.append(f"{cls} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        (labels_dir / annotation_file.name).write_text("".join(lines), encoding="utf-8")
        converted += 1

    return converted


def image_count(images_dir: Path) -> int:
    if not images_dir.exists():
        return 0
    return sum(1 for path in images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def prepare_split(dataset_root: Path, split: str, archive_name: str, source_name: str, skip_download: bool, keep_raw: bool, force: bool) -> None:
    archive_path = dataset_root / archive_name
    source_dir = dataset_root / source_name
    images_dir = dataset_root / "images" / split
    labels_dir = dataset_root / "labels" / split

    if not skip_download:
        download(f"{ASSETS_URL}/{archive_name}", archive_path, force=force)
        unzip(archive_path, dataset_root, force=force)

    if not source_dir.exists():
        if images_dir.exists():
            print(f"[skip] split '{split}' already prepared at {images_dir}")
            return
        raise FileNotFoundError(f"raw split folder not found: {source_dir}")

    moved = move_images(source_dir / "images", images_dir)
    converted = convert_annotations(source_dir / "annotations", images_dir, labels_dir)
    print(
        f"[done] {split}: images={image_count(images_dir)} labels={converted} moved={moved} target={images_dir.parent}"
    )

    if not keep_raw:
        shutil.rmtree(source_dir, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and convert VisDrone2019-DET to YOLO format.")
    parser.add_argument("--root", type=Path, default=default_dataset_root(), help="Dataset root directory.")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=tuple(SPLITS),
        default=list(SPLITS),
        help="Dataset splits to prepare.",
    )
    parser.add_argument("--skip-download", action="store_true", help="Only convert already-downloaded archives/folders.")
    parser.add_argument("--keep-raw", action="store_true", help="Keep the extracted VisDrone folders after conversion.")
    parser.add_argument("--force", action="store_true", help="Re-download and re-extract archives if they already exist.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = args.root.resolve()
    dataset_root.mkdir(parents=True, exist_ok=True)
    print(f"[root] {dataset_root}")

    for split in args.splits:
        archive_name, source_name = SPLITS[split]
        prepare_split(
            dataset_root=dataset_root,
            split=split,
            archive_name=archive_name,
            source_name=source_name,
            skip_download=args.skip_download,
            keep_raw=args.keep_raw,
            force=args.force,
        )

    print("[ready] use data=VisDrone-local.yaml from the repo root for the workspace-local dataset.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
