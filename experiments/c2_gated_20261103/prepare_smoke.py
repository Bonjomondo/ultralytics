"""Create server-only 32-train/8-val lists from the existing local dataset."""
from pathlib import Path

import yaml

from common import EXP, ROOT, RUNS


def main():
    data = yaml.safe_load((ROOT / "configs/visdrone-local.yaml").read_text())
    dataset = Path(data["path"])
    for split, count in [("train", 32), ("val", 8)]:
        images = sorted((dataset / data[split]).glob("*.jpg"))[:count]
        assert len(images) == count
        listing = RUNS / "preflight" / f"{split}.txt"
        listing.parent.mkdir(parents=True, exist_ok=True)
        listing.write_text("".join(str(p) + "\n" for p in images))
        data[split] = str(listing)
    (EXP / "data-smoke.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    print(EXP / "data-smoke.yaml")


if __name__ == "__main__":
    main()
