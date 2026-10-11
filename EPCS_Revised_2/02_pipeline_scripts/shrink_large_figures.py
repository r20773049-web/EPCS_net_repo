"""Downscale oversized result figures to 2000 px wide (about 300 dpi at journal column width) after the figure generators run.

Originals are copied once to 05_manuscript/_full_resolution_originals (never overwritten). Only PNGs above SIZE_LIMIT_MB in
05_manuscript/figures and 05_manuscript/figures_r3 are touched; schematic figures (fig1-4) are left alone. Idempotent:
a figure that is already <= 2000 px wide is skipped."""
import shutil
from pathlib import Path

from PIL import Image

M = Path(__file__).resolve().parents[1] / "05_manuscript"
ORIG = M / "_full_resolution_originals"
SIZE_LIMIT_MB = 1.5
WIDTH = 2000


def main():
    ORIG.mkdir(exist_ok=True)
    for folder in (M / "figures", M / "figures_r3"):
        for f in sorted(folder.glob("*.png")):
            if f.name.startswith(("fig1_", "fig2_", "fig3_", "fig4_")):
                continue
            size_mb = f.stat().st_size / 1e6
            if size_mb <= SIZE_LIMIT_MB:
                continue
            backup = ORIG / f.name
            if not backup.exists():
                shutil.copy2(f, backup)
            im = Image.open(f).convert("RGB")
            if im.width <= WIDTH:
                continue
            im = im.resize((WIDTH, round(im.height * WIDTH / im.width)), Image.LANCZOS)
            im.save(f, optimize=True, dpi=(300, 300))
            print(f"{f.name}: {size_mb:.2f} -> {f.stat().st_size / 1e6:.2f} MB ({im.width}x{im.height})")


if __name__ == "__main__":
    main()
