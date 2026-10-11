"""Build SEND_TO_MAM/EPCS_Paper_Draft_Overleaf_LITE.zip: the same project with figures downscaled to at most 1400 px wide
(fig1_pipeline 1500) so that it compiles faster on Overleaf's free plan. Text, tables and numbers are identical to the full
zip; only image resolution differs (about 210 dpi at full text width, fine for a review draft). Use the full zip for submission."""
import io
import zipfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "SEND_TO_MAM" / "EPCS_Paper_Draft_Overleaf.zip"
DST = ROOT / "SEND_TO_MAM" / "EPCS_Paper_Draft_Overleaf_LITE.zip"
LIMIT = {"fig1_pipeline.png": 1500}
DEFAULT = 1400

zin = zipfile.ZipFile(SRC)
before = after = 0
with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED) as zout:
    for info in zin.infolist():
        data = zin.read(info.filename)
        n = info.filename
        if n.startswith("figures/") and n.lower().endswith((".png", ".jpg")):
            before += len(data)
            im = Image.open(io.BytesIO(data)).convert("RGB")
            w = LIMIT.get(Path(n).name, DEFAULT)
            if im.width > w:
                im = im.resize((w, round(im.height * w / im.width)), Image.LANCZOS)
            buf = io.BytesIO()
            if n.lower().endswith(".jpg"):
                im.save(buf, "JPEG", quality=85)
            else:
                im.save(buf, "PNG", optimize=True, dpi=(300, 300))
            data = buf.getvalue()
            after += len(data)
        elif n == "README_OVERLEAF.md":
            data = (data.decode("utf-8") + "\n\n## LITE version\nThis zip has the same text as the full version; figures are downscaled to at most 1400 px wide to compile faster on the free plan. Use the full zip for the final submission.\n").encode("utf-8")
        zout.writestr(n, data)
print(f"figures {before / 1e6:.1f} MB -> {after / 1e6:.1f} MB; zip {DST.stat().st_size / 1e6:.1f} MB")
