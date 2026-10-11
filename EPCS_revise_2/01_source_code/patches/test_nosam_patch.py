"""CPU test for the no-SAM patch. Never touches the live models_r3.py.
Run:  python test_nosam_patch.py     (forces CPU)
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
LIVE = HERE.parent / "shared" / "models_r3.py"
tmp = Path(tempfile.mkdtemp(prefix="nosam_patch_"))
shutil.copy(LIVE, tmp / "models_live.py")
subprocess.check_call([sys.executable, str(HERE / "models_r3_nosam_patch.py"),
                       "--target", str(tmp / "models_live.py"),
                       "--out", str(tmp / "models_patched.py")])


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


import torch
torch.manual_seed(0)
live = load("m_live", tmp / "models_live.py")
pat = load("m_pat", tmp / "models_patched.py")
fails = []


def check(ok, msg):
    print(("PASS  " if ok else "FAIL  ") + msg)
    if not ok:
        fails.append(msg)


# 1. new variant builds with a bogus checkpoint path (none needed) and runs
spec = pat.MODELS["abl0_nosam_epcsbackbone"]
m = pat.build_model("abl0_nosam_epcsbackbone", sam_checkpoint=str(tmp / "no_such_sam.pth"))
check(m.context is None and spec["sam"] is False,
      "no SAM encoder constructed, registry sam=False (bogus checkpoint path accepted)")
check(spec["group"] == "ablation" and spec["loss"] == "bce_dice"
      and spec["label"] == "0. EPCS backbone, no SAM", "registry fields")
m.eval()
with torch.no_grad():
    y = m(torch.randn(1, 3, 256, 256))
check(tuple(y.shape) == (1, 1, 256, 256), f"forward shape {tuple(y.shape)}")
tot, trn = pat.count_params(m)
print(f"      params total={tot:,} trainable={trn:,}")
check(tot == trn and tot > 0, "all parameters trainable (no frozen encoder)")
check(not any(n.split(".")[0] in ("f1", "f2", "f3", "f4", "context") for n, _ in m.named_parameters()),
      "no fusion/context parameters")
ref = pat.EPCSNet(None, sam_source="random")   # full EPCS-Net, no checkpoint needed


def sub(mod, pre):
    return sum(p.numel() for n, p in mod.named_parameters() if n.split(".")[0] in pre)


core = ("e1", "e2", "e3", "e4", "d1", "d2", "d3", "d4", "head")
check(sub(m, core) == sub(ref, core),
      f"encoder+decoder+head params identical to EPCS-Net ({sub(m, core):,})")

# 2. existing models unchanged
for mid in ["unet", "unetpp", "unet3plus", "eunet3plus", "abl1_nosam", "ablB_random_embed"]:
    a = live.count_params(live.build_model(mid))
    b = pat.count_params(pat.build_model(mid))
    check(a == b, f"{mid}: live {a} == patched {b}")
if Path(live.SAM_CHECKPOINT).is_file():
    for mid in ["sam_unet", "epcs_net", "abl2_bottleneck", "abl3_pspi", "abl4_bdsg", "abl5_dksr",
                "ablA_acfb_nopspi", "ablC_pspi_concat", "ablD_gn_encoder",
                "ref21_bottleneck", "ref21_pyramid"]:
        a = live.count_params(live.build_model(mid))
        b = pat.count_params(pat.build_model(mid))
        check(a == b, f"{mid}: live {a} == patched {b}")
else:
    print("      (SAM checkpoint absent; SAM models compared via random-embed twins)")
    for tag, kw in [("epcs", {}), ("bottleneck_concat", dict(injection="bottleneck", fusion="concat")),
                    ("pspi_bdsg", dict(fusion="bdsg")), ("gn", dict(encoder_norm="gn"))]:
        a = live.count_params(live.EPCSNet(None, sam_source="random", **kw))
        b = pat.count_params(pat.EPCSNet(None, sam_source="random", **kw))
        check(a == b, f"EPCSNet(random,{tag}): live {a} == patched {b}")
check(set(pat.MODELS) - set(live.MODELS) == {"abl0_nosam_epcsbackbone"}
      and set(live.MODELS) <= set(pat.MODELS), "registry differs only by the new key")
check("abl0_nosam_epcsbackbone" not in LIVE.read_text(encoding="utf-8"), "live models_r3.py not modified")
print("\nALL PASSED" if not fails else f"\n{len(fails)} FAILED")
shutil.rmtree(tmp, ignore_errors=True)
sys.exit(1 if fails else 0)
