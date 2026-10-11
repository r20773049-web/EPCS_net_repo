"""
Patch for shared/models_r3.py (audit finding S1): adds EPCSNet(sam_source="none")
and the registry entry 'abl0_nosam_epcsbackbone'.

Usage:
    python models_r3_nosam_patch.py --target PATH/TO/models_r3.py            # patch in place
    python models_r3_nosam_patch.py --target SRC.py --out DST.py             # patch a copy
Idempotent: refuses to patch twice. Every edit asserts its anchor matches exactly once.

DESIGN (least invasive faithful definition), from reading EPCSNet:
  * EPCSNet = frozen context encoder + 4-stage ConvBlock encoder (e1..e4) + one
    fusion block per scale (f1..f4, ACFB by default) + dense decoder (d4..d1)
    that consumes f1..f4. The decoder never sees e1..e4 directly, only f1..f4.
  * The existing injection="bottleneck" branch already defines "no fusion at a
    scale" as the identity: f1,f2,f3 = e1,e2,e3. We extend that identity to the
    fourth scale (f4 = e4). Result: the decoder, channel widths, skip topology
    and ConvBlocks are exactly those of EPCS-Net; only the SAM context and ALL
    fusion blocks are removed. No new layers, no 1x1 projections, so the only
    difference from rows 2+ is the absence of SAM input (and the fusion
    parameters needed to consume it). This isolates "add SAM" from
    "change backbone".
  * Alternatives rejected: (a) zero/constant context fed through ACFB keeps the
    fusion parameters and gates, so it is not "no SAM" (ablB's random embedding
    already probes that); (b) a new nn.Module backbone would duplicate the
    decoder and risk drift.
  * No checkpoint: the context encoder is not constructed (self.context=None),
    forward() never touches ctx, and the registry sets sam=False, so
    build_model() skips the checkpoint check and the protocol skips SamCache.
  * Existing behaviour is unchanged for sam_source in ("sam","random"): the new
    code paths are only taken when sam_source == "none".
"""
from __future__ import annotations

import argparse
from pathlib import Path

EDITS = [
    # 1. accept the new option
    ('        if sam_source not in ("sam", "random"):\n',
     '        if sam_source not in ("sam", "random", "none"):\n'),
    # 2. do not build a context encoder for "none"
    ('        self.context = (SAMImageEncoder(checkpoint) if sam_source == "sam"\n'
     '                        else RandomFrozenEncoder(256))\n',
     '        self.sam_source = sam_source\n'
     '        self.no_context = (sam_source == "none")\n'
     '        if self.no_context:\n'
     '            self.context = None      # no SAM, no checkpoint, no fusion blocks\n'
     '        else:\n'
     '            self.context = (SAMImageEncoder(checkpoint) if sam_source == "sam"\n'
     '                            else RandomFrozenEncoder(256))\n'),
    # 3. no fusion blocks for "none" (decoder consumes e1..e4 as f1..f4)
    ('        make = FUSIONS[fusion]\n',
     '        make = FUSIONS[fusion]\n'
     '        if self.no_context:\n'
     '            injection = "none"        # skip both fusion-construction branches below\n'
     '            self.injection = "none"\n'),
    ('        else:\n            self.f4 = make(8*b, 256, None)\n',
     '        elif injection == "bottleneck":\n            self.f4 = make(8*b, 256, None)\n'),
    # 4. forward: identity pass-through, ctx ignored
    ('        ctx = self.context(x) if ctx is None else ctx\n'
     '        e1 = self.e1(x); e2 = self.e2(self.pool(e1))\n'
     '        e3 = self.e3(self.pool(e2)); e4 = self.e4(self.pool(e3))\n',
     '        if not self.no_context:\n'
     '            ctx = self.context(x) if ctx is None else ctx\n'
     '        e1 = self.e1(x); e2 = self.e2(self.pool(e1))\n'
     '        e3 = self.e3(self.pool(e2)); e4 = self.e4(self.pool(e3))\n'),
    ('        if self.injection == "pyramid":\n            f1 = self.f1(e1, ctx_for(e1))\n',
     '        if self.no_context:\n'
     '            f1, f2, f3, f4 = e1, e2, e3, e4      # identity at all four scales\n'
     '        elif self.injection == "pyramid":\n            f1 = self.f1(e1, ctx_for(e1))\n'),
    # 5. registry entry, placed before row 1
    ('    "abl1_nosam":        dict(build=lambda c: EUNet3Plus()',
     '    "abl0_nosam_epcsbackbone": dict(build=_epcs(sam_source="none"), loss="bce_dice", sam=False,\n'
     '                              group="ablation", label="0. EPCS backbone, no SAM"),\n'
     '    "abl1_nosam":        dict(build=lambda c: EUNet3Plus()'),
]


def apply(text: str) -> str:
    if "abl0_nosam_epcsbackbone" in text:
        raise RuntimeError("already patched")
    for old, new in EDITS:
        n = text.count(old)
        if n != 1:
            raise RuntimeError(f"anchor matched {n} times (expected 1):\n{old}")
        text = text.replace(old, new)
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", default=None, help="write here instead of in place")
    a = ap.parse_args()
    src = Path(a.target)
    raw = src.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw          # preserve the file's line endings
    patched = apply(raw.replace("\r\n", "\n"))
    if crlf:
        patched = patched.replace("\n", "\r\n")
    dst = Path(a.out) if a.out else src
    dst.write_bytes(patched.encode("utf-8"))
    print(f"patched -> {dst}")
