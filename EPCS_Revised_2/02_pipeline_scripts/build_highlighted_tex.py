"""Build EPCS_revise3_HIGHLIGHTED.tex: new/changed prose wrapped in \\hl{} (soul), changed
tables/figures/equations get a red '[Revised in this version]' line (never wrapped).
Baseline = OLD_ROUNDS_1_2/06_manuscript_revision/EPCS_revise2.tex (contains Table 13 with 0.8371 the reviewers cite;
revise1 also contains the string but revise2 is the last submitted version).
Usage: python build_highlighted_tex.py   (rerunnable; read-only on sources)"""
import re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "OLD_ROUNDS_1_2" / "06_manuscript_revision" / "EPCS_revise2.tex"
NEW = ROOT / "07_revision3" / "05_manuscript" / "EPCS_revise3.tex"
OUT = ROOT / "07_revision3" / "05_manuscript" / "package" / "EPCS_revise3_HIGHLIGHTED.tex"
MARK = r"\par\noindent{\color{red}\small[Revised in this version]}\par"
SOUL = "\\usepackage{soul}\n\\sethlcolor{yellow}\n"
ENVS = r"table\*?|figure\*?|equation\*?|align\*?|eqnarray\*?"
ENV_RE = re.compile(r"(?m)^[ \t]*\\begin\{(%s)\}.*?^[ \t]*\\end\{\1\}[ \t]*$" % ENVS, re.S)
STRUCT = re.compile(r"^\s*\\(section|subsection|subsubsection|title|author|date|maketitle|caption|label|bibitem|keywords|begin\{abstract\}|end\{abstract\})")
RECURSE = {"textbf", "textit"}   # \emph is forbidden inside \hl, so it is protected, not recursed
NOHL = re.compile(r"\\(?:cite[a-z]*|ref|eqref|url|texttt|emph|href|label|footnote)\b|\$|\\begin|\\end|\\\\")


def norm(s):
    s = re.sub(r"(?m)(?<!\\)%.*$", "", s)
    return re.sub(r"\s+", " ", s).strip()


def norm_env(s):
    return norm(s).replace("figures/", "")


def braced(s, i):
    """s[i]=='{' -> index after the matching '}'"""
    d = 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            d += 1
        elif c == "}":
            d -= 1
            if d == 0:
                return i + 1
        i += 1
    return len(s)


def pieces(s):
    """Split into ('t',text) / ('p',protected) / ('r',cmd,inner) items."""
    out, buf, i = [], [], 0

    def flush():
        if buf:
            out.append(("t", "".join(buf)))
            buf.clear()

    while i < len(s):
        c = s[i]
        if c == "%":
            j = s.find("\n", i)
            j = len(s) if j < 0 else j
            flush(); out.append(("p", s[i:j])); i = j
        elif c == "$":
            j = s.find("$", i + 1)
            j = len(s) - 1 if j < 0 else j
            flush(); out.append(("p", s[i:j + 1])); i = j + 1
        elif c == "\\":
            m = re.match(r"\\([a-zA-Z]+)\*?", s[i:])
            if m:
                name = m.group(1)
                j = i + m.end()
                if name in RECURSE and j < len(s) and s[j] == "{":
                    e = braced(s, j)
                    flush(); out.append(("r", "\\" + name, s[j + 1:e - 1])); i = e
                    continue
                while j < len(s) and s[j] in "[{":
                    if s[j] == "{":
                        j = braced(s, j)
                    else:
                        k = s.find("]", j)
                        j = len(s) if k < 0 else k + 1
                flush(); out.append(("p", s[i:j])); i = j
            else:
                nxt = s[i:i + 2]
                if nxt in ("\\%", "\\&"):
                    buf.append(nxt)
                else:
                    flush(); out.append(("p", nxt))
                i += 2
        elif c in "{}~":
            flush(); out.append(("p", c)); i += 1
        else:
            buf.append(c); i += 1
    flush()
    return out


def wrap_text(t, old_all, ctx_ok):
    parts = re.split(r"((?<=[.!?])\s+)", t)
    chunks = parts[0::2]
    seps = parts[1::2] + [""]
    chg = []
    for c in chunks:
        n = norm(c)
        if not re.search(r"[A-Za-z0-9]", n):
            chg.append(False)
        elif len(chunks) == 1 and len(n) < 30:
            chg.append(not ctx_ok(n))
        else:
            chg.append(n not in old_all)
    res, k = [], 0
    while k < len(chunks):
        if not chg[k]:
            res.append(chunks[k] + seps[k]); k += 1
            continue
        j = k
        while j + 1 < len(chunks) and chg[j + 1]:
            j += 1
        seg = "".join(chunks[x] + (seps[x] if x < j else "") for x in range(k, j + 1))
        lead = seg[:len(seg) - len(seg.lstrip())]
        trail = seg[len(seg.rstrip()):]
        res.append(lead + "\\hl{" + seg.strip() + "}" + trail + seps[j])
        k = j + 1
    return "".join(res)


def hl_para(s, old_all):
    items = pieces(s)
    out = []
    for idx, it in enumerate(items):
        if it[0] == "p":
            out.append(it[1])
        elif it[0] == "r":
            out.append(it[1] + "{" + hl_para(it[2], old_all) + "}")
        else:
            prev = items[idx - 1][1][-25:] if idx and items[idx - 1][0] == "p" else ""
            nxt = items[idx + 1][1][:25] if idx + 1 < len(items) and items[idx + 1][0] == "p" else ""
            ok = lambda n, a=prev, b=nxt: norm(a + " " + n + " " + b) in old_all or norm(a + n + b) in old_all
            out.append(wrap_text(it[1], old_all, ok))
    return "".join(out)


def hl_chunk(chunk, old_all):
    """Paragraph = consecutive non-blank, non-comment, non-structural lines."""
    out, para = [], []

    def fl():
        if para:
            out.append(hl_para("".join(para), old_all))
            para.clear()

    for ln in chunk.splitlines(keepends=True):
        if not ln.strip() or ln.lstrip().startswith("%") or STRUCT.match(ln):
            fl(); out.append(ln)
        else:
            para.append(ln)
    fl()
    return "".join(out)


def build(old_txt, new_txt):
    old_env = {norm_env(m.group(0)) for m in ENV_RE.finditer(old_txt)}
    old_all = norm(old_txt)
    bd = new_txt.find("\\begin{document}")
    bib = new_txt.find("\\begin{thebibliography}")
    head, body, tail = new_txt[:bd], new_txt[bd:bib], new_txt[bib:]
    head = head.replace("\\usepackage{xcolor}\n", "\\usepackage{xcolor}\n" + SOUL, 1)
    assert "usepackage{soul}" in head, "xcolor anchor not found"
    res, pos, marked = [], 0, []
    for m in ENV_RE.finditer(body):
        res.append(hl_chunk(body[pos:m.start()], old_all))
        if norm_env(m.group(0)) not in old_env:
            lab = re.search(r"\\label\{([^}]+)\}", m.group(0))
            marked.append((m.group(1), lab.group(1) if lab else "-"))
            res.append(MARK + "\n")
        res.append(m.group(0))
        pos = m.end()
    res.append(hl_chunk(body[pos:], old_all))
    return head + "".join(res) + tail, marked


def hl_spans(t):
    spans, i = [], 0
    while True:
        i = t.find("\\hl{", i)
        if i < 0:
            return spans
        e = braced(t, i + 3)
        spans.append((i, t[i + 4:e - 1]))
        i = e


def counts(t):
    b = re.sub(r"(?m)(?<!\\)%.*$", "", t)
    d = {k: len(re.findall(r"\\begin\{%s\}" % re.escape(k), b)) for k in
         ["table", "table*", "figure", "figure*", "equation", "align", "tabular", "itemize", "enumerate"]}
    d["includegraphics"] = len(re.findall(r"\\includegraphics", b))
    d["label"] = len(re.findall(r"\\label\{", b))
    d["bibitem"] = len(re.findall(r"\\bibitem\{", b))
    d["cite"] = len(re.findall(r"\\cite[a-z]*\{", b))
    d["ref"] = len(re.findall(r"\\(?:eq)?ref\{", b))
    return d


def verify(clean, hlt):
    ok = True
    cc, ch = counts(clean), counts(hlt)
    print("%-16s %6s %6s" % ("count", "clean", "HL"))
    for k in cc:
        flag = "" if cc[k] == ch[k] else "  <-- MISMATCH"
        ok &= cc[k] == ch[k]
        print("%-16s %6d %6d%s" % (k, cc[k], ch[k], flag))
    labs = lambda t: re.findall(r"\\label\{([^}]+)\}", t)
    bibs = lambda t: re.findall(r"\\bibitem\{([^}]+)\}", t)
    same = labs(clean) == labs(hlt) and bibs(clean) == bibs(hlt)
    ok &= same
    print("labels/bibitems identical sequence:", same)
    spans = hl_spans(hlt)
    bad = []
    for pos, s in spans:
        line = hlt.count("\n", 0, pos) + 1
        if re.search(r"(?<!\\)%", s): bad.append((line, "unescaped %"))
        m = NOHL.search(s)
        if m: bad.append((line, "forbidden: " + m.group(0)))
        if s.count("{") != s.count("}"): bad.append((line, "unbalanced braces"))
        if s.count("\\begin") != s.count("\\end"): bad.append((line, "unbalanced begin/end"))
        if "\\hl{" in s: bad.append((line, "nested hl"))
    print("hl spans: %d, violations: %d" % (len(spans), len(bad)))
    for b in bad[:20]:
        print("   line %d: %s" % b)
    ok &= not bad
    rt = hlt.replace(SOUL, "").replace(MARK + "\n", "")
    rt2, i = "", 0
    for pos, s in hl_spans(rt):
        rt2 += rt[i:pos] + s
        i = pos + 4 + len(s) + 1
    rt2 += rt[i:]
    rtok = rt2 == clean
    ok &= rtok
    print("round-trip (strip \\hl, markers, soul) == clean:", rtok)
    print("VERIFY:", "PASS" if ok else "FAIL")
    return ok


def main():
    old = OLD.read_text(encoding="utf-8")
    new = NEW.read_text(encoding="utf-8")
    hlt, marked = build(old, new)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(hlt, encoding="utf-8", newline="")
    print("baseline:", OLD.name, "| new:", NEW.name, "\nwrote", OUT)
    print("hl spans:", len(hl_spans(hlt)), "| env blocks marked revised:", len(marked))
    for env, lab in marked:
        print("   marked", env, lab)
    return 0 if verify(new, hlt) else 1


if __name__ == "__main__":
    sys.exit(main())
