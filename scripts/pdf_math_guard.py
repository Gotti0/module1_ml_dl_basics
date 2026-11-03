"""LLM이 복원한 쪽을 PDF 글리프와 대조한다. 판정 기준은 초안이 아니라 PDF이다."""

import json
import re
import subprocess
import unicodedata
from collections import Counter
from pathlib import Path

import pymupdf

ROOT = Path(__file__).resolve().parent.parent
MATH_RE = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$", re.S)
HANGUL_RE = re.compile(r"[가-힣]")
# KaTeX는 렌더링 때 일부 기호를 다른 코드포인트로 바꾼다. PDF 쪽 추출과 같은 형태로 맞춘다.
FOLD = str.maketrans({"⁡": "", "⁢": "", "⁣": "", "−": "-", "∗": "*", "′": "'", "′": "'"})


def fold(text):
    # `½`는 NFKC에서 분수 빗금을 낀 `1⁄2`가 된다. \frac{1}{2}와 같은 글자로 세려고 빗금을 뺀다.
    return unicodedata.normalize("NFKC", text.translate(FOLD)).replace("⁄", "")


def glyph_counter(text):
    return Counter(c for c in fold(text) if not c.isspace() and c != "​")


def pdf_page_glyphs(page, footer_top=745):
    """쪽을 KaTeX 글리프와 그 밖의 글자로 나눈다. footer_top 아래는 꼬리글이다."""
    katex, other = Counter(), Counter()
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line["spans"]:
                if span["bbox"][1] >= footer_top:
                    continue
                # KaTeX 글꼴에 없는 `½`는 대체 글꼴로 그려지지만 수식의 일부이다.
                target = katex if "KaTeX" in span["font"] or "½" in span["text"] else other
                target.update(glyph_counter(span["text"]))
    return katex, other


def squeeze(text):
    return re.sub(r"[^가-힣A-Za-z0-9]", "", text)


def position_flags(page, markdown, footer_top=745):
    """독립 수식 앞에 놓인 본문 줄 수를 PDF의 세로 좌표와 비교한다.

    같은 문구가 반복되면 잘못 표시될 수 있어 실패로 치지 않고 사람이 볼 목록으로만 쓴다.
    """
    body_lines, math_boxes = [], []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            if line["bbox"][1] >= footer_top:
                continue
            body = squeeze("".join(s["text"] for s in line["spans"] if "KaTeX" not in s["font"]))
            if body:
                body_lines.append((line["bbox"][1], line["bbox"][3], body))
            if any("KaTeX" in s["font"] and s["text"].strip() for s in line["spans"]):
                math_boxes.append(line["bbox"])
    display = []
    for box in sorted(math_boxes, key=lambda b: b[1]):
        middle = (box[1] + box[3]) / 2
        if any(top < middle < bottom for top, bottom, _ in body_lines):
            continue
        if display and box[1] < display[-1][1] + 6:
            display[-1] = (display[-1][0], max(display[-1][1], box[3]))
        else:
            display.append((box[1], box[3]))
    squeezed, formulas = "", []
    for part in re.split(r"(\$\$.*?\$\$)", markdown, flags=re.S):
        if part.startswith("$$"):
            formulas.append((len(squeezed), part))
        else:
            squeezed += squeeze(re.sub(r"\$[^$\n]+\$", " ", part))
    located = [(bottom, squeezed.find(text)) for _, bottom, text in body_lines if len(text) >= 6 and text in squeezed]
    flags = []
    for (offset, tex), (top, _) in zip(formulas, display):
        expected = sum(1 for bottom, _ in located if bottom <= top + 2)
        actual = sum(1 for _, at in located if at < offset)
        if expected != actual:
            flags.append(tex[:60])
    return flags


def split_math(markdown):
    formulas = [a or b for a, b in MATH_RE.findall(markdown)]
    body = MATH_RE.sub(" ", markdown)
    return formulas, body


def render_glyphs(formulas):
    proc = subprocess.run(
        ["node", str(ROOT / "scripts" / "katex_check.mjs")],
        input=json.dumps(formulas, ensure_ascii=False),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return json.loads(proc.stdout)


def check_page(page, markdown):
    """쪽 하나의 결과를 돌려준다. ok는 파싱과 수식 기호가 모두 맞을 때만 참이다."""
    formulas, body = split_math(markdown)
    results = render_glyphs(formulas) if formulas else []
    parse_errors = [
        {"tex": tex, "error": r["error"]} for tex, r in zip(formulas, results) if not r["ok"]
    ]
    rendered = Counter()
    for r in results:
        if r["ok"]:
            rendered.update(glyph_counter(r["glyphs"]))
    rendered_hangul = Counter({c: n for c, n in rendered.items() if HANGUL_RE.fullmatch(c)})
    rendered_math = rendered - rendered_hangul

    katex, other = pdf_page_glyphs(page)
    # 표의 `R² < 0` 같은 부등호를 태그로 오인하지 않도록 실제 태그 형태만 지운다.
    body_clean = re.sub(r"<!--.*?-->|</?[a-z]+>|[#*_`|]|^\s*-{3,}\s*$", " ", body, flags=re.S | re.M)
    body_hangul = Counter(c for c in body_clean if HANGUL_RE.fullmatch(c))
    pdf_hangul = Counter({c: n for c, n in other.items() if HANGUL_RE.fullmatch(c)})

    math_missing = katex - rendered_math
    math_extra = rendered_math - katex
    hangul_diff = (pdf_hangul - body_hangul - rendered_hangul) + (body_hangul + rendered_hangul - pdf_hangul)
    return {
        "formulas": len(formulas),
        "parse_errors": parse_errors,
        "math_missing": dict(math_missing),
        "math_extra": dict(math_extra),
        "hangul_diff": dict(hangul_diff),
        "unreadable": markdown.count("<!-- math unreadable -->"),
        "position_flags": position_flags(page, markdown),
        "ok": not parse_errors and not math_missing and not math_extra and not hangul_diff,
    }
