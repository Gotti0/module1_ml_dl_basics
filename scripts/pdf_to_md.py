"""교안 PDF를 수식(KaTeX)과 도표(Mermaid)를 보존한 Markdown으로 변환한다.

draft  : PDF에서 쪽별 초안과 LLM 입력(쪽 이미지, 원본 이미지)을 만든다.
apply  : LLM 결과를 PDF와 대조해 반영하고 문서를 완성한다.
verify : 보고서와 사람이 볼 검토 페이지를 만든다.

LLM 호출은 이 스크립트 밖에서 하고 결과만 docs/notion-md/_math, _vision에서 읽는다.
"""

import argparse
import collections
import difflib
import hashlib
import html
import io
import json
import re
import warnings
from pathlib import Path

import pymupdf
import pymupdf4llm
from kiwipiepy import Kiwi
from PIL import Image

from pdf_math_guard import check_page

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "docs" / "notion"
OUT_DIR = ROOT / "docs" / "notion-md"
MATH_DIR = OUT_DIR / "_math"
VISION_DIR = OUT_DIR / "_vision"
BUILD_DIR = ROOT / ".cache" / "build"
REVIEW_DIR = ROOT / ".cache" / "review"

# 꼬리글(쪽 번호)이 놓이는 y 좌표. Chromium 인쇄물이라 모든 쪽이 같다.
FOOTER_TOP = 745
MAX_IMAGE_SIDE = 1600
MATH_DPI = 170
# 이어 붙인 점수가 이 값을 넘을 때만 합친다. 차이가 0에 가까우면 형태소 분석이 판단을 못 한 것이다.
JOINT_MIN_GAIN = 1.0

PARTICLES = (
    "은|는|이|가|을|를|의|에|에서|에서는|에는|에도|에게|에만|로|으로|로서|와|과|도|만|만큼|라는|이라는|"
    "라고|이라고|입니다|이며|이고|일|니다|습니다|합니다|해야|처럼|보다|까지|부터|마다|을까|을까요|인|인가|인가요"
)
PROTECTED_RE = re.compile(
    r"```.*?```|\$\$.*?\$\$|\$[^$\n]+\$|`[^`\n]+`|<!--.*?-->", re.S
)
SPACE_BEFORE_PUNCT_RE = re.compile(r"(?<=[^\s|]) +(?=[.,:;?!)\]](?:\s|$|\*|))", re.M)
SPACE_AFTER_OPEN_RE = re.compile(r"(?<=[(\[]) +(?=\S)")
SPACE_BEFORE_PARTICLE_RE = re.compile(
    rf"(?<=[A-Za-z0-9)\]%*]) +(?=(?:{PARTICLES})(?:[\s.,:;?!)*]|$))", re.M
)
SPACE_BEFORE_BOLD_PARTICLE_RE = re.compile(rf"(?<=[A-Za-z0-9]) +(?=\*\*(?:{PARTICLES})[\s*])")
HANGUL_RE = re.compile(r"[가-힣]")
IMAGE_TOKEN_RE = re.compile(r"<!-- image: ([0-9a-f]{64}) -->")
PICTURE_MARK_RE = re.compile(r"<!-- (?:Start|End) of picture text -->")
GARBLED_CHARS = set("�]LMʼ")
# `-3`이 `]3`으로, `(-4)`가 `L]4M`으로 읽힌 흔적. 수식 안은 보지 않도록 앞 글자를 제한한다.
GARBLED_MINUS_RE = re.compile(r"(?<![\\$^_{(\[A-Za-z0-9])\][\d.]+")


def doc_id(pdf):
    return hashlib.sha1(pdf.stem.encode("utf-8")).hexdigest()[:8]


def out_path(pdf):
    day = re.match(r"Day_(\d)", pdf.stem).group(1)
    name = re.sub(r"^Day_\d[._]*", "", pdf.stem)
    name = re.sub(r"_완$", "", name)
    name = re.sub(r"[\s_.]+", "-", name).strip("-")
    return OUT_DIR / f"day{day}" / f"{name}.md"


def select_pdfs(only):
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    return [p for p in pdfs if not only or any(o in p.stem for o in only)]


def has_katex(page):
    return any("KaTeX" in f[3] for f in page.get_fonts())


def box_spans(page, bbox):
    rect = pymupdf.Rect(bbox)
    return [
        span
        for block in page.get_text("dict", clip=rect)["blocks"]
        for line in block.get("lines", [])
        for span in line["spans"]
        if span["text"].strip()
    ]


def body_size(doc):
    sizes = collections.Counter()
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    sizes[round(span["size"], 1)] += len(span["text"])
    return sizes.most_common(1)[0][0]


def heading_levels(doc, chunks, base):
    """pymupdf4llm의 제목 수준은 쪽마다 흔들리므로 글자 크기 순위로 문서 전체를 다시 매긴다."""
    sizes = set()
    for page, chunk in zip(doc, chunks):
        for box in chunk["page_boxes"]:
            if box["class"] != "section-header":
                continue
            spans = box_spans(page, box["bbox"])
            if spans and not any("KaTeX" in s["font"] for s in spans):
                size = max(round(s["size"], 1) for s in spans)
                if size > base + 0.3:
                    sizes.add(size)
    return {size: min(i + 1, 4) for i, size in enumerate(sorted(sizes, reverse=True))}


def fix_heading(text, page, bbox, levels, base):
    spans = box_spans(page, bbox)
    title = re.sub(r"^#+\s*", "", text.strip()).replace("**", "").strip()
    if not spans or any("KaTeX" in s["font"] for s in spans):
        # 수식 조각이 제목으로 분류된 경우다. 수식 복원 단계가 처리하도록 제목 표시만 뗀다.
        return f"{title}\n\n"
    size = max(round(s["size"], 1) for s in spans)
    if size <= base + 0.3:
        return f"**{title}**\n\n"
    return f"{'#' * levels[size]} {title}\n\n"


def wrap_joints(page):
    """줄바꿈으로 갈라졌을 수 있는 (앞 단어, 뒤 단어) 쌍. PDF에는 공백 유무의 단서가 없다."""
    joints = []
    for block in page.get_text("dict")["blocks"]:
        lines = block.get("lines", [])
        for upper, lower in zip(lines, lines[1:]):
            head = "".join(s["text"] for s in upper["spans"]).rstrip()
            tail = "".join(s["text"] for s in lower["spans"]).lstrip()
            if (
                head and tail
                and HANGUL_RE.match(head[-1]) and HANGUL_RE.match(tail[0])
                and lower["bbox"][1] > upper["bbox"][1] + 3
            ):
                joints.append([head.split()[-1], tail.split()[0]])
    return joints


def build_page(page, chunk, levels, base, images):
    """상자 위치를 뒤에서부터 고쳐 앞쪽 상자의 오프셋이 밀리지 않게 한다."""
    text = chunk["text"]
    rasters = [
        (pymupdf.Rect(info["bbox"]), info["xref"])
        for info in page.get_image_info(xrefs=True)
        if info["xref"]
    ]
    placed = set()
    boxes = sorted(chunk["page_boxes"], key=lambda b: b["pos"][0], reverse=True)
    for box in boxes:
        start, stop = box["pos"]
        if box["class"] == "section-header":
            text = text[:start] + fix_heading(text[start:stop], page, box["bbox"], levels, base) + text[stop:]
        elif box["class"] == "picture":
            area = pymupdf.Rect(box["bbox"])
            inside = [
                (rect, xref)
                for rect, xref in rasters
                if xref not in placed and (rect & area).get_area() > 0.6 * rect.get_area()
            ]
            if inside:
                inside.sort(key=lambda item: item[0].y0)
                placed.update(xref for _, xref in inside)
                tokens = "\n\n".join(f"<!-- image: {images[xref]} -->" for _, xref in inside)
                # OCR을 껐으므로 상자 안의 글자는 PDF 글자층의 실제 내용이다. 버리면 본문이 사라진다.
                kept = PICTURE_MARK_RE.sub("", text[start:stop]).replace("<br>", "\n").strip()
                text = text[:start] + tokens + "\n\n" + (kept + "\n\n" if kept else "") + text[stop:]
    unplaced = [xref for _, xref in rasters if xref not in placed]
    return restore_dropped(repair_garbled(text, page), page), [images[x] for x in unplaced]


def restore_dropped(text, page):
    """pymupdf4llm이 드물게 버리는 줄을 되살린다. 줄의 글자가 모두 초안에서 비어 있을 때만 넣는다."""
    lines = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = [s for s in line["spans"] if s["bbox"][1] < FOOTER_TOP and "KaTeX" not in s["font"]]
            joined = "".join(s["text"] for s in spans).strip()
            if joined:
                lines.append((line["bbox"][1], joined))
    lines.sort(key=lambda item: item[0])
    expected = collections.Counter(c for _, line in lines for c in line if c.isalnum())
    missing = expected - collections.Counter(c for c in text if c.isalnum())
    anchor = None
    for _, line in lines:
        squeezed = re.sub(r"\s", "", line)
        need = collections.Counter(c for c in line if c.isalnum())
        if need and squeezed not in re.sub(r"\s", "", text) and not need - missing:
            at = len(text)
            if anchor:
                found = re.search(r"\s*".join(map(re.escape, anchor)), text)
                if found:
                    newline = text.find("\n", found.end())
                    at = newline if newline >= 0 else len(text)
            text = f"{text[:at]}\n\n{line}\n\n{text[at:].lstrip(chr(10))}"
            missing -= need
        anchor = squeezed[-12:]
    return text


def repair_garbled(text, page):
    """pymupdf4llm의 글자 오류를 같은 쪽의 일반 텍스트 추출과 맞춰 되돌린다.

    일부 Type3 글꼴의 기호를 잘못 읽고, 드물게 한글을 겹쳐 쓴다(`첫첫 번째째`).
    """
    reference = page.get_text()
    # 표 칸의 구분자와 줄바꿈이 끼어 글자 정렬로는 맞출 수 없으므로 흔적을 직접 되돌린다. 원본에 같은 글자가 있으면 진짜 글자이다.
    text = GARBLED_MINUS_RE.sub(lambda m: m.group(0) if m.group(0) in reference else "-" + m.group(0)[1:], text)
    text = re.sub(r"L\](\d)", r"(-\1", text)
    text = re.sub(r"\(-([\d.]+)M", r"(-\1)", text)
    duplicated = hangul_counter(text) - hangul_counter(reference)
    if "�" not in text and not duplicated:
        return text
    matcher = difflib.SequenceMatcher(None, text, reference, autojunk=False)
    repaired = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        source, target = text[i1:i2], reference[j1:j2]
        replaceable = tag == "replace" and (
            "�" in source or (len(source) == len(target) and set(source) <= GARBLED_CHARS)
        )
        if tag == "delete" and duplicated and HANGUL_RE.search(source):
            continue
        repaired.append(target if replaceable else source)
    result = "".join(repaired)
    # 글자를 지웠는데 겹침이 안 풀렸거나 본문이 줄었다면 정렬이 어긋난 것이므로 지운 결과를 믿지 않는다.
    if hangul_counter(result) != hangul_counter(text) - duplicated:
        print(f"  겹침 복구 실패: p{page.number + 1}")
        return text
    return result


def hangul_counter(text):
    return collections.Counter(HANGUL_RE.findall(text))


def save_images(doc, page):
    found = {}
    for info in page.get_image_info(xrefs=True):
        xref = info["xref"]
        if not xref:
            continue
        raw = doc.extract_image(xref)["image"]
        sha = hashlib.sha256(raw).hexdigest()
        found[xref] = sha
        target = BUILD_DIR / "images" / f"{sha}.png"
        if not target.exists():
            image = Image.open(io.BytesIO(raw)).convert("RGB")
            image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
            image.save(target)
    return found


def cmd_draft(args):
    (BUILD_DIR / "images").mkdir(parents=True, exist_ok=True)
    (BUILD_DIR / "math").mkdir(parents=True, exist_ok=True)
    for pdf in select_pdfs(args.only):
        did = doc_id(pdf)
        doc = pymupdf.open(pdf)
        chunks = pymupdf4llm.to_markdown(
            str(pdf), use_ocr=False, header=False, footer=False,
            write_images=False, page_chunks=True, show_progress=False,
        )
        base = body_size(doc)
        levels = heading_levels(doc, chunks, base)
        pages, unplaced, images, math_pages, joints = [], [], [], [], []
        for number, (page, chunk) in enumerate(zip(doc, chunks), 1):
            found = save_images(doc, page)
            images.extend(found.values())
            text, missing = build_page(page, chunk, levels, base, found)
            unplaced.extend(missing)
            pages.append(text)
            joints.extend(wrap_joints(page))
            if has_katex(page):
                math_id = f"{did}_p{number:02}"
                math_pages.append(number)
                (BUILD_DIR / "math" / f"{math_id}.md").write_text(text, encoding="utf-8")
                page.get_pixmap(dpi=MATH_DPI).save(BUILD_DIR / "math" / f"{math_id}.png")
        meta = {
            "pdf": pdf.name,
            "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
            "pages": len(doc),
            "out": str(out_path(pdf).relative_to(ROOT)),
            "math_pages": math_pages,
            "images": images,
            "unplaced_images": unplaced,
            "joints": joints,
            "drafts": pages,
        }
        (BUILD_DIR / f"{did}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        print(f"{pdf.stem[:40]}\tpages={len(doc)} math_pages={len(math_pages)} images={len(images)} unplaced={len(unplaced)}")


def clean_markup(text):
    # 이미 굵은 문장 안의 강조는 굵게로 또 감싸면 표시가 깨진다.
    text = re.sub(r"\*\*<mark>(.*?)</mark>", r"**\1", text, flags=re.S)
    text = re.sub(r"<mark>(.*?)</mark>\*\*", r"\1**", text, flags=re.S)
    text = re.sub(r"<mark>(.*?)</mark>", lambda m: "**" + m.group(1).replace("**", "") + "**", text, flags=re.S)
    text = re.sub(r"</?mark>", "", text)
    # 이미지가 아닌 그림 영역은 글상자나 글자로 그린 그림이다. 줄과 공백을 지키려고 코드 블록으로 둔다.
    text = re.sub(
        r"<!-- Start of picture text -->(.*?)<!-- End of picture text -->",
        lambda m: "```\n" + m.group(1).replace("<br>", "\n").strip("\n") + "\n```",
        text,
        flags=re.S,
    )
    text = re.sub(r"</?u>", "", text).replace("ʼ", "’")
    return re.sub(r"\n{3,}", "\n\n", text)


def last_block_is_table(text):
    return text.rstrip().rsplit("\n", 1)[-1].startswith("|")


def merge_pages(pages):
    """쪽 경계에서 끊긴 표와 코드 블록을 잇는다."""
    merged = ""
    for page in pages:
        page = page.strip("\n")
        if not page:
            continue
        tail = merged.rstrip()
        table = re.match(r"\|[^\n]*\n\|(?:\s*:?-+:?\s*\|)+\n", page)
        if tail.endswith("```") and page.startswith("```") and tail.count("```") % 2 == 0:
            merged = tail[:-3].rstrip() + "\n" + page.split("\n", 1)[1]
        elif table and last_block_is_table(tail):
            # 이어지는 표의 첫 행은 머리글이 아니라 본문 행이다. 구분선만 지운다.
            header = table.group(0).split("\n", 1)[0]
            merged = tail + "\n" + header + "\n" + page[table.end():]
        else:
            merged = (tail + "\n\n" + page) if tail else page
    return merged


def match_reference_spacing(text, page):
    """PDF 줄 안의 띄어쓰기는 원본 그대로이므로, 원본에 없는 공백을 지운다. 공백을 새로 넣지는 않는다."""
    marks = "*`"
    chars = [(i, c) for i, c in enumerate(text) if not c.isspace() and c not in marks]
    squeezed = "".join(c for _, c in chars)
    drop = set()
    for line in page.get_text().split("\n"):
        key = "".join(c for c in line if not c.isspace() and c not in marks)
        # 짧은 줄은 다른 곳과 우연히 일치할 수 있다.
        at = squeezed.find(key) if len(key) >= 4 else -1
        if at < 0:
            continue
        position, spaced = at, False
        for c in line.strip():
            if c.isspace():
                spaced = True
            elif c not in marks:
                if position > at and not spaced:
                    left, right = chars[position - 1][0], chars[position][0]
                    drop.update(i for i in range(left + 1, right) if text[i] == " ")
                position += 1
                spaced = False
    return "".join(c for i, c in enumerate(text) if i not in drop)


def fix_spacing(text, joints, kiwi):
    protected = []

    def stash(match):
        protected.append(match.group(0))
        return f"{len(protected) - 1}"

    text = PROTECTED_RE.sub(stash, text)
    text = SPACE_BEFORE_PUNCT_RE.sub("", text)
    text = SPACE_AFTER_OPEN_RE.sub("", text)
    text = SPACE_BEFORE_PARTICLE_RE.sub("", text)
    text = SPACE_BEFORE_BOLD_PARTICLE_RE.sub("", text)
    for head, tail in joints:
        head = re.search(r"[가-힣]+$", head)
        tail = re.match(r"[가-힣]+", tail)
        if not head or not tail:
            continue
        head, tail = head.group(0), tail.group(0)
        # 굵게 표시가 줄바꿈 자리에 끼어 있어도 같은 자리로 본다.
        pair = re.compile(rf"{head}(\*\*)? +(\*\*)?{tail}")
        if not pair.search(text):
            continue
        apart = kiwi.analyze(f"{head} {tail}", top_n=1)[0][1]
        joined = kiwi.analyze(head + tail, top_n=1)[0][1]
        if joined - apart > JOINT_MIN_GAIN:
            text = pair.sub(rf"{head}\1\2{tail}", text, count=1)
    # 공백을 지우면 이웃한 굵은 글자가 맞붙어 표시가 깨지므로 공백을 지운 뒤에 합친다.
    text = text.replace("****", "")
    return re.sub(r"(\d+)", lambda m: protected[int(m.group(1))], text)


def load_vision(sha):
    path = VISION_DIR / f"{sha}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def replace_images(text, pending, counts):
    def replace(match):
        sha = match.group(1)
        decision = load_vision(sha)
        if decision is None:
            pending.append(sha)
            return match.group(0)
        counts[decision["kind"]] += 1
        if decision["kind"] != "diagram":
            return f"<!-- image removed: {sha[:8]} -->"
        caption = decision.get("caption")
        block = f"```mermaid\n{decision['mermaid'].strip()}\n```"
        # 캡션은 줄 단위 글이다. 그대로 두면 Markdown이 한 문단으로 합쳐 버린다.
        return f"{block}\n\n" + caption.strip().replace("\n", "  \n") if caption else block

    return IMAGE_TOKEN_RE.sub(replace, text)


def cmd_apply(args):
    report = {}
    kiwi = Kiwi()
    for pdf in select_pdfs(args.only):
        did = doc_id(pdf)
        meta = json.loads((BUILD_DIR / f"{did}.json").read_text(encoding="utf-8"))
        doc = pymupdf.open(pdf)
        pages = list(meta["drafts"])
        math_failed, math_pending, position_flags, math_approved = {}, [], {}, []
        # 수식이 없는 쪽도 초안 결함을 손으로 고친 파일이 있으면 같은 검사를 거쳐 반영한다.
        for number in range(1, meta["pages"] + 1):
            result = MATH_DIR / f"{did}_p{number:02}.md"
            if not result.exists():
                if number in meta["math_pages"]:
                    math_pending.append(number)
                continue
            restored = result.read_text(encoding="utf-8")
            check = check_page(doc[number - 1], restored)
            # 수식 쪽에 든 이미지 토큰을 LLM이 지우면 도표가 조용히 사라지므로 함께 검사한다.
            check["image_tokens_kept"] = IMAGE_TOKEN_RE.findall(restored) == IMAGE_TOKEN_RE.findall(pages[number - 1])
            if check["position_flags"]:
                position_flags[number] = check["position_flags"]
            # 글리프가 일부만 KaTeX로 그려진 쪽은 자동 대조가 불가능하다. 사람이 원본과 대조한 경우에만 통과시킨다.
            approved = result.with_suffix(".approved").exists() and not check["parse_errors"]
            if approved:
                math_approved.append(number)
            if (check["ok"] or approved) and check["image_tokens_kept"]:
                pages[number - 1] = restored
            else:
                # 검사에 실패한 쪽은 반영하지 않고 초안을 유지해 사람이 볼 수 있게 남긴다.
                math_failed[number] = check
        # 초안을 다시 만들면 복원 결과와 어긋나므로, 남은 오염은 복원 결과 위에서 고친다.
        pages = [match_reference_spacing(repair_garbled(text, page), page) for text, page in zip(pages, doc)]
        text = fix_spacing(clean_markup(merge_pages(pages)), meta["joints"], kiwi)
        pending, counts = [], collections.Counter()
        text = replace_images(text, pending, counts)
        front = (
            f"---\nsource_pdf: \"{meta['pdf']}\"\nsource_sha256: {meta['sha256']}\n"
            f"source_pages: {meta['pages']}\n---\n\n"
        )
        target = ROOT / meta["out"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(front + text.strip() + "\n", encoding="utf-8")
        report[meta["pdf"]] = {
            "out": meta["out"],
            "math_pages": len(meta["math_pages"]),
            "math_pending": math_pending,
            "math_failed": math_failed,
            "position_flags": position_flags,
            "math_approved": math_approved,
            "images": len(meta["images"]),
            "unplaced_images": meta["unplaced_images"],
            "vision_pending": sorted(set(pending)),
            "diagrams": counts["diagram"],
            "images_removed": counts["image"],
        }
        print(
            f"{pdf.stem[:40]}\tmath ok={len(meta['math_pages']) - len(math_failed) - len(math_pending)}"
            f" fail={len(math_failed)} pending={len(math_pending)}"
            f" diagram={counts['diagram']} removed={counts['image']} vision_pending={len(pending)}"
        )
    (OUT_DIR / "_apply_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


def pdf_chars(doc):
    chars = collections.Counter()
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    if span["bbox"][1] < FOOTER_TOP and "KaTeX" not in span["font"]:
                        chars.update(c for c in span["text"] if c.isalnum())
    return chars


def strip_generated(text):
    """캡션과 Mermaid는 PDF 글자층에 없는 이미지 속 글이므로 대조에서 뺀다."""
    text = text.split("---\n", 2)[-1]
    captions = {
        decision["caption"].strip().replace("\n", "  \n")
        for path in VISION_DIR.glob("*.json")
        for decision in [json.loads(path.read_text(encoding="utf-8"))]
        if decision.get("caption")
    } if VISION_DIR.exists() else set()
    text = re.sub(r"```mermaid.*?```", "", text, flags=re.S)
    for caption in captions:
        text = text.replace(caption, "")
    text = re.sub(r"\$\$.*?\$\$|\$[^$\n]+\$|<!--.*?-->", "", text, flags=re.S)
    return text


def review_page(report):
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    order = {"low": 0, "medium": 1, "high": 2}
    decisions = []
    for path in VISION_DIR.glob("*.json") if VISION_DIR.exists() else []:
        decision = json.loads(path.read_text(encoding="utf-8"))
        decision["sha"] = path.stem
        decisions.append(decision)
    decisions.sort(key=lambda d: (order.get(d.get("confidence"), 3) != 0, d["kind"] != "diagram", d["sha"]))
    rows = []
    for d in decisions:
        mermaid = html.escape(d.get("mermaid") or "")
        rows.append(
            f'<section class="row" data-sha="{d["sha"]}"><img loading="lazy" src="../build/images/{d["sha"]}.png" alt="원본 이미지">'
            f'<div><h3>{html.escape(d["kind"])} · {html.escape(str(d.get("confidence")))} · {d["sha"][:8]}</h3>'
            f'<p>{html.escape(d.get("reason") or "")}</p>'
            + (f'<pre class="src" hidden>{mermaid}</pre><div class="out"></div>' if mermaid else "")
            + (f'<p class="cap">{html.escape(d["caption"])}</p>' if d.get("caption") else "")
            + "</div></section>"
        )
    math_rows = []
    failed_first = sorted(
        ((pdf, entry) for pdf, entry in report.items() if entry["math_pages"]),
        key=lambda item: -len(item[1]["math_failed"]),
    )
    for pdf, entry in failed_first:
        did = doc_id(PDF_DIR / pdf)
        meta = json.loads((BUILD_DIR / f"{did}.json").read_text(encoding="utf-8"))
        for number in meta["math_pages"]:
            source = MATH_DIR / f"{did}_p{number:02}.md"
            if not source.exists():
                continue
            status = "실패" if str(number) in entry["math_failed"] or number in entry["math_failed"] else "통과"
            math_rows.append(
                f'<section class="row"><img loading="lazy" src="../build/math/{did}_p{number:02}.png" alt="원본 쪽">'
                f'<div class="mathpage"><h3>{html.escape(pdf[:40])} p{number} · {status}</h3>'
                f'<pre class="mathsrc" hidden>{html.escape(source.read_text(encoding="utf-8"))}</pre><div class="mathout"></div></div></section>'
            )
    page = f"""<!doctype html><meta charset="utf-8"><title>교안 변환 검토</title>
<link rel="stylesheet" href="../node/node_modules/katex/dist/katex.min.css">
<style>body{{font:14px sans-serif;margin:16px}}.row{{display:grid;grid-template-columns:1fr 1fr;gap:16px;border-top:1px solid #ccc;padding:12px 0}}.row img{{max-width:100%}}.cap,.mathout{{white-space:pre-wrap}}.out svg{{max-width:100%}}</style>
<h1>교안 변환 검토</h1><pre id="summary">렌더링 중</pre>
<h2>도표와 이미지 ({len(decisions)})</h2>{''.join(rows)}
<h2>수식 쪽 ({len(math_rows)})</h2>{''.join(math_rows)}
<script src="../node/node_modules/mermaid/dist/mermaid.min.js"></script>
<script src="../node/node_modules/katex/dist/katex.min.js"></script>
<script>
const esc=s=>s.replace(/[&<>]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;'}}[c]));
document.querySelectorAll('.mathsrc').forEach(p=>{{
  const out=p.nextElementSibling;
  out.innerHTML=p.textContent.split(/(\\$\\$[\\s\\S]+?\\$\\$|\\$[^$\\n]+\\$)/).map(t=>{{
    const d=t.startsWith('$$');const m=d?t.slice(2,-2):(t.startsWith('$')&&t.length>1?t.slice(1,-1):null);
    if(m===null)return esc(t);
    try{{return katex.renderToString(m,{{displayMode:d,throwOnError:true}})}}catch(e){{return '<b style="color:red">'+esc(t)+'</b>'}}
  }}).join('');
}});
mermaid.initialize({{startOnLoad:false}});
(async()=>{{const failed=[];let i=0,total=0;
for(const row of document.querySelectorAll('[data-sha]')){{const src=row.querySelector('.src');if(!src)continue;total++;
  try{{await mermaid.parse(src.textContent);const r=await mermaid.render('m'+(i++),src.textContent);row.querySelector('.out').innerHTML=r.svg}}
  catch(e){{failed.push({{sha:row.dataset.sha,error:String(e).slice(0,300)}});row.querySelector('.out').textContent='파싱 실패: '+e}}}}
window.MERMAID_RESULT={{total,failed}};
document.getElementById('summary').textContent=JSON.stringify(window.MERMAID_RESULT);}})();
</script>"""
    (REVIEW_DIR / "index.html").write_text(page, encoding="utf-8")


def cmd_verify(args):
    apply_report = json.loads((OUT_DIR / "_apply_report.json").read_text(encoding="utf-8"))
    results = {}
    for pdf in select_pdfs(args.only):
        entry = apply_report.get(pdf.name)
        if entry is None:
            results[pdf.name] = {"error": "apply 결과 없음"}
            continue
        text = (ROOT / entry["out"]).read_text(encoding="utf-8")
        doc = pymupdf.open(pdf)
        expected = pdf_chars(doc)
        actual = collections.Counter(c for c in strip_generated(text) if c.isalnum())
        missing, extra = expected - actual, actual - expected
        total = sum(expected.values()) or 1
        results[pdf.name] = {
            "char_coverage": round(1 - sum(missing.values()) / total, 4),
            "chars_missing": sum(missing.values()),
            "chars_extra": sum(extra.values()),
            "garbled_chars": text.count("�"),
            "image_links": len(re.findall(r"!\[", text)),
            "image_tokens_left": len(IMAGE_TOKEN_RE.findall(text)),
            "fence_balanced": text.count("```") % 2 == 0,
            "unreadable_math": text.count("<!-- math unreadable -->"),
            **{k: entry[k] for k in ("math_pages", "math_pending", "images", "unplaced_images", "diagrams", "images_removed")},
            "math_failed_pages": sorted(int(k) for k in entry["math_failed"]),
        }
    (OUT_DIR / "_verify_report.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    review_page(apply_report)
    for name, r in results.items():
        print(name[:36], json.dumps({k: v for k, v in r.items() if k not in ("images",)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["draft", "apply", "verify"])
    parser.add_argument("--only", nargs="*", help="파일 이름에 이 문자열이 든 PDF만 처리한다")
    args = parser.parse_args()
    {"draft": cmd_draft, "apply": cmd_apply, "verify": cmd_verify}[args.command](args)


if __name__ == "__main__":
    main()
