"""Static quality gates for the public AI4S in Hand curriculum."""

from pathlib import Path
import re
import sys
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

CHAPTERS = [DOCS / "preface" / "index.md"]
CHAPTERS += [DOCS / "part1-ai-foundations" / f"chapter{i}.md" for i in range(1, 6)]
CHAPTERS += [DOCS / "part2-science-foundations" / f"chapter{i}.md" for i in range(6, 12)]
CHAPTERS += [DOCS / "part3-research-paradigm" / f"chapter{i}.md" for i in range(12, 20)]

PLACEHOLDERS = ["TBD", "TODO", "待补充", "这里写", "参考答案略", "第1章的标题", "repo-template"]
REQUIRED_CONCEPTS = {
    "problem_or_goal": ("本章解决的问题", "学习目标", "本章目标"),
    "prerequisites": ("前置知识", "前置与路线", "适合谁", "学习路线"),
    "artifact_or_experiment": ("本章产物", "实践", "实验", "可检查产物"),
    "failure_modes": ("失败", "常见错误", "误区"),
    "validity_check": ("有效性检查", "证据检查", "检查清单", "科学有效性"),
    "exercises": ("练习",),
    "references": ("参考文献", "原始来源", "延伸阅读", "参考资料"),
}

LINK_PATTERN = re.compile(r"(?<!!)\[[^\]]+\]\(([^)\s]+)(?:\s+['\"][^'\"]+['\"])?\)")
DOI_PATTERN = re.compile(r"(?<!doi\.org/)(?<![\w/])(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)")
FENCED_CODE_PATTERN = re.compile(
    r"^(?:```|~~~).*?^(?:```|~~~)\s*$", flags=re.MULTILINE | re.DOTALL
)


def strip_fenced_code(text: str) -> str:
    """Remove fenced examples before checking prose-level Markdown rules."""

    return FENCED_CODE_PATTERN.sub("", text)


def resolve_internal_link(source: Path, target: str) -> Path | None:
    target = unquote(target.split("#", 1)[0].split("?", 1)[0])
    if not target or target.startswith(("http://", "https://", "mailto:")):
        return None

    if target.startswith("/"):
        relative = target.lstrip("/")
        if relative.startswith(("images/", "datawhale-logo", "learning.GIF")):
            return DOCS / "public" / relative
        route = DOCS / relative
    else:
        route = (source.parent / target).resolve()

    if route.suffix:
        return route
    if target.endswith("/"):
        return route / "index.md"
    markdown_candidate = route.with_suffix(".md")
    if markdown_candidate.exists():
        return markdown_candidate
    return route / "index.md"


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    doi_count = 0
    external_link_count = 0

    for chapter in CHAPTERS:
        relative = chapter.relative_to(ROOT)
        if not chapter.exists():
            errors.append(f"missing chapter: {relative}")
            continue

        text = chapter.read_text(encoding="utf-8")
        prose = strip_fenced_code(text)
        h1_count = len(re.findall(r"^# [^#]", prose, flags=re.MULTILINE))
        if h1_count != 1:
            errors.append(f"{relative}: expected exactly one H1, found {h1_count}")
        if len(text) < 2500:
            warnings.append(f"{relative}: short draft ({len(text)} characters)")

        for placeholder in PLACEHOLDERS:
            if placeholder in prose:
                errors.append(f"{relative}: placeholder/template token {placeholder!r}")
        if re.search(r"\[\^[^\]]+\]", prose):
            errors.append(f"{relative}: unsupported Markdown footnote syntax")

        for concept, alternatives in REQUIRED_CONCEPTS.items():
            if not any(alternative in text for alternative in alternatives):
                errors.append(f"{relative}: missing chapter responsibility {concept}")

        bare_dois = DOI_PATTERN.findall(prose)
        if bare_dois:
            errors.append(f"{relative}: DOI not linked through https://doi.org/: {bare_dois[0]}")
        doi_count += prose.count("https://doi.org/")

        for match in LINK_PATTERN.finditer(prose):
            target = match.group(1)
            if target.startswith(("http://", "https://")):
                external_link_count += 1
                continue
            resolved = resolve_internal_link(chapter, target)
            if resolved is not None and not resolved.exists():
                errors.append(
                    f"{relative}: broken internal link {target!r} -> "
                    f"{resolved.relative_to(ROOT) if resolved.is_relative_to(ROOT) else resolved}"
                )

    public_pages = [ROOT / "README.md"]
    public_pages += [
        path
        for path in DOCS.rglob("*.md")
        if "superpowers" not in path.relative_to(DOCS).parts
    ]
    for page in public_pages:
        if page in CHAPTERS:
            continue
        relative = page.relative_to(ROOT)
        prose = strip_fenced_code(page.read_text(encoding="utf-8"))
        bare_dois = DOI_PATTERN.findall(prose)
        if bare_dois:
            errors.append(f"{relative}: DOI not linked through https://doi.org/: {bare_dois[0]}")
        doi_count += prose.count("https://doi.org/")
        for match in LINK_PATTERN.finditer(prose):
            target = match.group(1)
            if target.startswith(("http://", "https://")):
                external_link_count += 1
                continue
            resolved = resolve_internal_link(page, target)
            if resolved is not None and not resolved.exists():
                errors.append(
                    f"{relative}: broken internal link {target!r} -> "
                    f"{resolved.relative_to(ROOT) if resolved.is_relative_to(ROOT) else resolved}"
                )

    print(
        f"chapters={len(CHAPTERS)}, doi_links={doi_count}, "
        f"external_links={external_link_count}, warnings={len(warnings)}, errors={len(errors)}"
    )
    for warning in warnings:
        print(f"WARN {warning}")
    for error in errors:
        print(f"ERROR {error}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
