import json
import re
from pathlib import Path

from pdfminer.high_level import extract_text

def extract_pdf_text(pdf_file_path: Path) -> str:
    if not pdf_file_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_file_path}")
    return extract_text(str(pdf_file_path))


def build_quick_summary(text: str) -> dict:
    keywords = {
        "CVM": r"\bCVM\b",
        "computer_vision": r"\bcomputer[\s-]?vision\b",
        "vision": r"\bvision\b",
        "multimodal": r"\bmultimodal\b",
        "image": r"\bimages?\b",
        "video": r"\bvideos?\b",
        "customer_value": r"\bcustomer value\b",
        "retrieval": r"\bretrieval\b",
        "grounding": r"\bground(ing|ed)\b",
        "reasoning": r"\breason(ing)?\b",
    }
    counts = {
        key: len(re.findall(pattern, text, flags=re.IGNORECASE))
        for key, pattern in keywords.items()
    }

    # Heuristic heading candidates: short, title-case or UPPER
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    candidate_heads = [
        ln
        for ln in lines
        if (len(ln.split()) <= 8 and (ln.isupper() or ln.istitle()))
    ]
    seen = set()
    sample_headings = []
    for h in candidate_heads:
        if h not in seen:
            sample_headings.append(h)
            seen.add(h)
        if len(sample_headings) >= 25:
            break

    def extract_context(term_pattern: str, max_samples: int = 6):
        samples = []
        for m in re.finditer(term_pattern, text, flags=re.IGNORECASE):
            start_sentence = max(0, text.rfind(".", 0, m.start()))
            end_sentence = text.find(".", m.end())
            if end_sentence == -1:
                end_sentence = len(text)
            snippet = text[start_sentence + 1 : end_sentence + 1].strip()
            if snippet:
                samples.append(snippet)
            if len(samples) >= max_samples:
                break
        return samples

    contexts = {
        "vision": extract_context(r"\bvision\b"),
        "multimodal": extract_context(r"\bmultimodal\b"),
        "image": extract_context(r"\bimages?\b"),
        "video": extract_context(r"\bvideos?\b"),
        "grounding": extract_context(r"\bground(ing|ed)\b"),
        "retrieval": extract_context(r"\bretrieval\b"),
        "reasoning": extract_context(r"\breason(ing)?\b"),
    }

    return {
        "chars": len(text),
        "words": len(text.split()),
        "keywords": counts,
        "sample_headings": sample_headings,
        "contexts": contexts,
    }


def main() -> None:
    pdf_path = Path("Google Gemini.pdf")
    text = extract_pdf_text(pdf_path)

    # Save full text for reference
    Path("Google Gemini.txt").write_text(text, encoding="utf-8", errors="ignore")

    summary = build_quick_summary(text)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


