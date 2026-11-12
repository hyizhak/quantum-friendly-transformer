"""Create a review copy replacing only response bodies for Comments 4 and 8."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "response.tex"
    original = source.read_text()
    draft = original
    for number in (4, 8):
        start = draft.index(r"\subsection*{Comment " + str(number) + "}")
        marker = r"\textbf{Response}\nopagebreak[4]"
        body_start = draft.index(marker, start) + len(marker)
        next_comment = draft.index(r"\subsection*{Comment " + str(number + 1) + "}", body_start)
        body_end = draft.rfind(r"\Needspace{6\baselineskip}", body_start, next_comment)
        if body_end < body_start:
            raise ValueError("Could not locate response boundary")
        snippet = (ROOT / f"results/reviewer/release/comment{number}_response.tex").read_text()
        draft = draft[:body_start] + "\n\n" + snippet.strip() + "\n\n\n" + draft[body_end:]
    output = ROOT / "results/reviewer/release/response_comments_4_8_draft.tex"
    output.write_text(draft)
    report = {
        "original_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "draft_sha256": hashlib.sha256(draft.encode()).hexdigest(),
        "replaced_response_bodies": [4, 8],
        "original_unchanged": source.read_text() == original,
        "latex_compilation": "Not run; no pdflatex executable available",
        "publication_status": "Draft; Comment 8 still requires a published deposit and updated DOI",
    }
    (output.parent / "response_draft_provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
