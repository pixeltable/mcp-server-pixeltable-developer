"""Guard against deprecated Pixeltable / skill anti-patterns leaking into prompts or docs.

If you intentionally need to demonstrate a banned pattern (e.g. a warning block),
mark it in code with the inline allowlist token ``# allow: <pattern>`` on the
same line; the regex check below ignores those lines.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROMPT_FILE = Path(__file__).resolve().parents[1] / "src" / "mcp_server_pixeltable_stio" / "prompt.py"

# Each entry: (regex, friendly name, recommended replacement).
BANNED_PATTERNS = [
    (r"openai\.vision", "openai.vision", "openai.chat_completions with image_url blocks"),
    (
        r"from\s+pixeltable\.iterators\s+import\s+FrameIterator",
        "pixeltable.iterators.FrameIterator",
        "from pixeltable.functions.video import frame_iterator",
    ),
    (
        r"FrameIterator\.create\(",
        "FrameIterator.create(",
        "frame_iterator(...) function call",
    ),
    (
        r"from\s+pixeltable\.iterators\s+import\s+DocumentSplitter",
        "pixeltable.iterators.DocumentSplitter",
        "from pixeltable.functions.document import document_splitter",
    ),
    (r"string_embed=", "string_embed= kwarg", "embedding= kwarg in add_embedding_index"),
    (
        r"\.similarity\(\s*['\"]",
        "positional .similarity('col', ...)",
        "column.similarity(string=query) keyword form",
    ),
    (r"pixeltable_smart_install", "pixeltable_smart_install", "pixeltable_install_dependency"),
    (r"pixeltable\.ext\.functions", "pixeltable.ext.functions", "pixeltable.functions.<provider>"),
]

# Lines containing any of these tokens (case-insensitive) are treated as
# explicit "wrong vs right" warnings and excluded from the deprecated-pattern
# scan. Use `# allow: <reason>` on a single line for ad-hoc false positives.
ALLOWLIST_TOKENS = (
    "# allow:",
    "there is no ",
    "not ",
    "don't ",
    "no openai.vision",
    "no frameiterator",
    "no DocumentSplitter".lower(),
    "use `embedding=`",
    "use `chat_completions`",
)


def _strip_allowlisted(text: str) -> str:
    """Return the file content with allowlisted lines removed."""
    kept = []
    for line in text.splitlines():
        lowered = line.lower()
        if any(tok in lowered for tok in ALLOWLIST_TOKENS):
            continue
        kept.append(line)
    return "\n".join(kept)


@pytest.mark.parametrize("regex, name, fix", BANNED_PATTERNS, ids=lambda x: x if isinstance(x, str) else "")
def test_prompts_have_no_deprecated_patterns(regex, name, fix):
    text = PROMPT_FILE.read_text()
    cleaned = _strip_allowlisted(text)
    hits = re.findall(regex, cleaned)
    assert not hits, (
        f"{name} found in prompt.py outside allowlisted warning lines. "
        f"Use {fix} instead. Matches: {hits[:5]}"
    )
