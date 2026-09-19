#!/usr/bin/env python3
"""Render Claude Code session transcripts (~/.claude/projects/<project>/*.jsonl) to Markdown for
the A2 Q7.4 chat-history export, with credentials redacted.

    PYTHONPATH=. .venv/bin/python scripts/export_chat.py --out ~/A2_submission/chat_exports/aayush \
        --redact 'string1' --redact 'string2'

Kept: every human prompt, every assistant text reply, and one line per tool call (tool name +
a short description or command) so the sequence of actions is visible. Dropped: assistant
"thinking" blocks, tool *results* (file dumps, logs — the repo and RESULTS.md carry those),
system/attachment/bookkeeping records. Redaction replaces each given string with [REDACTED]
in the Markdown output only; the raw JSONL is left untouched on disk.
"""
import argparse, json, re
from datetime import datetime
from pathlib import Path

PROJECT = Path.home() / ".claude/projects/-home-aayush-IRE-News-Recommendation-System"

# Targeted patterns for a credential that is also an ordinary word elsewhere in the transcript
# (a name / username): only the password-bearing lines are touched.
REDACT_PATTERNS = [
    (r"(sudo password[^\n]*\n\s*)\S+", r"\1[REDACTED]"),                       # "…my sudo password\n<pw>"
    (r"printf '%s\\n' '[^']+' \| sudo -S", "printf '%s\\n' '[REDACTED]' | sudo -S"),   # the install command
]


def text_of(content) -> list[tuple[str, str]]:
    """(kind, text) pieces from a message content list."""
    if isinstance(content, str):
        return [("text", content)]
    out = []
    for block in content or []:
        t = block.get("type")
        if t == "text":
            out.append(("text", block.get("text", "")))
        elif t == "tool_use":
            inp = block.get("input", {})
            desc = inp.get("description") or inp.get("command") or inp.get("file_path") or inp.get("prompt") or ""
            out.append(("tool", f"{block.get('name')}: {str(desc)[:200]}"))
        elif t == "tool_result":
            out.append(("result", ""))          # dropped
    return out


def render(path: Path, redact: list[str]) -> str:
    lines = [f"# Claude Code session `{path.stem}`\n", f"Source: `{path.name}` · project `{PROJECT.name}` · rendered {datetime.now():%Y-%m-%d %H:%M}\n",
             "Human prompts and assistant replies in order; one line per tool call; tool outputs and thinking omitted; credentials redacted.\n"]
    n_user = n_asst = n_tool = 0
    first = last = None
    for raw in path.open():
        try:
            r = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if r.get("type") not in ("user", "assistant"):
            continue
        ts = r.get("timestamp")
        if ts:
            first = first or ts; last = ts
        pieces = text_of(r.get("message", {}).get("content"))
        if r["type"] == "user":
            texts = [t for k, t in pieces if k == "text" and t.strip() and not t.startswith("[Request interrupted")]
            texts = [t for t in texts if "<system-reminder>" not in t and "SYSTEM NOTIFICATION" not in t and "<task-notification>" not in t]
            if texts:
                n_user += 1
                lines.append(f"\n---\n\n**Aayush** ({(ts or '')[:16].replace('T', ' ')}):\n\n" + "\n\n".join(t.strip() for t in texts) + "\n")
        else:
            for k, t in pieces:
                if k == "text" and t.strip():
                    n_asst += 1
                    lines.append(f"\n**Claude:**\n\n{t.strip()}\n")
                elif k == "tool":
                    n_tool += 1
                    lines.append(f"- 🛠 `{t}`")
    body = "\n".join(lines)
    for s in redact:
        if s:
            body = body.replace(s, "[REDACTED]")
    for pat, rep in REDACT_PATTERNS:
        body = re.sub(pat, rep, body)
    head = f"\n{n_user} prompts · {n_asst} replies · {n_tool} tool calls · {str(first)[:10]} → {str(last)[:10]}\n"
    return body.replace("credentials redacted.\n", "credentials redacted.\n" + head, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--redact", action="append", default=[])
    ap.add_argument("--project", type=Path, default=PROJECT)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for p in sorted(a.project.glob("*.jsonl")):
        md = render(p, a.redact)
        for s in a.redact:
            assert s not in md, f"redaction failed for a string in {p.name}"
        assert not re.search(r"sudo password[^\n]*\n\s*(?!\[REDACTED\])\S", md), f"password line survived in {p.name}"
        assert "| sudo -S" not in md or "'[REDACTED]' | sudo -S" in md
        out = a.out / f"session_{p.stem[:8]}.md"
        out.write_text(md)
        print(f"{out}  ({len(md)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
