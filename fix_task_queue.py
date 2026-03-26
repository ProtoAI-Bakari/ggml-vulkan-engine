#!/usr/bin/env python3
"""
Fix accumulated corruption in TASK_QUEUE_v5.md:
1. Duplicate timestamps on DONE tasks: "| 0% | started:..." trailing after "[DONE ... completed:...]"
2. Duplicate/stacked bracket garbage: "| t110] | t100] | t90] ..." on lines
3. DONE tasks get clean format: [DONE by AGENT | completed:TIMESTAMP]
4. READY tasks get NO timestamps
5. IN_PROGRESS tasks: strip trailing "| tNN]" noise, keep clean format
6. Missing closing brackets on DONE status tags
7. Missing space between "]" and description text
8. Recalculate TOC header counts to match actual content
"""

import re
import sys
from pathlib import Path

FILEPATH = Path.home() / "AGENT" / "TASK_QUEUE_v5.md"


def fix_task_line(line: str) -> str:
    """Fix a single ### T line."""
    if not line.startswith("### T"):
        return line

    # Extract task number prefix "### TNN: "
    m = re.match(r"(### T\d+:\s*)", line)
    if not m:
        return line
    prefix = m.group(1)
    rest = line[len(prefix):]

    # --- DONE tasks ---
    # Match DONE status with agent and completed timestamp, possibly followed by garbage
    # Pattern: [DONE by AGENT | completed:TIMESTAMP] possibly followed by junk then description
    done_match = re.match(
        r"\[DONE\s+by\s+([^\]|]+?)\s*\|\s*completed:(\S+?)\]"  # proper DONE tag
        r"(.*)",  # everything after
        rest
    )
    if done_match:
        agent = done_match.group(1).strip()
        timestamp = done_match.group(2).strip().rstrip("]")
        trailing = done_match.group(3)
        # Strip ALL the garbage after the first ]: duplicate timestamps, | 0% | started:..., | tNN], etc.
        # The description is the last meaningful text after stripping bracket garbage
        desc = re.sub(r"(\s*\|[^]]*\])+", "", trailing).strip()
        if not desc:
            # Maybe description was consumed; shouldn't happen but fallback
            desc = trailing.strip()
        # Ensure space before description
        return f"{prefix}[DONE by {agent} | completed:{timestamp}] {desc}"

    # DONE with just [DONE] - no agent/timestamp
    done_simple = re.match(r"\[DONE\](.*)", rest)
    if done_simple:
        desc = done_simple.group(1).strip()
        return f"{prefix}[DONE] {desc}"

    # DONE with agent but NO completed timestamp and NO closing bracket
    # e.g., [DONE by OmniAgent [Main] Description text
    done_no_close = re.match(
        r"\[DONE\s+by\s+([^\]]+?(?:\[[^\]]*\])?[^\]]*?)\s+"  # agent (may contain [sysN])
        r"(?!\|)(.+)",  # description (not starting with |)
        rest
    )
    if done_no_close:
        agent = done_no_close.group(1).strip()
        desc = done_no_close.group(2).strip()
        return f"{prefix}[DONE by {agent}] {desc}"

    # --- IN_PROGRESS tasks ---
    ip_match = re.match(
        r"\[IN_PROGRESS\s+by\s+([^\]|]+?)\s*\|\s*(\d+%)\s*\|\s*started:(\S+?)\]"
        r"(.*)",
        rest
    )
    if ip_match:
        agent = ip_match.group(1).strip()
        pct = ip_match.group(2)
        started = ip_match.group(3).strip().rstrip("]")
        trailing = ip_match.group(4)
        # Strip "| tNN]" garbage from trailing
        desc = re.sub(r"(\s*\|[^]]*\])+", "", trailing).strip()
        if not desc:
            desc = trailing.strip()
        return f"{prefix}[IN_PROGRESS by {agent} | {pct} | started:{started}] {desc}"

    # --- READY tasks ---
    ready_match = re.match(r"\[READY\b[^\]]*\](.*)", rest)
    if ready_match:
        desc = ready_match.group(1).strip()
        return f"{prefix}[READY] {desc}"

    return line


def ensure_space_after_bracket(line: str) -> str:
    """Ensure there's a space between ] and the description text on task lines."""
    if not line.startswith("### T"):
        return line
    # Fix "]TextWithNoSpace" -> "] TextWithNoSpace"
    # But only for the last ] before the actual description
    # Match: bracket followed by a letter/digit (no space)
    line = re.sub(r"\]([A-Za-z0-9])", r"] \1", line)
    return line


def count_statuses(lines):
    """Count DONE, IN_PROGRESS, READY tasks."""
    done = 0
    in_progress = 0
    ready = 0
    task_lines = [l for l in lines if l.startswith("### T")]
    for l in task_lines:
        if "[DONE" in l:
            done += 1
        elif "[IN_PROGRESS" in l:
            in_progress += 1
        elif "[READY" in l:
            ready += 1
    return done, in_progress, ready, len(task_lines)


def count_phase_done(lines, task_range_start, task_range_end):
    """Count DONE tasks in a given task number range."""
    done = 0
    total = 0
    for l in lines:
        m = re.match(r"### T(\d+):", l)
        if m:
            tnum = int(m.group(1))
            if task_range_start <= tnum <= task_range_end:
                total += 1
                if "[DONE" in l:
                    done += 1
    return done, total


def fix_toc(lines):
    """Recalculate and fix the TOC header counts."""
    # Phase definitions: (label_pattern, start_task, end_task)
    phases = [
        (0, 1, 10),
        (1, 11, 25),
        (2, 26, 48),
        (3, 49, 55),
        (4, 56, 65),
        (5, 66, 70),
        (6, 71, 76),
        (7, 77, 80),
        (8, 81, 85),
    ]

    done_total, ip_total, ready_total, task_total = count_statuses(lines)

    # Fix the summary box
    for i, line in enumerate(lines):
        if "DONE:" in line and "IN_PROGRESS:" in line and "READY:" in line:
            lines[i] = f"# | DONE: {done_total:>2} | IN_PROGRESS: {ip_total:>2} | READY: {ready_total:>2} |"
        if "Total:" in line and "tasks across" in line:
            lines[i] = f"# | Total: {task_total:>2} tasks across 9 phases       |"

    # Fix per-phase lines
    for phase_num, ts, te in phases:
        d, t = count_phase_done(lines, ts, te)
        for i, line in enumerate(lines):
            pat = rf"^# PHASE {phase_num}:"
            if re.match(pat, line):
                # Preserve the label text after the em-dash
                m2 = re.match(r"(# PHASE \d+:[^[]*\[T\d+-T\d+\]\s*)\S\s*(.*)", line)
                if m2:
                    before = m2.group(1)
                    status_word = "DONE" if d == t else ("READY" if d == 0 else "MIXED")
                    lines[i] = f"{before}\u2014 {d:>2}/{t}  {status_word}"
                break

    # Fix the summary section at the bottom
    for i, line in enumerate(lines):
        if line.startswith("- ") and "tasks across 9 phases" in line:
            lines[i] = f"- {task_total} tasks across 9 phases"
        elif re.match(r"^- \d+ DONE,", line):
            lines[i] = f"- {done_total} DONE, {ip_total} IN_PROGRESS, {ready_total} READY"

    return lines


def main():
    text = FILEPATH.read_text()
    lines = text.split("\n")

    print(f"Read {len(lines)} lines from {FILEPATH}")

    # Pre-fix counts
    d0, ip0, r0, t0 = count_statuses(lines)
    print(f"BEFORE: {d0} DONE, {ip0} IN_PROGRESS, {r0} READY, {t0} total tasks")

    # Show corrupted lines before fixing
    print("\n=== CORRUPTED LINES (before fix) ===")
    for i, line in enumerate(lines, 1):
        if not line.startswith("### T"):
            continue
        # Detect: duplicate timestamps, bracket garbage, missing space
        problems = []
        if re.search(r"\]\s*\|\s*0%\s*\|\s*started:", line):
            problems.append("duplicate_timestamp")
        if re.search(r"\|\s*t\d+\]", line):
            problems.append("tNN_garbage")
        if re.search(r"\][A-Za-z]", line):
            problems.append("missing_space")
        if re.search(r"\[DONE\s+by\s+[^\]]*$", line):
            problems.append("unclosed_bracket")
        if problems:
            print(f"  L{i}: {', '.join(problems)}")
            print(f"    {line[:120]}...")

    # Fix each task line
    new_lines = []
    for line in lines:
        fixed = fix_task_line(line)
        fixed = ensure_space_after_bracket(fixed)
        new_lines.append(fixed)

    # Fix TOC counts
    new_lines = fix_toc(new_lines)

    # Post-fix counts
    d1, ip1, r1, t1 = count_statuses(new_lines)
    print(f"\nAFTER: {d1} DONE, {ip1} IN_PROGRESS, {r1} READY, {t1} total tasks")

    # Show fixed lines
    print("\n=== FIXED LINES ===")
    for old, new in zip(lines, new_lines):
        if old != new:
            print(f"  OLD: {old[:120]}")
            print(f"  NEW: {new[:120]}")
            print()

    # Write result
    FILEPATH.write_text("\n".join(new_lines))
    print(f"Wrote {len(new_lines)} lines to {FILEPATH}")


if __name__ == "__main__":
    main()
