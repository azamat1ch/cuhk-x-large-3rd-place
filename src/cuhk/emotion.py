"""Input-only candidate discovery and supplied memberships; no answer lookup."""

import csv
import re
from itertools import combinations
from pathlib import Path

from cuhk.data import fingerprint, read_jsonl


def emotion_groups(manifest, split, groups_path="auto", *, numbering="auto"):
    records = [r for r in read_jsonl(manifest) if r["split"] == split]
    by_id = {r["recording_id"]: r for r in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate recording IDs")
    wanted = {
        key: {**r, "questions": [q for q in r["questions"] if q["category"] == "emotion"]}
        for key, r in by_id.items()
        if any(q["category"] == "emotion" for q in r["questions"])
    }
    if groups_path == "auto":
        return discover_groups(records, numbering=numbering)
    seen, groups = set(), []
    with Path(groups_path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            ids = row.get("recording_ids", "").split(";")
            if not all(ids) or len(set(ids)) != len(ids) or seen.intersection(ids):
                raise ValueError("Invalid or overlapping recording groups")
            seen.update(ids)
            present = set(ids).intersection(by_id)
            if present and present != set(ids):
                raise ValueError("A recording group is only partly present in this split")
            if not present:
                continue
            members = [wanted[key] for key in ids if key in wanted]
            if not members:
                continue
            if len(members) != len(ids) or len(members) > 3:
                raise ValueError("Emotion groups require 1–3 emotion-bearing members")
            # Canonical order is independent of CSV row/member order and original labels.
            members.sort(key=lambda r: r["recording_id"])
            questions = [q for r in members for q in r["questions"]]
            groups.append(
                {
                    "recording_id": "g" + fingerprint([r["recording_id"] for r in members])[:16],
                    "members": members,
                    "questions": questions,
                }
            )
    if set(wanted) - seen:
        raise ValueError("Grouping file missing emotion recordings")
    qids = [q["qa_id"] for g in groups for q in g["questions"]]
    if len(qids) != len(set(qids)):
        raise ValueError("Duplicate question IDs")
    return sorted(groups, key=lambda g: g["recording_id"])


def _norm(value):
    return " ".join(value.casefold().split())


def _shape(record):
    return sorted((q["category"], _norm(q["question"])) for q in record["questions"])


def _options(record, category):
    return {
        tuple(sorted(_norm(part) for part in value.split(",")))
        if category == "combination"
        else _norm(value)
        for q in record["questions"]
        if q["category"] == category
        for value in q["options"].values()
    }


def _components(nodes, edges):
    neighbours = {node: set() for node in nodes}
    for a, b in edges:
        neighbours[a].add(b)
        neighbours[b].add(a)
    remaining = set(nodes)
    result = []
    while remaining:
        pending = [min(remaining)]
        component = set()
        while pending:
            node = pending.pop()
            if node in component:
                continue
            component.add(node)
            pending.extend(neighbours[node] - component)
        remaining -= component
        result.append(sorted(component))
    return result


def discover_groups(records, *, numbering="auto"):
    """Input-only proposals; visual compatibility is decided during investigation."""
    if numbering not in {"auto", "off"}:
        raise ValueError("grouping_numbers must be auto or off")
    records = [r for r in records if any(q["category"] == "emotion" for q in r["questions"])]
    by_id = {r["recording_id"]: r for r in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate recording IDs")
    ids = sorted(by_id)
    numbers = {}
    if numbering == "auto":
        for key, record in by_id.items():
            match = re.fullmatch(r"LM_test_(\d+)", Path(record.get("path", "")).name)
            if match:
                numbers[key] = int(match[1])
    if len(set(numbers.values())) != len(numbers):
        numbers = {}  # Duplicate indices cannot establish adjacency.
    numbered = len(numbers) == len(ids) and bool(ids)
    shapes = {key: _shape(r) for key, r in by_id.items()}
    options = {
        key: {c: _options(r, c) for c in ("emotion", "sequence", "combination")}
        for key, r in by_id.items()
    }
    edges = []
    for a, b in combinations(ids, 2):
        shared = len(options[a]["emotion"] & options[b]["emotion"])
        if numbered:
            link = abs(numbers[a] - numbers[b]) == 1 and shared >= 3
        else:
            routine = len(options[a]["sequence"] & options[b]["sequence"]) >= 2 or bool(
                options[a]["combination"] & options[b]["combination"]
            )
            link = shapes[a] == shapes[b] and shared >= 3 and routine
        if link:
            edges.append((a, b))
    components = _components(ids, edges)
    units = [
        part
        for component in components
        for part in ([component] if len(component) <= 3 else [[key] for key in component])
    ]
    proposals = [
        (unit, "numbered_overlap" if numbered else "question_option_overlap") for unit in units
    ]
    if numbered:
        # Only unresolved units may merge. Competing extensions abstain together.
        # Pair adjacent residual singletons before considering extensions of pairs.
        solo_indices = [i for i, unit in enumerate(units) if len(unit) == 1]
        solo_links = [
            (i, j)
            for i, j in combinations(solo_indices, 2)
            if shapes[units[i][0]] == shapes[units[j][0]]
            and abs(numbers[units[i][0]] - numbers[units[j][0]]) == 1
        ]
        resolved, consumed = [], set()
        for component in _components(solo_indices, solo_links):
            if 2 <= len(component) <= 3:
                resolved.append(
                    ([units[i][0] for i in component], "uncertain_numbered_question_match")
                )
                consumed.update(component)
        units = [unit for i, unit in enumerate(units) if i not in consumed]
        links = []
        for i, j in combinations(range(len(units)), 2):
            combined = units[i] + units[j]
            if len(combined) > 3 or len({tuple(shapes[key]) for key in combined}) != 1:
                continue
            ns = sorted(numbers[key] for key in combined)
            if ns[-1] - ns[0] == len(ns) - 1:
                links.append((i, j))
        proposals = resolved
        for component in _components(range(len(units)), links):
            combined = sorted(key for i in component for key in units[i])
            if len(component) > 1 and len(combined) <= 3:
                proposals.append((combined, "uncertain_numbered_question_match"))
            else:
                proposals.extend((units[i], "numbered_overlap") for i in component)
    groups = []
    for unit, basis in proposals:
        members = [by_id[key] for key in sorted(unit)]
        questions = [q for r in members for q in r["questions"] if q["category"] == "emotion"]
        groups.append(
            {
                "recording_id": "g" + fingerprint(sorted(unit))[:16],
                "members": members,
                "questions": questions,
                "candidate_basis": basis if len(unit) > 1 else "singleton",
            }
        )
    qids = [q["qa_id"] for g in groups for q in g["questions"]]
    if len(qids) != len(set(qids)):
        raise ValueError("Duplicate question IDs")
    return sorted(groups, key=lambda g: g["recording_id"])
