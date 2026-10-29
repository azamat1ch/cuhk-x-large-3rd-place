"""Downloaded inputs -> recording manifest. No model calls."""

import csv
import hashlib
import json
import re
import shutil
import stat
import subprocess
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

REPO_ID = "Kevin-Pal/CUHK-X_Large_Model_Track"
MODALITIES = {"Depth", "Depth_Color", "IR", "Thermal"}
ROOTS = {"HARn", "HAU", "large_model_track_test"}


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value, *, sort_keys=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=sort_keys) + "\n")
    temporary.replace(path)


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records))
    temporary.replace(path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def recording_path(value):
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe data path: {value}")
    starts = [i for i, part in enumerate(path.parts) if part in ROOTS]
    if len(starts) != 1:
        raise ValueError(f"Unrecognized recording layout: {value}")
    path = PurePosixPath(*path.parts[starts[0] :])
    if path.suffix.lower() == ".mp4":
        if path.parent.name not in MODALITIES:
            raise ValueError(f"Unknown video modality: {value}")
        path = path.parent.parent
    return str(path)


def safe_extract(archive, destination):
    """Validate the entire ZIP before extraction; never overwrite changed inputs."""
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            relative = PurePosixPath(member.filename.replace("\\", "/"))
            target = destination.joinpath(*relative.parts).resolve()
            mode = member.external_attr >> 16
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not target.is_relative_to(destination)
                or stat.S_ISLNK(mode)
            ):
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
            if target.exists() and not member.is_dir():
                raise ValueError(f"Extraction would overwrite {target}; use a fresh raw directory")
        for member in bundle.infolist():
            target = destination.joinpath(*PurePosixPath(member.filename.replace("\\", "/")).parts)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)


def probe(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    info = json.loads(result.stdout)
    stream = info["streams"][0]
    numerator, denominator = map(float, stream["avg_frame_rate"].split("/"))
    fps = numerator / denominator if denominator else None
    duration = float(stream.get("duration", info["format"].get("duration", 0)))
    if duration <= 0 or not fps or fps <= 0:
        raise ValueError(f"Invalid duration/frame rate: {path}")
    return {
        "bytes": path.stat().st_size,
        "sha256": digest(path),
        "duration_s": duration,
        "fps": fps,
        "width": stream["width"],
        "height": stream["height"],
        "frames": int(stream["nb_frames"]) if stream.get("nb_frames", "").isdigit() else None,
    }


def index_dataset(downloads, raw, probe_fn=probe, include_unreferenced=False):
    grouped = {}
    qa_inputs = []
    for split, name in [("training", "training_qa.csv"), ("testing", "test_qa.csv")]:
        paths = sorted(Path(downloads).rglob(name))
        if len(paths) > 1:
            raise ValueError(f"Multiple {name} files; isolate dataset revisions")
        if not paths:
            continue
        qa_inputs.append({"path": str(paths[0]), "sha256": digest(paths[0])})
        seen = set()
        with paths[0].open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"qa_id", "source", "path", "category", "question", "A", "B", "C", "D"}
            if split == "training":
                required.add("answer")
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(
                    f"Missing columns in {name}: {required - set(reader.fieldnames or [])}"
                )
            for row in reader:
                qid = row["qa_id"].strip()
                if not qid or qid in seen:
                    raise ValueError(f"Empty/duplicate qa_id in {name}: {qid}")
                seen.add(qid)
                path = recording_path(row["path"])
                key = (split, path)
                source = row["source"].strip()
                if source not in {"HAU", "HARn"}:
                    raise ValueError("CSV source must be HAU or HARn")
                if split == "training" and source != path.split("/")[0]:
                    raise ValueError("CSV source conflicts with training path")
                if key in grouped and source != grouped[key]["source"]:
                    raise ValueError("Conflicting CSV sources for one recording")
                match = re.search(r"(?:^|/)user(\d+)(?:/|$)", path)
                if split == "training" and not match:
                    raise ValueError(f"Missing training subject: {path}")
                if key not in grouped:
                    grouped[key] = {
                        "schema_version": 1,
                        "recording_id": hashlib.sha256(f"{split}:{path}".encode()).hexdigest()[:16],
                        "split": split,
                        "source": source,
                        "path": path,
                        "subject": int(match[1]) if split == "training" else None,
                        "questions": [],
                        "streams": {},
                    }
                question = {
                    "qa_id": qid,
                    "category": row["category"],
                    "question": row["question"],
                    "options": {k: row[k] for k in "ABCD"},
                }
                if split == "training":
                    answer = row["answer"].strip()
                    if not answer or any(c not in "ABCD" for c in answer):
                        raise ValueError(f"Unexpected answer format for {qid}: inspect source CSV")
                    question["answer"] = answer
                grouped[key]["questions"].append(question)
    if not grouped:
        raise ValueError("No QA CSVs found. Download metadata first.")
    media = defaultdict(dict)
    for path in sorted(Path(raw).rglob("*.mp4")):
        modality = path.parent.name
        if modality not in MODALITIES:
            raise ValueError(f"Unrecognized video modality: {path}")
        key = recording_path(path.relative_to(raw).as_posix())
        if modality in media[key]:
            raise ValueError(f"Duplicate stream for {key}/{modality}")
        media[key][modality] = {"path": str(path), **probe_fn(path)}
    for record in grouped.values():
        record["streams"] = media.get(record["path"], {})
    unreferenced = sorted(set(media) - {r["path"] for r in grouped.values()})
    if include_unreferenced:
        for path in unreferenced:
            split = "testing" if path.startswith("large_model_track_test/") else "training"
            match = re.search(r"(?:^|/)user(\d+)(?:/|$)", path)
            grouped[(split, path)] = {
                "schema_version": 1,
                "recording_id": hashlib.sha256(f"{split}:{path}".encode()).hexdigest()[:16],
                "split": split,
                "source": path.split("/")[0],
                "path": path,
                "subject": int(match[1]) if match and split == "training" else None,
                "questions": [],
                "streams": media[path],
            }
    records = sorted(grouped.values(), key=lambda r: (r["split"], r["path"]))
    return records, {
        "qa_inputs": qa_inputs,
        "unreferenced_recording_paths": unreferenced,
    }


def summary(records):
    def stats(values):
        values = sorted(values)
        if not values:
            return None
        mid = len(values) // 2
        return {
            "min": values[0],
            "median": (values[mid] + values[~mid]) / 2,
            "max": values[-1],
            "mean": sum(values) / len(values),
        }

    groups = defaultdict(list)
    for r in records:
        groups[f"{r['split']}/{r['source']}/subject={r['subject']}"].append(r)
    streams = defaultdict(list)
    for r in records:
        for modality, stream in r["streams"].items():
            streams[f"{r['split']}/{r['source']}/{modality}"].append(stream)
    return {
        "recordings": len(records),
        "questions": sum(len(r["questions"]) for r in records),
        "missing_all_media": sum(not r["streams"] for r in records),
        "groups": {
            k: {"recordings": len(v), "questions": sum(len(r["questions"]) for r in v)}
            for k, v in sorted(groups.items())
        },
        "categories": dict(
            Counter(f"{r['split']}/{q['category']}" for r in records for q in r["questions"])
        ),
        "questions_per_recording": stats([len(r["questions"]) for r in records]),
        "available_modality_sets": dict(
            Counter("+".join(sorted(r["streams"])) or "none" for r in records)
        ),
        "streams": {
            k: {
                "count": len(v),
                "bytes": stats([s["bytes"] for s in v]),
                "duration_s": stats([s["duration_s"] for s in v]),
                "fps": dict(Counter(str(s["fps"]) for s in v)),
                "resolutions": dict(Counter(f"{s['width']}x{s['height']}" for s in v)),
            }
            for k, v in sorted(streams.items())
        },
    }


def subject_split(records, subjects):
    available = {r["subject"] for r in records if r["split"] == "training"}
    if not subjects or not subjects < available:
        raise ValueError(
            "Choose existing training subjects, leaving at least one subject for training"
        )
    result = []
    for r in records:
        copy = dict(r)
        if r["split"] == "training" and r["subject"] in subjects:
            copy["split"] = "validation"
        result.append(copy)
    return result


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()
