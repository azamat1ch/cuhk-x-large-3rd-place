"""Standalone input-to-CSV runner; no reference answers or campaign dependencies."""

import argparse
import asyncio
import csv
import fcntl
import hashlib
import json
import tomllib
from pathlib import Path

from cuhk.answering import answer_choices, question_input
from cuhk.data import digest, fingerprint, probe, recording_path, write_json
from cuhk.emotion import discover_groups
from cuhk.investigator import investigate
from cuhk.knowledge import SkillLibrary
from cuhk.providers import api_endpoint, api_key, load_keys


def load_testing(root):
    root = Path(root).resolve()
    paths = list(root.rglob("test_qa.csv"))
    if len(paths) != 1:
        raise ValueError("Supply a directory containing exactly one test_qa.csv")
    records, order = {}, []
    with paths[0].open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"qa_id", "source", "path", "category", "question", "A", "B", "C", "D"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("test_qa.csv is missing required columns")
        for row in reader:
            qid = row["qa_id"].strip()
            if not qid or qid in order:
                raise ValueError("Empty or duplicate question ID")
            order.append(qid)
            path = recording_path(row["path"])
            record = records.setdefault(
                path,
                {
                    "schema_version": 1,
                    "recording_id": hashlib.sha256(f"testing:{path}".encode()).hexdigest()[:16],
                    "split": "testing",
                    "source": row["source"],
                    "path": path,
                    "subject": None,
                    "questions": [],
                    "streams": {},
                },
            )
            question = {
                "qa_id": qid,
                "category": row["category"],
                "question": row["question"],
                "options": {k: row[k] for k in "ABCD"},
            }
            answer_choices(question_input(question))
            record["questions"].append(question)
    if not records:
        raise ValueError("Empty test question file")
    for video in sorted(root.rglob("*.mp4")):
        modality = video.parent.name
        if modality not in {"IR", "Depth", "Thermal"}:
            continue
        relative = video.relative_to(root).as_posix()
        # Accept either the Testing parent or the published video root itself.
        path = recording_path(f"{root.name}/{relative}")
        if path not in records:
            continue
        streams = records[path]["streams"]
        if modality in streams:
            raise ValueError(f"Duplicate {modality} stream for recording")
        streams[modality] = {"path": str(video), **probe(video)}
    if any(not r["streams"] for r in records.values()):
        raise ValueError("A question recording has no permitted video stream")
    return list(records.values()), order


def assets_for(root, scenario):
    names = {"visual-inspection", "action-cues"}
    names |= {"emotion-comparison"} if scenario == "emotion" else {"objects", "question-reasoning"}
    paths = [root / f"prompts/investigators/{scenario}.md"]
    skill_root = root / "prompts/investigators/skills"
    paths += [
        p for p in sorted(skill_root.rglob("*.md")) if p.relative_to(skill_root).parts[0] in names
    ]
    assets = {p.relative_to(root).as_posix(): p.read_text() for p in paths}
    instructions = assets[f"prompts/investigators/{scenario}.md"]
    return assets, instructions + "\n\nAVAILABLE SKILLS\n" + json.dumps(
        SkillLibrary(assets).list(), indent=2
    )


async def run(root, data_dir, output, *, validate_only=False):
    config = tomllib.loads((root / "configs/final.toml").read_text())
    endpoint = api_endpoint(config.get("base_url"))
    records, order = load_testing(data_dir)
    groups = discover_groups(records)
    jobs = [
        ("normal", {**r, "questions": [q for q in r["questions"] if q["category"] != "emotion"]})
        for r in records
        if any(q["category"] != "emotion" for q in r["questions"])
    ]
    jobs += [("emotion", g) for g in groups]
    print(
        f"{len(records)} recordings, {len(order)} questions, {len(groups)} emotion candidates",
        flush=True,
    )
    if validate_only:
        return
    load_keys()
    api_key(endpoint)  # Fail before decoding or creating attempts when credentials are absent.
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Output already exists; choose a new output path")
    if output.is_relative_to(Path(data_dir).resolve()):
        raise ValueError("Output must be outside the input data directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    work = output.with_name(output.name + ".work")
    work.mkdir(exist_ok=True)
    with (work / "lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assets = {s: assets_for(root, s) for s in ("normal", "emotion")}
        identity = fingerprint(
            {
                "config": config,
                "endpoint": endpoint,
                "records": records,
                "assets": assets,
                "source": {p.name: digest(p) for p in (root / "src/cuhk").glob("*.py")},
                "media": {
                    v["path"]: digest(v["path"]) for r in records for v in r["streams"].values()
                },
            }
        )
        frozen = work / "identity.json"
        if frozen.exists() and json.loads(frozen.read_text()) != identity:
            raise ValueError("Inputs or implementation changed; choose a fresh output path")
        write_json(frozen, identity)
        workers = asyncio.Semaphore(config["workers"])
        decode = asyncio.Semaphore(config["preparation_workers"])
        predictions, failures = {}, []

        async def one(scenario, record):
            async with workers:
                unit = work / scenario / record["recording_id"]
                saved = unit / "completed.json"
                if saved.exists():
                    receipt = json.loads(saved.read_text())
                else:
                    attempt = unit / f"attempt-{len(list(unit.glob('attempt-*'))) + 1:03}"
                    skill_assets, instructions = assets[scenario]
                    receipt = await investigate(
                        record,
                        work / "views",
                        attempt,
                        instructions=instructions,
                        skill_assets=skill_assets,
                        scenario=scenario,
                        model=config["model"],
                        reasoning=config["reasoning"],
                        base_url=endpoint,
                        timeout_s=config[f"{scenario}_timeout_s"],
                        decode_semaphore=decode,
                        run_identity={"package_run": identity},
                    )
                    if receipt["status"] == "completed":
                        write_json(saved, receipt)
                if receipt["status"] != "completed":
                    failures.append(record["recording_id"])
                    print(
                        f"{scenario} {record['recording_id']}: failed ({receipt.get('error_type', 'unknown')})",
                        flush=True,
                    )
                    return
                answers = receipt["result"]["answers"]
                questions = record["questions"]
                if set(answers) != {f"q{i}" for i in range(1, len(questions) + 1)}:
                    raise ValueError("Saved answer coverage mismatch")
                for i, q in enumerate(questions, 1):
                    answer = answers[f"q{i}"]["answer"]
                    if answer not in answer_choices(question_input(q)) or q["qa_id"] in predictions:
                        raise ValueError("Invalid or duplicate prediction")
                    predictions[q["qa_id"]] = answer
                print(f"{scenario} {record['recording_id']}: completed", flush=True)

        await asyncio.gather(*(one(*job) for job in jobs))
        if failures or set(predictions) != set(order):
            raise RuntimeError(
                "Incomplete inference; completed jobs retained. Repeat the same command to retry unfinished jobs."
            )
        temporary = output.with_name(output.name + ".tmp")
        with temporary.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["qa_id", "prediction"])
            writer.writerows((qid, predictions[qid]) for qid in order)
        temporary.replace(output)
        print(f"Wrote {len(order)} predictions to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument(
        "--validate-only", action="store_true", help="Check inputs and grouping without model calls"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    asyncio.run(run(root, args.data_dir, args.output_csv, validate_only=args.validate_only))


if __name__ == "__main__":
    main()
