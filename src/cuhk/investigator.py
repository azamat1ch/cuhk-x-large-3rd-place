"""One bounded visual investigation; evaluation answers never enter this module."""

import asyncio
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Literal

from agents import (
    Agent,
    ModelSettings,
    RunConfig,
    RunHooks,
    Runner,
    function_tool,
    set_tracing_disabled,
)
from agents.models.openai_responses import OpenAIResponsesModel
from agents.tool import ToolOutputImage, ToolOutputText
from openai import AsyncOpenAI
from openai.types.shared import Reasoning
from pydantic import BaseModel, ConfigDict, create_model

from cuhk.answering import answer_choices, question_input, validate_joint_answer
from cuhk.data import digest
from cuhk.inspection import Inspection
from cuhk.knowledge import SkillLibrary
from cuhk.providers import api_endpoint, api_key, data_url, load_keys
from cuhk.transport_retry import transport_retry_settings

RECOVERABLE_VIEW_ERRORS = {
    "Unavailable or forbidden source",
    "View parameters must be finite numbers",
    "Invalid interval, sampling rate or gamma",
    "Unknown layout",
    "Detail layout requires a crop",
    "Zoom must be between 1 and 8",
    "Crop must contain four integer pixel coordinates",
    "Crop outside source dimensions",
    "Sampling rate must be positive or null for native frames",
    "Invalid contrast or sharpening amount (0 to 2)",
    "Rotation must be 0, 90, 180 or 270 degrees clockwise",
    "Page must be nonnegative and page_size positive",
    "Explicit timestamps must be nonempty and within the interval",
    "No native frames in interval",
    "Page outside selected frames",
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Answer(StrictModel):
    answer: str
    evidence: str


class Event(StrictModel):
    source: Literal["IR", "Depth", "Thermal"]
    start_s: float
    end_s: float
    visible_feature: str
    action_candidates: list[str]
    references: list[str]


class GroupEvent(Event):
    recording_id: str


class GroupAssessment(StrictModel):
    decision: Literal["compatible", "incompatible", "uncertain", "singleton"]
    reason: str


def output_model(questions, scenario="normal", *, group_assessment_required=True):
    """Explicit question keys let strict structured output reject missing answers."""
    if scenario not in {"normal", "emotion"}:
        raise ValueError("Unknown investigator scenario")
    answers = create_model(
        "InvestigatorAnswers",
        __base__=StrictModel,
        **{
            q["id"]: (
                create_model(
                    f"Answer_{q['id']}",
                    __base__=Answer,
                    answer=(Literal[tuple(answer_choices(q))], ...),
                ),
                ...,
            )
            for q in questions
        },
    )
    confidence = create_model(
        "InvestigatorConfidence",
        __base__=StrictModel,
        **{q["id"]: (Literal["high", "medium", "low"], ...) for q in questions},
    )
    return create_model(
        "InvestigatorResult",
        __base__=StrictModel,
        scene=(str, ...),
        answers=(answers, ...),
        timeline=(list[GroupEvent] if scenario == "emotion" else list[Event], ...),
        confidence=(confidence, ...),
        uncertainty=(list[str], ...),
        **(
            {
                "comparisons": (list[str], ...),
                "strongest_alternative": (str, ...),
                **(
                    {"group_assessment": (GroupAssessment, ...)}
                    if group_assessment_required
                    else {}
                ),
            }
            if scenario == "emotion"
            else {}
        ),
    )


def normal_questions(record):
    return [
        {**question_input(q), "id": f"q{i}"}
        for i, q in enumerate((q for q in record["questions"] if q["category"] != "emotion"), 1)
    ]


def inference_questions(record, scenario="normal"):
    if scenario == "normal":
        return normal_questions(record)
    if scenario != "emotion":
        raise ValueError("Unknown investigator scenario")
    questions = []
    for i, member in enumerate(record["members"], 1):
        for question in member["questions"]:
            if question["category"] == "emotion":
                questions.append(
                    {
                        **question_input(question),
                        "id": f"q{len(questions) + 1}",
                        "recording_id": f"r{i}",
                    }
                )
    return questions


def _group_metadata(metadata, recording_id):
    """Namespace citations without changing immutable shared inspection caches."""
    value = dict(metadata, recording_id=recording_id)
    if "view_id" in value:
        value["view_id"] = f"{recording_id}/{value['view_id']}"
    if "frames" in value:
        value["frames"] = [
            dict(frame, frame_ref=f"{recording_id}/{frame['frame_ref']}")
            for frame in value["frames"]
        ]
    if "context_frame_refs" in value:
        value["context_frame_refs"] = [
            f"{recording_id}/{ref}" for ref in value["context_frame_refs"]
        ]
    return value


def validate_result(value, questions, inspection, scenario="normal"):
    if isinstance(value, BaseModel):
        value = value.model_dump()
    value = output_model(questions, scenario).model_validate(value).model_dump()
    validate_joint_answer({k: value[k] for k in ("scene", "answers")}, questions)
    inspections = inspection if scenario == "emotion" else {"r1": inspection}
    if scenario == "emotion":
        if set(inspections) != {q["recording_id"] for q in questions}:
            raise ValueError("Invalid group question coverage")
        assessment = value["group_assessment"]
        if not assessment["reason"].strip():
            raise ValueError("Missing grouping assessment reason")
        if (assessment["decision"] == "singleton") != (len(inspections) == 1):
            raise ValueError("Grouping assessment does not match member count")
        if not value["strongest_alternative"].strip() or not any(
            text.strip() for text in value["comparisons"]
        ):
            raise ValueError("Missing group comparison or alternative")
    sources_by_recording = {}
    references = {}
    for recording_id, item in inspections.items():
        sources = item.describe()["sources"]
        sources_by_recording[recording_id] = sources
        if not item.coverage:
            raise ValueError("Missing chronological visual coverage")
        longest = max(source["duration_s"] for source in sources.values())
        if not any(sources[s]["duration_s"] >= longest - 1 for s in item.coverage):
            raise ValueError("Missing overview of longer available source")
        refs = references[recording_id] = {}
        prefix = f"{recording_id}/" if scenario == "emotion" else ""
        for view_id, metadata in item.delivered.items():
            times = [frame["actual_s"] for frame in metadata["frames"]]
            if times:
                refs[prefix + view_id] = (metadata["source"], min(times), max(times))
            for frame in metadata["frames"]:
                refs[prefix + frame["frame_ref"]] = (
                    frame["source"],
                    frame["actual_s"],
                    frame["actual_s"],
                )
    if not value["timeline"] and not any(s.strip() for s in value["uncertainty"]):
        raise ValueError("Empty timeline requires explicit uncertainty")
    for event in value["timeline"]:
        recording_id = event["recording_id"] if scenario == "emotion" else "r1"
        if recording_id not in sources_by_recording:
            raise ValueError("Unknown event recording")
        sources = sources_by_recording[recording_id]
        source = event["source"]
        if (
            source not in sources
            or not 0 <= event["start_s"] <= event["end_s"] <= sources[source]["duration_s"]
            or not event["visible_feature"].strip()
            or not event["references"]
        ):
            raise ValueError("Invalid event bounds or missing evidence")
        for ref in event["references"]:
            if ref not in references[recording_id]:
                raise ValueError("Undelivered event reference")
            ref_source, start, end = references[recording_id][ref]
            if ref_source != source or end < event["start_s"] or start > event["end_s"]:
                raise ValueError("Event reference does not overlap event")
    return value


def _json(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    raise TypeError(type(value).__name__)


def _save(path, value):
    path.write_text(json.dumps(value, default=_json, indent=2) + "\n")


class Checkpoints(RunHooks):
    def __init__(self, folder):
        self.folder = folder
        self.calls = []
        self.usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    async def on_llm_start(self, context, agent, system_prompt, input_items):
        # Retain protocol items including encrypted reasoning; never a prose-only summary.
        _save(self.folder / "last-model-input.json", input_items)

    async def on_llm_end(self, context, agent, response):
        self.calls.append({"response_id": response.response_id, "output": response.output})
        for key in self.usage:
            self.usage[key] += getattr(response.usage, key, 0) or 0
        _save(self.folder / "model-calls.json", self.calls)
        _save(self.folder / "usage.json", self.usage)


async def investigate(
    record,
    folder,
    output_dir,
    *,
    instructions,
    model="gpt-6-astra",
    reasoning="xhigh",
    decode_semaphore=None,
    max_turns=None,
    timeout_s=900,
    base_url=None,
    skill_assets=None,
    run_identity=None,
    scenario="normal",
):
    """Run once; return completed/failed receipt. Existing attempts are never overwritten."""
    if model != "gpt-6-astra":
        raise ValueError("Investigator requires explicit Astra model")
    base_url = api_endpoint(base_url)
    if max_turns is not None and (type(max_turns) is not int or max_turns < 1):
        raise ValueError("max_turns must be a positive integer or None")
    turn_limit = sys.maxsize if max_turns is None else max_turns
    skills = SkillLibrary(skill_assets or {})
    skill_reads = []
    transport_retries = []
    questions = inference_questions(record, scenario)
    if not questions:
        raise ValueError(f"No {scenario} questions")
    members = record["members"] if scenario == "emotion" else [record]
    if scenario == "emotion" and (
        not 1 <= len(members) <= 3
        or any(
            not any(q["category"] == "emotion" for q in member["questions"]) for member in members
        )
    ):
        raise ValueError("Emotion groups require one to three members with emotion questions")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    hooks = Checkpoints(output_dir)
    receipt = {
        "status": "failed",
        "requested_model": model,
        "actual_models": [],
        "usage": hooks.usage,
        "continuation_supported": False,
        "validation_retries": 0,
        "transport_retries": transport_retries,
        "model_turns": 0,
        "run_identity": dict(run_identity or {}),
    }
    inspections = {}
    client = None
    result = None
    semaphore = decode_semaphore or asyncio.Semaphore(1)
    decodes = []

    def record_transport_retry(event):
        transport_retries.append(event)
        _save(output_dir / "transport-retries.json", transport_retries)

    async def decode(function, *args, **kwargs):
        started_decode = asyncio.Event()

        async def work():
            async with semaphore:
                started_decode.set()
                return await asyncio.to_thread(function, *args, **kwargs)

        task = asyncio.create_task(work())
        decodes.append(task)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if not started_decode.is_set():
                task.cancel()
            # Running threads cannot be stopped. Keep capacity and drain them before
            # publishing the receipt; they may still be writing delivery/cache files.
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue  # SDK cleanup may cancel the same tool more than once.
                except Exception:
                    break
            raise

    try:
        async with asyncio.timeout(timeout_s):
            for i, member in enumerate(members, 1):
                inspection_dir = output_dir / "inspection"
                if scenario == "emotion":
                    inspection_dir /= f"r{i}"
                inspections[f"r{i}"] = await decode(Inspection, member, inspection_dir, folder)

            def select_inspection(recording_id):
                if recording_id not in inspections:
                    raise ValueError("Unknown recording_id; use a recording ID from the input")
                return inspections[recording_id]

            @function_tool(failure_error_function=None)
            async def describe_recording(recording_id: str = "r1") -> str:
                """Read safe source dimensions, coverage and capabilities for a recording."""
                try:
                    metadata = select_inspection(recording_id).describe()
                except ValueError as exc:
                    return json.dumps({"error": str(exc), "correctable": True})
                if scenario == "emotion":
                    metadata = dict(metadata, recording_id=recording_id)
                return json.dumps(metadata)

            @function_tool(failure_error_function=None)
            async def view_recording(
                source: Literal["IR", "Depth", "Thermal"],
                start_s: float | None = None,
                end_s: float | None = None,
                fps: float | None = 0.25,
                crop: list[int] | None = None,
                gamma: float = 1.0,
                layout: Literal["sheets", "individual", "detail"] = "sheets",
                timestamps_s: list[float] | None = None,
                contrast: Literal["none", "stretch", "clahe"] = "none",
                sharpen: float = 0.0,
                rotation: Literal[0, 90, 180, 270] = 0,
                page: int = 0,
                page_size: int = 24,
                zoom: float = 1.0,
                recording_id: str = "r1",
            ) -> list[ToolOutputText | ToolOutputImage]:
                """Inspect one page of timestamped real frames from a source-clock interval.

                recording_id selects a local member (r1, r2, r3) from the input.
                Omit start/end for full-source sampling; fps=None selects every native
                frame. timestamps_s selects arbitrary source times instead of fps.
                Follow next_page with the same request to see all selected frames.
                Crops are [left,top,right,bottom] in original pixels and retain context.
                Detail layout requires a crop: one unchanged context frame per page,
                then compact chronological crops. Zoom 1..8 uses bicubic enlargement
                of adjusted views only, not recovered detail; originals stay native.
                Gamma above one brightens; contrast is none, stretch or clahe.
                Sharpen changes display edges only; rotation is clockwise degrees.
                Transformations do not recover missing information. Source clocks differ.
                """
                try:
                    viewed = await decode(
                        select_inspection(recording_id).view,
                        source,
                        start_s=start_s,
                        end_s=end_s,
                        fps=fps,
                        crop=crop,
                        gamma=gamma,
                        layout=layout,
                        timestamps_s=timestamps_s,
                        contrast=contrast,
                        sharpen=sharpen,
                        rotation=rotation,
                        page=page,
                        page_size=page_size,
                        zoom=zoom,
                    )
                except ValueError as exc:
                    if str(exc) not in RECOVERABLE_VIEW_ERRORS and not str(exc).startswith(
                        "Unknown recording_id;"
                    ):
                        raise
                    return [
                        ToolOutputText(
                            text=json.dumps(
                                {
                                    "error": str(exc),
                                    "correctable": True,
                                    "guidance": "Revise the view arguments using describe_recording metadata.",
                                }
                            )
                        )
                    ]
                metadata = viewed["metadata"]
                if scenario == "emotion":
                    metadata = _group_metadata(metadata, recording_id)
                return [
                    ToolOutputText(text=json.dumps(metadata)),
                    *[
                        ToolOutputImage(image_url=data_url(Path(p), "image/png"), detail="high")
                        for p in viewed["image_paths"]
                    ],
                ]

            @function_tool(failure_error_function=None)
            async def list_skills() -> str:
                """Discover frozen investigation skills and when each is useful."""
                return json.dumps(skills.list())

            @function_tool(failure_error_function=None)
            async def read_skill(skill_id: str, resource: str | None = None) -> str:
                """Read a frozen skill or one of its listed resources by exact identifier.

                Omit resource to read the main guidance; use only resource names
                returned by the skill catalog or main guidance, never file paths.
                """
                try:
                    content = skills.read(skill_id, resource)
                except ValueError:
                    return json.dumps(
                        {
                            "error": "Unavailable skill or resource; use list_skills",
                            "correctable": True,
                        }
                    )
                skill_reads.append({key: val for key, val in content.items() if key != "content"})
                _save(output_dir / "skill-reads.json", skill_reads)
                return json.dumps(content)

            _save(output_dir / "skill-reads.json", skill_reads)
            packet = {
                "recording_id": "r1",
                "questions": questions,
                "recording": inspections["r1"].describe(),
            }
            if scenario == "emotion":
                packet = {
                    "questions": questions,
                    "candidate_basis": record.get("candidate_basis", "supplied_membership"),
                    "routine_questions": [
                        {
                            "recording_id": f"r{i}",
                            "questions": [
                                {k: v for k, v in question_input(q).items() if k != "id"}
                                for q in member["questions"]
                                if q["category"] != "emotion"
                            ],
                        }
                        for i, member in enumerate(members, 1)
                    ],
                    "recordings": [
                        {"recording_id": key, "recording": item.describe()}
                        for key, item in inspections.items()
                    ],
                }
            _save(output_dir / "input.json", packet)
            (output_dir / "instructions.md").write_text(instructions)
            load_keys()
            set_tracing_disabled(True)
            client = AsyncOpenAI(
                api_key=api_key(base_url),
                base_url=base_url,
                max_retries=0,
                timeout=timeout_s,
            )
            agent = Agent(
                name="Visual investigator",
                instructions=instructions,
                model=OpenAIResponsesModel(model=model, openai_client=client),
                tools=[describe_recording, view_recording, list_skills, read_skill],
                output_type=output_model(questions, scenario),
                model_settings=ModelSettings(
                    reasoning=Reasoning(effort=reasoning),
                    store=False,
                    response_include=["reasoning.encrypted_content"],
                    parallel_tool_calls=True,
                    retry=transport_retry_settings(record_transport_retry),
                ),
            )
            model_input = json.dumps(packet)
            first_candidate = None
            for candidate_index in range(2):
                remaining_turns = turn_limit - receipt["model_turns"]
                if remaining_turns <= 0:
                    raise ValueError("No turns remaining for evidence validation correction")
                result = Runner.run_streamed(
                    agent if candidate_index == 0 else agent.clone(tools=[]),
                    input=model_input,
                    max_turns=remaining_turns,
                    hooks=hooks,
                    run_config=RunConfig(tracing_disabled=True),
                )
                async for event in result.stream_events():
                    if (
                        event.type == "raw_response_event"
                        and event.data.type == "response.completed"
                    ):
                        actual = event.data.response.model
                        if actual and actual not in receipt["actual_models"]:
                            receipt["actual_models"].append(actual)
                receipt["model_turns"] += result.current_turn
                history = result.to_input_list()
                state = result.to_state().to_json()
                _save(output_dir / f"candidate-{candidate_index}-history.json", history)
                _save(output_dir / f"candidate-{candidate_index}-state.json", state)
                _save(output_dir / "history.json", history)
                _save(output_dir / "state.json", state)
                if not receipt["actual_models"] or any(
                    not actual.startswith(model) for actual in receipt["actual_models"]
                ):
                    raise ValueError("Astra response model identity not verified")
                raw = result.final_output
                raw = raw.model_dump() if isinstance(raw, BaseModel) else raw
                _save(output_dir / f"candidate-{candidate_index}.json", raw)
                if first_candidate is not None and {
                    q: entry["answer"] for q, entry in raw["answers"].items()
                } != {q: entry["answer"] for q, entry in first_candidate["answers"].items()}:
                    raise ValueError("Evidence correction changed answer choices")
                preserved = ("confidence", "uncertainty")
                if scenario == "emotion":
                    preserved += ("comparisons", "strongest_alternative", "group_assessment")
                if first_candidate is not None and any(
                    raw[key] != first_candidate[key] for key in preserved
                ):
                    raise ValueError("Evidence correction changed uncertainty assessment")
                try:
                    value = validate_result(
                        raw,
                        questions,
                        inspections if scenario == "emotion" else inspections["r1"],
                        scenario,
                    )
                    break
                except ValueError as exc:
                    if str(exc) not in {
                        "Undelivered event reference",
                        "Event reference does not overlap event",
                    }:
                        raise
                    _save(
                        output_dir / f"candidate-{candidate_index}-validation.json",
                        {
                            "error": str(exc),
                            "correction": "exact delivered evidence references only",
                        },
                    )
                    if candidate_index or receipt["model_turns"] >= turn_limit:
                        raise
                    first_candidate = raw
                    receipt["validation_retries"] = 1
                    model_input = [
                        *history,
                        {
                            "role": "user",
                            "content": "Local evidence validation failed: " + str(exc) + ". "
                            "Use exact delivered frame/view references separately; no ranges or "
                            "abbreviations. References must match the event source and overlap its "
                            "time interval. Correct only invalid evidence bookkeeping. Preserve "
                            "all answer choices, confidence, uncertainty, and any group_assessment, "
                            "comparisons and strongest_alternative exactly. "
                            "Scene/evidence wording may change only "
                            "to correct evidence bookkeeping. "
                            "Return the complete structured result again.",
                        },
                    ]
            _save(output_dir / "result.json", value)
            receipt.update(status="completed", result=value)
    except Exception as exc:
        # Error messages may echo transport payloads/credentials or raw source paths.
        receipt["error_type"] = type(exc).__name__
        state = getattr(exc, "state", None)
        if state is not None:
            _save(output_dir / "interrupted-state.json", state.to_json())
    finally:
        if result is not None and receipt["status"] != "completed":
            result.cancel(mode="immediate")
            try:
                async for _event in result.stream_events():
                    pass
            except (Exception, asyncio.CancelledError):
                pass
        # Also cover tool threads whose SDK task is still processing cancellation.
        if result is not None and getattr(result, "run_loop_task", None) is not None:
            await asyncio.gather(result.run_loop_task, return_exceptions=True)
        await asyncio.gather(*decodes, return_exceptions=True)
        if client is not None:
            await client.close()
        receipt["wall_time_s"] = round(time.monotonic() - started, 3)
        if inspections:
            receipt["delivered_evidence"] = (
                {
                    f"{key}/{view_id}": _group_metadata(metadata, key)
                    for key, item in inspections.items()
                    for view_id, metadata in item.delivered.items()
                }
                if scenario == "emotion"
                else inspections["r1"].delivered
            )
        receipt["files"] = {
            p.relative_to(output_dir).as_posix(): digest(p)
            for p in sorted(output_dir.rglob("*"))
            if p.is_file()
        }
        _save(output_dir / "receipt.json", receipt)
    return receipt
