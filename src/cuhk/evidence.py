"""Local timestamp requests and verified frame extraction; no model calls."""

import bisect
import json
import math
import subprocess
import tempfile
from pathlib import Path

from cuhk.data import MODALITIES, digest, probe, write_json


def select_frames(timestamps, requested, duration):
    """Map seconds to nearest decoded PTS, preferring the earlier frame on ties."""
    if not timestamps or not math.isfinite(duration) or duration <= 0:
        raise ValueError("Missing frame timestamps or invalid duration")
    if any(not math.isfinite(t) or t < 0 or t >= duration for t in timestamps):
        raise ValueError("Frame timestamps must be within stream duration")
    if any(b <= a for a, b in zip(timestamps, timestamps[1:], strict=False)):
        raise ValueError("Frame timestamps must be strictly increasing")
    if not isinstance(requested, list) or not requested:
        raise ValueError("Request at least one timestamp")
    result = []
    for t in requested:
        if type(t) not in (int, float) or not math.isfinite(t) or not 0 <= t < duration:
            raise ValueError("Requested seconds must be finite and within [0, duration)")
        pos = bisect.bisect_left(timestamps, t)
        candidates = {max(0, pos - 1), min(len(timestamps) - 1, pos)}
        index = min(candidates, key=lambda i: (abs(timestamps[i] - t), i))
        result.append({"requested_s": t, "frame_index": index, "actual_s": timestamps[index]})
    return result


def extract_evidence(video, request_path, destination, size=None):
    """Extract requested evidence atomically; a completed destination is immutable."""
    video, request_path, destination = map(Path, (video, request_path, destination))
    if destination.exists():
        raise FileExistsError(f"Evidence destination already exists: {destination}")
    request_bytes = request_path.read_bytes()
    request = json.loads(request_bytes)
    if not isinstance(request, dict) or set(request) != {"modality", "timestamps_s"}:
        raise ValueError("Request must contain exactly modality and timestamps_s")
    if request["modality"] not in {*MODALITIES, "Skeleton"}:
        raise ValueError("Unknown modality")
    if size is not None and (len(size) != 2 or any(type(v) is not int or v < 2 for v in size)):
        raise ValueError("Invalid frame dimensions")
    metadata = probe(video)
    response = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "json",
            str(video),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    timestamps = [
        float(f["best_effort_timestamp_time"]) for f in json.loads(response.stdout)["frames"]
    ]
    if not timestamps:
        raise ValueError("No decoded video frames")
    origin = timestamps[0]
    relative = [t - origin for t in timestamps]
    selections = select_frames(relative, request["timestamps_s"], metadata["duration_s"])
    indices = sorted({s["frame_index"] for s in selections})
    filenames = {index: f"frame-{i:04d}.png" for i, index in enumerate(indices, 1)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".evidence-", dir=destination.parent) as temp:
        folder = Path(temp)
        expression = "+".join(f"eq(n\\,{i})" for i in indices)
        filters = f"select={expression}"
        if size is not None:
            w, h = size
            filters += (
                f",scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos"
                f",pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1"
            )
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-threads",
                "1",
                "-noautorotate",
                "-i",
                str(video),
                "-map",
                "0:v:0",
                "-an",
                "-vf",
                filters,
                "-fps_mode",
                "vfr",
                "-threads",
                "1",
                str(folder / "frame-%04d.png"),
            ],
            capture_output=True,
            check=True,
            timeout=120,
        )
        images = sorted(folder.glob("frame-*.png"))
        if len(images) != len(indices) or any(p.stat().st_size == 0 for p in images):
            raise ValueError("FFmpeg did not produce all requested evidence frames")
        if digest(video) != metadata["sha256"] or request_path.read_bytes() != request_bytes:
            raise ValueError("Input changed during extraction")
        for selection in selections:
            selection["file"] = filenames[selection["frame_index"]]
        result = {
            "schema_version": 1,
            "source_sha256": metadata["sha256"],
            "modality": request["modality"],
            "time_base": "seconds_since_first_decoded_frame_of_this_stream",
            "source_first_pts_s": origin,
            "duration_s": metadata["duration_s"],
            "selections": selections,
            "files_sha256": {p.name: digest(p) for p in images},
            "actual_calls": 0,
            "frame_size": size,
        }
        write_json(folder / "evidence.json", result)
        # Refuse pre-existing destinations, including those created during processing.
        if destination.exists():
            raise FileExistsError(f"Evidence destination already exists: {destination}")
        folder.rename(destination)
    return result
