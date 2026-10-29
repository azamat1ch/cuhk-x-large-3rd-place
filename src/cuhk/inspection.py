"""Paginated local viewing tools. No model calls or source paths in tool metadata."""

import bisect
import fcntl
import hashlib
import json
import math
import subprocess
import tempfile
import threading
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageOps

from cuhk import evidence
from cuhk.data import digest, probe, write_json

ALLOWED = ("IR", "Depth", "Thermal")


def _key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _clahe(image):
    """Luminance CLAHE: 8x8 tiles, clip factor 2, bilinear LUT interpolation."""
    luminance, cb, cr = image.convert("YCbCr").split()
    width, height = image.size
    nx, ny = min(8, width), min(8, height)
    xs = [i * width // nx for i in range(nx + 1)]
    ys = [i * height // ny for i in range(ny + 1)]
    maps = []
    for y in range(ny):
        row = []
        for x in range(nx):
            histogram = luminance.crop((xs[x], ys[y], xs[x + 1], ys[y + 1])).histogram()
            count = sum(histogram)
            limit = max(1, round(2 * count / 256))
            excess = sum(max(0, n - limit) for n in histogram)
            base, remainder = divmod(excess, 256)
            histogram = [min(n, limit) + base for n in histogram]
            # Distribute the residual evenly instead of biasing the darkest bins.
            for i in range(remainder):
                histogram[i * 256 // remainder] += 1
            cumulative, lut = 0, []
            for n in histogram:
                cumulative += n
                lut.append(round(255 * cumulative / count))
            row.append(lut)
        maps.append(row)

    def weights(bounds, size):
        centers = [(a + b - 1) / 2 for a, b in zip(bounds, bounds[1:], strict=False)]
        result = []
        for position in range(size):
            right = min(len(centers) - 1, bisect.bisect_left(centers, position))
            left = max(0, right - 1)
            fraction = (
                max(0, min(1, (position - centers[left]) / (centers[right] - centers[left])))
                if left != right
                else 0
            )
            result.append((left, right, fraction))
        return result

    wx, wy = weights(xs, width), weights(ys, height)
    pixels = luminance.tobytes()
    output = bytearray(len(pixels))
    for y, (top, bottom, fy) in enumerate(wy):
        for x, (left, right, fx) in enumerate(wx):
            value = pixels[y * width + x]
            upper = maps[top][left][value] * (1 - fx) + maps[top][right][value] * fx
            lower = maps[bottom][left][value] * (1 - fx) + maps[bottom][right][value] * fx
            output[y * width + x] = round(upper * (1 - fy) + lower * fy)
    changed = Image.frombytes("L", image.size, bytes(output))
    return Image.merge("YCbCr", (changed, cb, cr)).convert("RGB")


class Inspection:
    """One recording's delivered evidence; callers must serialize model deliveries."""

    def __init__(self, record, folder, cache_root):
        self.folder, self.cache_root = Path(folder), Path(cache_root)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.delivered = {}
        self.coverage = set()
        self._frames = 0
        self._overview_pages = {}
        self._timestamps = {}
        self._lock = threading.Lock()
        self._sources = {}
        for name in ALLOWED:
            if name not in record["streams"]:
                continue
            stream = record["streams"][name]
            path = Path(stream["path"])
            metadata = probe(path)
            if metadata["sha256"] != stream["sha256"]:
                raise ValueError("Source fingerprint mismatch")
            self._sources[name] = (path, metadata)
        if not self._sources:
            raise ValueError("No allowed visual sources")
        self._implementation = _key([digest(Path(__file__)), digest(Path(evidence.__file__))])

    def describe(self):
        """Coverage is sampling coverage, never a claim that streams are aligned."""
        return {
            "time_base": "seconds_since_first_decoded_frame_of_each_stream",
            "streams_aligned": False,
            "sources": {
                name: {**meta, "overview_delivered": name in self.coverage}
                for name, (_, meta) in self._sources.items()
            },
            "usage": {"views_used": len(self.delivered), "frames_used": self._frames},
        }

    def view(
        self,
        source,
        start_s=None,
        end_s=None,
        fps=0.25,
        crop=None,
        gamma=1.0,
        layout="sheets",
        timestamps_s=None,
        contrast="none",
        sharpen=0.0,
        rotation=0,
        page=0,
        page_size=24,
        zoom=1.0,
    ):
        with self._lock:
            return self._view(
                source,
                start_s,
                end_s,
                fps,
                crop,
                gamma,
                layout,
                timestamps_s,
                contrast,
                sharpen,
                rotation,
                page,
                page_size,
                zoom,
            )

    def _view(
        self,
        source,
        start_s,
        end_s,
        fps,
        crop,
        gamma,
        layout,
        timestamps_s,
        contrast,
        sharpen,
        rotation,
        page,
        page_size,
        zoom,
    ):
        if source not in self._sources:
            raise ValueError("Unavailable or forbidden source")
        path, meta = self._sources[source]
        duration = meta["duration_s"]
        start = 0 if start_s is None else start_s
        end = duration if end_s is None else end_s
        if any(
            type(x) not in (int, float) or not math.isfinite(x)
            for x in (start, end, gamma, sharpen, zoom)
        ):
            raise ValueError("View parameters must be finite numbers")
        if not 0 <= start < end <= duration or not 0.1 <= gamma <= 5:
            raise ValueError("Invalid interval, sampling rate or gamma")
        if fps is not None and (
            type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0
        ):
            raise ValueError("Sampling rate must be positive or null for native frames")
        if contrast not in ("none", "stretch", "clahe") or not 0 <= sharpen <= 2:
            raise ValueError("Invalid contrast or sharpening amount (0 to 2)")
        if type(rotation) is not int or rotation not in (0, 90, 180, 270):
            raise ValueError("Rotation must be 0, 90, 180 or 270 degrees clockwise")
        if type(page) is not int or page < 0 or type(page_size) is not int or page_size < 1:
            raise ValueError("Page must be nonnegative and page_size positive")
        if timestamps_s is not None and (
            not isinstance(timestamps_s, list)
            or not timestamps_s
            or any(
                type(t) not in (int, float) or not math.isfinite(t) or not start <= t < end
                for t in timestamps_s
            )
        ):
            raise ValueError("Explicit timestamps must be nonempty and within the interval")
        if not 1 <= zoom <= 8:
            raise ValueError("Zoom must be between 1 and 8")
        if layout == "detail" and crop is None:
            raise ValueError("Detail layout requires a crop")
        if layout not in ("sheets", "individual", "detail"):
            raise ValueError("Unknown layout")
        if crop is not None:
            if (
                not isinstance(crop, (list, tuple))
                or len(crop) != 4
                or any(type(x) is not int for x in crop)
            ):
                raise ValueError("Crop must contain four integer pixel coordinates")
            left, top, right, bottom = crop
            if not (0 <= left < right <= meta["width"] and 0 <= top < bottom <= meta["height"]):
                raise ValueError("Crop outside source dimensions")
            crop = list(crop)
        if digest(path) != meta["sha256"]:
            raise ValueError("Source fingerprint changed")
        times = self._frame_times(source, path, duration)
        if timestamps_s is not None:
            requested = timestamps_s
        elif fps is None:
            requested = [t for t in times if start <= t < end]
        else:
            # Work scales with native frames, including arbitrarily high requested FPS.
            requested = [start, math.nextafter(end, start)]
            span = (end - start) * fps
            if not math.isfinite(span):
                requested += [t for t in times if start <= t < end]
            else:
                count = math.ceil(span)
                for t in times:
                    position = (t - start) * fps
                    if not math.isfinite(position):
                        continue
                    for near in (math.floor(position), math.ceil(position)):
                        i = min(count - 1, max(0, near))
                        candidate = start + i / fps
                        if candidate < end:
                            requested.append(candidate)
        if not requested:
            raise ValueError("No native frames in interval")
        selections = evidence.select_frames(times, requested, duration)
        selected_times = sorted({item["actual_s"] for item in selections})
        total = len(selected_times)
        pages = math.ceil(total / page_size)
        if page >= pages:
            raise ValueError("Page outside selected frames")
        delivered_times = selected_times[page * page_size : (page + 1) * page_size]
        request = {
            "source": source,
            "source_sha256": meta["sha256"],
            "start_s": start,
            "end_s": end,
            "fps": fps,
            "crop": crop,
            "gamma": gamma,
            "layout": layout,
            "zoom": zoom,
            "timestamps_s": timestamps_s,
            "contrast": contrast,
            "sharpen": sharpen,
            "rotation": rotation,
            "page_size": page_size,
            "implementation": self._implementation,
        }
        selection_id = "s_" + _key(request)
        request.update(page=page, selection_id=selection_id, total_frames=total, pages=pages)
        view_id = "v_" + _key(request)
        duplicate = view_id in self.delivered
        destination = self.cache_root / view_id
        with (self.cache_root / (view_id + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not destination.exists():
                self._create(destination, path, request, delivered_times, meta, view_id)
            receipt = self._validate(destination, request)
        safe = receipt["metadata"]
        if not duplicate:
            self.delivered[view_id] = safe
            self._frames += len(safe["frames"])
            if safe["overview_request"]:
                seen = self._overview_pages.setdefault(selection_id, set())
                seen.add(page)
                if len(seen) == pages:
                    self.coverage.add(source)
            write_json(self.folder / "delivered-views.json", self.delivered)
        return {
            "metadata": {**safe, "duplicate": duplicate},
            "image_paths": [destination / name for name in receipt["images"]],
        }

    def _frame_times(self, source, path, duration):
        if source not in self._timestamps:
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
                    str(path),
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=120,
            )
            pts = [
                float(f["best_effort_timestamp_time"])
                for f in json.loads(response.stdout)["frames"]
            ]
            if not pts:
                raise ValueError("No decoded video frames")
            times = [t - pts[0] for t in pts]
            evidence.select_frames(times, [0], duration)
            self._timestamps[source] = times
        return self._timestamps[source]

    def _create(self, destination, path, request, requested, meta, view_id):
        with tempfile.TemporaryDirectory(prefix=".view-", dir=self.cache_root) as temp:
            folder = Path(temp)
            req = folder / "request.json"
            write_json(req, {"modality": request["source"], "timestamps_s": requested})
            raw = evidence.extract_evidence(path, req, folder / "frames")
            if raw["source_sha256"] != request["source_sha256"]:
                raise ValueError("Source fingerprint changed during extraction")
            selected = {s["frame_index"]: s for s in raw["selections"]}
            selected = [selected[k] for k in sorted(selected)]
            frames, tiles, context = [], [], []
            for i, item in enumerate(selected):
                ref = f"{view_id}:f{i:03d}"
                frames.append(
                    {
                        "frame_ref": ref,
                        "actual_s": item["actual_s"],
                        "source": request["source"],
                        "source_sha256": meta["sha256"],
                    }
                )
                with Image.open(folder / "frames" / item["file"]) as opened:
                    original = opened.convert("RGB")
                caption = f"f{i:03d} | {request['source']} {item['actual_s']:.3f}s"
                if request["layout"] == "detail":
                    if i == 0:
                        context.append((original, caption + " original context"))
                else:
                    tiles.append((original, caption + " original"))
                if (
                    request["crop"] is not None
                    or request["gamma"] != 1
                    or request["contrast"] != "none"
                    or request["sharpen"]
                    or request["rotation"]
                    or request["zoom"] != 1
                ):
                    changed = original.crop(request["crop"]) if request["crop"] else original.copy()
                    lut = [round(255 * (n / 255) ** (1 / request["gamma"])) for n in range(256)]
                    changed = changed.point(lut * 3)
                    if request["contrast"] == "stretch":
                        changed = ImageOps.autocontrast(changed, cutoff=1, preserve_tone=True)
                    elif request["contrast"] == "clahe":
                        changed = _clahe(changed)
                    if request["sharpen"]:
                        changed = changed.filter(
                            ImageFilter.UnsharpMask(
                                radius=1, percent=round(100 * request["sharpen"]), threshold=3
                            )
                        )
                    if request["rotation"]:
                        changed = changed.transpose(
                            {
                                90: Image.Transpose.ROTATE_270,
                                180: Image.Transpose.ROTATE_180,
                                270: Image.Transpose.ROTATE_90,
                            }[request["rotation"]]
                        )
                    if request["zoom"] != 1:
                        changed = changed.resize(
                            tuple(round(n * request["zoom"]) for n in changed.size),
                            Image.Resampling.BICUBIC,
                        )
                    tiles.append((changed, caption + " adjusted"))
            if request["layout"] == "detail":
                images = self._render(folder, context, "individual", prefix="context")
                images += self._render(folder, tiles, "sheets", compact=True)
            else:
                images = self._render(folder, tiles, request["layout"])
            safe = {
                "view_id": view_id,
                "source": request["source"],
                "source_sha256": meta["sha256"],
                "duration_s": meta["duration_s"],
                "start_s": request["start_s"],
                "end_s": request["end_s"],
                "overview_request": (
                    request["timestamps_s"] is None
                    and request["start_s"] == 0
                    and request["end_s"] == meta["duration_s"]
                ),
                "overview": (
                    request["timestamps_s"] is None
                    and request["start_s"] == 0
                    and request["end_s"] == meta["duration_s"]
                    and request["pages"] == 1
                ),
                "selection_id": request["selection_id"],
                "page": request["page"],
                "page_size": request["page_size"],
                "pages": request["pages"],
                "total_frames": request["total_frames"],
                "next_page": request["page"] + 1
                if request["page"] + 1 < request["pages"]
                else None,
                "fps": request["fps"],
                "timestamps_s": request["timestamps_s"],
                "contrast": request["contrast"],
                "sharpen": request["sharpen"],
                "rotation": request["rotation"],
                "transform_order": [
                    "crop",
                    "gamma",
                    "contrast",
                    "sharpen",
                    "clockwise_rotation",
                    "zoom",
                ],
                "contrast_parameters": {
                    "stretch_percentiles": [1, 99],
                    "clahe_tiles": [8, 8],
                    "clahe_clip_limit": 2,
                },
                "sharpen_parameters": {"radius": 1, "threshold": 3},
                "actual_times_s": [f["actual_s"] for f in frames],
                "frames": frames,
                "crop": request["crop"],
                "gamma": request["gamma"],
                "layout": request["layout"],
                "zoom": request["zoom"],
                "zoom_interpolation": "bicubic" if request["zoom"] != 1 else None,
                "native_resolution": request["zoom"] == 1,
                "context_frame_refs": [
                    f["frame_ref"]
                    for f in (frames[:1] if request["layout"] == "detail" else frames)
                ],
                "endpoint_note": "Times identify decoded frames; explicit requests use nearest PTS",
                "source_first_pts_s": raw["source_first_pts_s"],
                "time_base": raw["time_base"],
            }
            files = {
                p.relative_to(folder).as_posix(): digest(p)
                for p in folder.rglob("*")
                if p.is_file()
            }
            receipt = {
                "request": request,
                "metadata": safe,
                "images": images,
                "files_sha256": files,
            }
            write_json(folder / "receipt.json", receipt)
            (folder / "receipt.sha256").write_text(digest(folder / "receipt.json"))
            folder.rename(destination)

    @staticmethod
    def _render(folder, tiles, layout, *, prefix="view", compact=False):
        # Packing never resamples: oversize individual tiles retain their explicit size.
        images = []
        pages = []
        page, draw, x, y, row_h, used_w = None, None, 0, 0, 0, 0
        for image, caption in tiles:
            w, h = image.size
            h += 24
            if layout == "individual" or w > 2048 or h > 2048:
                if page is not None:
                    pages.append(page.crop((0, 0, used_w if compact else 2048, y + row_h)))
                    page, draw, x, y, row_h, used_w = None, None, 0, 0, 0, 0
                tile = Image.new("RGB", (max(w, 360), h), "white")
                tile.paste(image, (0, 24))
                ImageDraw.Draw(tile).text((4, 4), caption, fill="black")
                pages.append(tile)
                continue
            if x + max(w, 360) > 2048:
                x, y, row_h = 0, y + row_h, 0
            if page is None or y + h > 2048:
                if page is not None:
                    pages.append(page.crop((0, 0, used_w if compact else 2048, y + row_h)))
                page = Image.new("RGB", (2048, 2048), "white")
                draw = ImageDraw.Draw(page)
                x, y, row_h, used_w = 0, 0, 0, 0
            page.paste(image, (x, y + 24))
            draw.text((x + 4, y + 4), caption, fill="black")
            x += max(w, 360)
            used_w = max(used_w, x)
            row_h = max(row_h, h)
        if page is not None:
            pages.append(page.crop((0, 0, used_w if compact else 2048, y + row_h)))
        for i, page in enumerate(pages):
            name = f"{prefix}-{i:03d}.png"
            page.save(folder / name)
            images.append(name)
        return images

    @staticmethod
    def _validate(folder, request):
        try:
            if digest(folder / "receipt.json") != (folder / "receipt.sha256").read_text():
                raise ValueError("Corrupt view receipt")
            receipt = json.loads((folder / "receipt.json").read_text())
            if receipt["request"] != request:
                raise ValueError("View cache request mismatch")
            for name, sha in receipt["files_sha256"].items():
                file = folder / name
                if not file.resolve().is_relative_to(folder.resolve()) or digest(file) != sha:
                    raise ValueError("Corrupt view cache")
            for name in receipt["images"]:
                with Image.open(folder / name) as image:
                    image.verify()
            return receipt
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError("Missing or corrupt view cache") from exc
