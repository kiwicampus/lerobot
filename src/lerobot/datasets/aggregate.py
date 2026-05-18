#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import shutil
from pathlib import Path

import pandas as pd
import tqdm

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_DATA_PATH,
    DEFAULT_EPISODES_PATH,
    DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    DEFAULT_VIDEO_PATH,
    get_file_size_in_mb,
    get_parquet_file_size_in_mb,
    to_parquet_with_hf_images,
    update_chunk_file_indices,
    write_info,
    write_stats,
    write_tasks,
)
from lerobot.datasets.video_utils import concatenate_video_files, get_video_duration_in_s

# LOCAL-PATCH(visibility): Phase 0 instrumentation for the rtop merge pipeline.
# Why: a 5000-ep production run on 2026-05-13 went silent for ~46h between the
# "Find all tasks" log line and the 48h Batch cap. We had no idea which sub-step
# hung. This block makes aggregate_datasets() emit per-source progress + a final
# PHASE_TIMING JSON so we can target perf work with measurement, not inspection.
#
# Configurable via env vars (all optional):
#   LEROBOT_AGGREGATE_PROGRESS_EVERY  log every N source datasets (default 10)
#   LEROBOT_AGGREGATE_PHASE_TIMING_PATH  if set, write final summary JSON here
#
# Forward-rebase: every line tagged with `# LOCAL-PATCH(visibility):` belongs to
# this patch. To remove, grep that tag and delete the contiguous block.
import json as _viz_json
import os as _viz_os
import sys as _viz_sys
import time as _viz_time


def _viz_progress_every() -> int:
    """Read at call time so env-var changes between calls take effect."""
    return max(1, int(_viz_os.environ.get("LEROBOT_AGGREGATE_PROGRESS_EVERY", "10")))


def _viz_phase_timing_path() -> str | None:
    return _viz_os.environ.get("LEROBOT_AGGREGATE_PHASE_TIMING_PATH") or None


class _VizPhaseTimer:
    """Accumulates wall-clock per named phase across an aggregate_datasets() run."""

    def __init__(self) -> None:
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.run_start = _viz_time.monotonic()

    def record(self, name: str, elapsed: float) -> None:
        self.totals[name] = self.totals.get(name, 0.0) + elapsed
        self.counts[name] = self.counts.get(name, 0) + 1

    def summary(self, n_sources: int) -> dict:
        return {
            "wall_s_total": _viz_time.monotonic() - self.run_start,
            "n_sources": n_sources,
            "phases": [
                {"phase": name, "wall_s": self.totals[name], "n_calls": self.counts[name]}
                for name in sorted(self.totals.keys())
            ],
        }


def _viz_flush() -> None:
    try:
        _viz_sys.stdout.flush()
        _viz_sys.stderr.flush()
    except Exception:
        pass


def _viz_time_call(timer: _VizPhaseTimer, name: str, fn, *args, **kwargs):
    """Time a single phase call and record into the run timer."""
    t0 = _viz_time.monotonic()
    try:
        return fn(*args, **kwargs)
    finally:
        timer.record(name, _viz_time.monotonic() - t0)
# end LOCAL-PATCH(visibility)


# LOCAL-PATCH(visibility-inner): per-branch counters for aggregate_videos.
# Phase 0 outer profiling showed aggregate_videos at ~93% of wall-clock. To
# target Phase 3 work correctly we need to know which branch inside it
# dominates: shutil.copy on first-in-shard, shutil.copy on size-rotation,
# or ffmpeg concat on append-into-existing. PyAV duration probe is also
# counted here since it fires on every iteration.
#
# Module-level dict, reset at the start of each aggregate_datasets() run,
# populated by aggregate_videos(), merged into the PHASE_TIMING summary.
_VIZ_INNER: dict = {}


def _viz_inner_reset() -> None:
    _VIZ_INNER.clear()
    for k in ("videos.copy_first", "videos.copy_rotate", "videos.concat", "videos.probe"):
        _VIZ_INNER[k] = {"wall_s": 0.0, "n_calls": 0, "bytes_in": 0}


def _viz_inner_record(name: str, elapsed: float, bytes_in: int = 0) -> None:
    e = _VIZ_INNER.setdefault(name, {"wall_s": 0.0, "n_calls": 0, "bytes_in": 0})
    e["wall_s"] += elapsed
    e["n_calls"] += 1
    e["bytes_in"] += bytes_in
# end LOCAL-PATCH(visibility-inner)


# LOCAL-PATCH(perf-video-plan): Phase 3 / Patch D — plan-then-execute video
# aggregation (Stage A: planner). See
# ml/data_pipeline/lerobot_merge/docs/patch_d_design.html for the full spec.
#
# Why: the Phase 0.5 inner profile showed concatenate_video_files() is 98.5%
# of aggregate_videos wall-clock (66.7s/190 calls/36.1 GB read on 40 ep). The
# online code re-reads the growing destination on every concat, so cumulative
# I/O scales O(M^2) with sources-per-shard. The planner instead packs sources
# into ShardPlans up front; Stage B then runs one multi-source ffmpeg concat
# per shard (each source read exactly once), parallelised across shards.
#
# This block adds:
#   - SourceEntry, ShardPlan dataclasses
#   - IncompatibleSourceError
#   - plan_video_shards()  — computes the full plan + per-source videos_idx view
#
# Stage B (executor) and Stage C (integration into aggregate_datasets) come in
# follow-up patches. This block is callable + testable on its own; it does no
# I/O beyond stat() and get_video_duration_in_s() probes.
from concurrent import futures as _pvp_pool
from dataclasses import dataclass as _pvp_dataclass, field as _pvp_field
import subprocess as _pvp_subprocess
import tempfile as _pvp_tempfile


class IncompatibleSourceError(ValueError):
    """Raised by plan_video_shards() when sources for a camera disagree on
    codec / pix_fmt / resolution / fps. Aborts the merge before any I/O."""


@_pvp_dataclass(frozen=True)
class SourceEntry:
    src_path: Path
    src_chunk: int
    src_file: int
    src_size_bytes: int
    src_duration_s: float
    offset_in_dst_s: float


@_pvp_dataclass
class ShardPlan:
    key: str
    dst_chunk: int
    dst_file: int
    dst_path: Path
    sources: list = _pvp_field(default_factory=list)  # list[SourceEntry]

    @property
    def total_size_bytes(self) -> int:
        return sum(s.src_size_bytes for s in self.sources)

    @property
    def total_duration_s(self) -> float:
        return sum(s.src_duration_s for s in self.sources)


def _pvp_unique_src_pairs(src_meta, key) -> list[tuple[int, int]]:
    """Sorted unique (chunk, file) pairs for `key` in this source's episodes
    metadata. Same logic the online aggregate_videos uses."""
    return sorted({
        (chunk, file)
        for chunk, file in zip(
            src_meta.episodes[f"videos/{key}/chunk_index"],
            src_meta.episodes[f"videos/{key}/file_index"],
            strict=False,
        )
    })


def _pvp_validate_camera_consistency(all_metadata, key: str) -> None:
    """All source datasets must agree on codec / pix_fmt / resolution / fps
    for the given camera, otherwise ffmpeg -c copy concat will fail at
    runtime. We catch this up-front so the merge aborts cleanly."""
    if not all_metadata:
        return
    ref_feat = all_metadata[0].features.get(key)
    if ref_feat is None:
        raise IncompatibleSourceError(
            f"camera '{key}' missing from first source's features"
        )
    ref_info = ref_feat.get("info", {}) or {}
    ref_shape = ref_feat.get("shape")
    keys_to_check = ("video.codec", "video.pix_fmt", "video.fps",
                     "video.height", "video.width")
    for src_meta in all_metadata[1:]:
        f = src_meta.features.get(key)
        if f is None:
            raise IncompatibleSourceError(
                f"camera '{key}' missing from source {src_meta.repo_id}"
            )
        info = f.get("info", {}) or {}
        if f.get("shape") != ref_shape:
            raise IncompatibleSourceError(
                f"camera '{key}' shape mismatch: ref={ref_shape} "
                f"vs {src_meta.repo_id}={f.get('shape')}"
            )
        for k in keys_to_check:
            if info.get(k) != ref_info.get(k):
                raise IncompatibleSourceError(
                    f"camera '{key}' {k} mismatch: ref={ref_info.get(k)} "
                    f"vs {src_meta.repo_id}={info.get(k)}"
                )


def plan_video_shards(
    all_metadata,
    aggr_root: Path,
    video_keys: list[str],
    video_files_size_in_mb: float,
    chunk_size: int,
) -> tuple[dict, dict]:
    """Compute the full destination-shard plan for every camera + a per-source
    videos_idx view that aggregate_metadata() can consume without re-touching
    file state.

    Returns:
      plan:            {camera_key: [ShardPlan, ...]} in shard order
      per_source_view: {src_repo_id: {camera_key: videos_idx_dict}}

    `per_source_view[r][k]` matches the shape `aggregate_videos()` produces
    today, with one twist: `episode_duration` is pre-set to 0 so the existing
    `latest_duration += episode_duration` mutation in `aggregate_metadata()`
    becomes a no-op. The cumulative duration before this source's first src
    is recorded as `latest_duration` directly.
    """
    size_cap_bytes = int(video_files_size_in_mb * 1024 * 1024)
    plan: dict[str, list[ShardPlan]] = {}
    # per_source_view: src_repo_id -> {key: {chunk, file, latest_duration,
    # episode_duration, src_to_offset}}
    per_source_view: dict[str, dict[str, dict]] = {}

    # We need to enumerate sources in the same order aggregate_datasets() does
    # so the resulting layout matches one-for-one. Source order is the order
    # of `all_metadata`; within a source, the (chunk, file) pairs are sorted.
    for src_meta in all_metadata:
        per_source_view.setdefault(src_meta.repo_id, {})

    for key in video_keys:
        _pvp_validate_camera_consistency(all_metadata, key)
        shards: list[ShardPlan] = []
        chunk_idx = 0
        file_idx = 0
        # Running estimates of the current (still-open) destination shard.
        dst_size_estimate_bytes = 0
        dst_duration_estimate_s = 0.0
        current_shard: ShardPlan | None = None

        for src_meta in all_metadata:
            # Per-source view starts fresh; will be filled below.
            src_view = {
                "chunk": chunk_idx,
                "file": file_idx,
                "latest_duration": dst_duration_estimate_s
                if current_shard is not None and current_shard.sources
                else 0.0,
                "episode_duration": 0.0,           # see "contract trick" in design doc
                "src_to_offset": {},
            }
            # Track what we record into the view per (src_chunk, src_file).
            last_dst_chunk = chunk_idx
            last_dst_file = file_idx

            for src_chunk, src_file in _pvp_unique_src_pairs(src_meta, key):
                src_path = src_meta.root / DEFAULT_VIDEO_PATH.format(
                    video_key=key, chunk_index=src_chunk, file_index=src_file,
                )
                if not src_path.is_file():
                    raise FileNotFoundError(
                        f"missing source mp4: {src_path} "
                        f"(camera={key}, src_meta={src_meta.repo_id})"
                    )
                src_size_bytes = src_path.stat().st_size
                src_duration_s = float(get_video_duration_in_s(src_path))

                # Decide: first-in-shard, rotate, or append.
                if current_shard is None or not current_shard.sources:
                    # First src ever for this camera (or empty shard after rotate).
                    if current_shard is None:
                        current_shard = ShardPlan(
                            key=key,
                            dst_chunk=chunk_idx,
                            dst_file=file_idx,
                            dst_path=aggr_root / DEFAULT_VIDEO_PATH.format(
                                video_key=key, chunk_index=chunk_idx, file_index=file_idx,
                            ),
                        )
                    offset_in_dst_s = 0.0
                    dst_size_estimate_bytes = src_size_bytes
                    dst_duration_estimate_s = src_duration_s
                elif dst_size_estimate_bytes + src_size_bytes >= size_cap_bytes:
                    # Rotate: close current shard, start a new one.
                    shards.append(current_shard)
                    chunk_idx, file_idx = update_chunk_file_indices(
                        chunk_idx, file_idx, chunk_size
                    )
                    current_shard = ShardPlan(
                        key=key,
                        dst_chunk=chunk_idx,
                        dst_file=file_idx,
                        dst_path=aggr_root / DEFAULT_VIDEO_PATH.format(
                            video_key=key, chunk_index=chunk_idx, file_index=file_idx,
                        ),
                    )
                    offset_in_dst_s = 0.0
                    dst_size_estimate_bytes = src_size_bytes
                    dst_duration_estimate_s = src_duration_s
                else:
                    # Append to the current open shard.
                    offset_in_dst_s = dst_duration_estimate_s
                    dst_size_estimate_bytes += src_size_bytes
                    dst_duration_estimate_s += src_duration_s

                current_shard.sources.append(SourceEntry(
                    src_path=src_path,
                    src_chunk=src_chunk,
                    src_file=src_file,
                    src_size_bytes=src_size_bytes,
                    src_duration_s=src_duration_s,
                    offset_in_dst_s=offset_in_dst_s,
                ))

                # Record per-source-file destination + offset for update_meta_data().
                src_view["src_to_offset"][(src_chunk, src_file)] = offset_in_dst_s
                last_dst_chunk = current_shard.dst_chunk
                last_dst_file = current_shard.dst_file

            # After all src files for this source, record where they ended up.
            src_view["chunk"] = last_dst_chunk
            src_view["file"] = last_dst_file
            per_source_view[src_meta.repo_id][key] = src_view

        # Flush the still-open shard at end-of-camera.
        if current_shard is not None and current_shard.sources:
            shards.append(current_shard)
        plan[key] = shards

    # Sanity: every (key, dst_chunk, dst_file) is unique across the plan.
    seen: set[tuple[str, int, int]] = set()
    for key, shards in plan.items():
        for s in shards:
            tup = (key, s.dst_chunk, s.dst_file)
            if tup in seen:
                raise AssertionError(
                    f"duplicate shard key in plan: {tup} — planner bug"
                )
            seen.add(tup)

    return plan, per_source_view


# ─────── Stage B: executor (runs each ShardPlan to disk) ───────


@_pvp_dataclass(frozen=True)
class ShardResult:
    ok: bool
    detail: str = ""
    wall_s: float = 0.0
    bytes_written: int = 0
    method: str = ""  # "copy" or "ffmpeg"


class ShardExecutionError(RuntimeError):
    """Raised by execute_video_plan() when one or more shards fail."""


def _pvp_use_planned_videos() -> bool:
    """Read LEROBOT_AGGREGATE_VIDEO_PLAN at call time. Default False.
    When True, aggregate_datasets() runs Stages A+B (plan + parallel
    execute) before the per-source loop, instead of calling
    aggregate_videos() per source. The existing online path remains
    the fallback when this is False."""
    raw = _viz_os.environ.get("LEROBOT_AGGREGATE_VIDEO_PLAN", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _pvp_workers_env() -> int:
    """Read LEROBOT_AGGREGATE_VIDEO_PLAN_WORKERS at call time. Default
    min(8, cpu_count()) capped at 1+."""
    raw = _viz_os.environ.get("LEROBOT_AGGREGATE_VIDEO_PLAN_WORKERS")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return min(8, _viz_os.cpu_count() or 4)


def _pvp_timeout_env() -> float:
    """Read LEROBOT_AGGREGATE_VIDEO_PLAN_TIMEOUT_S at call time. Default 1200."""
    raw = _viz_os.environ.get("LEROBOT_AGGREGATE_VIDEO_PLAN_TIMEOUT_S")
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return 1200.0


def _execute_shard(shard: ShardPlan) -> ShardResult:
    """Materialize one ShardPlan to disk.

    Single-source shards use shutil.copy (byte-identical, no re-mux).
    Multi-source shards use one ffmpeg subprocess with `-f concat -c copy`.
    Output writes are atomic: ffmpeg writes to a sibling temp file which
    is moved to the final path on success; failures clean up the temp.

    This function is pickled to ProcessPoolExecutor workers, so all of its
    imports come from the module-level prologue (subprocess, tempfile via
    _pvp_subprocess / _pvp_tempfile). The function is intentionally pure
    of any non-picklable closure state.
    """
    t0 = _viz_time.monotonic()

    if not shard.sources:
        return ShardResult(ok=False, detail="empty source list", wall_s=0.0)

    shard.dst_path.parent.mkdir(parents=True, exist_ok=True)

    # Single-source fast path: byte-identical with shutil.copy
    if len(shard.sources) == 1:
        try:
            shutil.copy(str(shard.sources[0].src_path), str(shard.dst_path))
        except Exception as exc:  # noqa: BLE001
            return ShardResult(
                ok=False, detail=f"copy failed: {exc!r}",
                wall_s=_viz_time.monotonic() - t0, method="copy",
            )
        return ShardResult(
            ok=True, method="copy",
            wall_s=_viz_time.monotonic() - t0,
            bytes_written=shard.dst_path.stat().st_size,
        )

    # Multi-source: ffmpeg concat demuxer with stream-copy
    list_path = None
    tmp_dst = shard.dst_path.with_suffix(".mp4.tmp")
    try:
        with _pvp_tempfile.NamedTemporaryFile(
            "w", suffix=".ffconcat", delete=False,
            dir=str(shard.dst_path.parent),
        ) as f:
            f.write("ffconcat version 1.0\n")
            for src in shard.sources:
                f.write(f"file '{src.src_path.resolve()}'\n")
            list_path = f.name

        proc = _pvp_subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "concat", "-safe", "0",
                "-i", list_path,
                "-c", "copy",
                "-movflags", "faststart",
                "-f", "mp4",     # force muxer; tmp suffix is .mp4.tmp
                str(tmp_dst),
            ],
            capture_output=True, text=True,
            timeout=_pvp_timeout_env(),
        )
        if proc.returncode != 0:
            return ShardResult(
                ok=False,
                detail=f"ffmpeg rc={proc.returncode}: {proc.stderr[-2000:]}",
                wall_s=_viz_time.monotonic() - t0, method="ffmpeg",
            )
        shutil.move(str(tmp_dst), str(shard.dst_path))
        return ShardResult(
            ok=True, method="ffmpeg",
            wall_s=_viz_time.monotonic() - t0,
            bytes_written=shard.dst_path.stat().st_size,
        )
    except _pvp_subprocess.TimeoutExpired:
        return ShardResult(
            ok=False,
            detail=f"ffmpeg timed out after {_pvp_timeout_env()}s",
            wall_s=_viz_time.monotonic() - t0, method="ffmpeg",
        )
    except Exception as exc:  # noqa: BLE001
        return ShardResult(
            ok=False, detail=f"ffmpeg invocation failed: {exc!r}",
            wall_s=_viz_time.monotonic() - t0, method="ffmpeg",
        )
    finally:
        if list_path is not None:
            Path(list_path).unlink(missing_ok=True)
        if tmp_dst.exists():
            tmp_dst.unlink(missing_ok=True)


def execute_video_plan(plan: dict, max_workers: int | None = None) -> dict:
    """Run every ShardPlan in `plan` to produce the output mp4s.

    Args:
      plan:        {camera_key: [ShardPlan, ...]} from plan_video_shards()
      max_workers: if None, read LEROBOT_AGGREGATE_VIDEO_PLAN_WORKERS env
                   (default min(8, cpu_count())). If 1, runs sequentially
                   in-process (useful for tests and deterministic logs).

    Returns a summary dict (per-shard timings, byte counts, method counts).
    Raises ShardExecutionError on any failure (all other shards complete
    first; partial outputs are NOT cleaned up because the per-shard
    tmp-then-rename pattern ensures only successful outputs are present).
    """
    tasks: list[ShardPlan] = [s for shards in plan.values() for s in shards]
    if not tasks:
        return {
            "n_shards": 0, "n_ok": 0, "wall_s_total": 0.0,
            "bytes_written_total": 0, "method_counts": {},
            "max_workers": 0, "per_shard": [],
        }

    workers = max_workers if max_workers is not None else _pvp_workers_env()
    workers = min(workers, len(tasks))  # don't over-allocate

    t0 = _viz_time.monotonic()
    results: list[tuple[ShardPlan, ShardResult]] = []

    if workers <= 1:
        # Sequential path; preferred for tests and small plans.
        for s in tasks:
            results.append((s, _execute_shard(s)))
    else:
        # Parallel path; one ffmpeg subprocess per shard, up to `workers`
        # concurrently. Order of completion is non-deterministic.
        with _pvp_pool.ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_shard = {pool.submit(_execute_shard, s): s for s in tasks}
            for fut in _pvp_pool.as_completed(future_to_shard):
                s = future_to_shard[fut]
                try:
                    r = fut.result()
                except Exception as exc:  # noqa: BLE001
                    r = ShardResult(ok=False, detail=f"worker exception: {exc!r}")
                results.append((s, r))

    method_counts: dict[str, int] = {}
    for _, r in results:
        if r.method:
            method_counts[r.method] = method_counts.get(r.method, 0) + 1

    summary = {
        "n_shards": len(results),
        "n_ok": sum(1 for _, r in results if r.ok),
        "wall_s_total": _viz_time.monotonic() - t0,
        "bytes_written_total": sum(r.bytes_written for _, r in results if r.ok),
        "method_counts": method_counts,
        "max_workers": workers,
        "per_shard": [
            {
                "key": s.key, "dst_chunk": s.dst_chunk, "dst_file": s.dst_file,
                "dst_path": str(s.dst_path), "n_sources": len(s.sources),
                "ok": r.ok, "method": r.method, "wall_s": r.wall_s,
                "bytes_written": r.bytes_written, "detail": r.detail,
            }
            for s, r in results
        ],
    }

    failures = [(s, r) for s, r in results if not r.ok]
    if failures:
        head = "\n".join(
            f"  {s.key} dst=({s.dst_chunk},{s.dst_file}) n_src={len(s.sources)}: {r.detail}"
            for s, r in failures[:5]
        )
        raise ShardExecutionError(
            f"{len(failures)}/{len(results)} shards failed:\n{head}"
        )

    return summary
# end LOCAL-PATCH(perf-video-plan)


def validate_all_metadata(all_metadata: list[LeRobotDatasetMetadata]):
    """Validates that all dataset metadata have consistent properties.

    Ensures all datasets have the same fps, robot_type, and features to guarantee
    compatibility when aggregating them into a single dataset.

    Args:
        all_metadata: List of LeRobotDatasetMetadata objects to validate.

    Returns:
        tuple: A tuple containing (fps, robot_type, features) from the first metadata.

    Raises:
        ValueError: If any metadata has different fps, robot_type, or features
                   than the first metadata in the list.
    """

    fps = all_metadata[0].fps
    robot_type = all_metadata[0].robot_type
    features = all_metadata[0].features

    for meta in tqdm.tqdm(all_metadata, desc="Validate all meta data"):
        if fps != meta.fps:
            raise ValueError(f"Same fps is expected, but got fps={meta.fps} instead of {fps}.")
        if robot_type != meta.robot_type:
            raise ValueError(
                f"Same robot_type is expected, but got robot_type={meta.robot_type} instead of {robot_type}."
            )
        if features != meta.features:
            raise ValueError(
                f"Same features is expected, but got features={meta.features} instead of {features}."
            )

    return fps, robot_type, features


def update_data_df(df, src_meta, dst_meta):
    """Updates a data DataFrame with new indices and task mappings for aggregation.

    Adjusts episode indices, frame indices, and task indices to account for
    previously aggregated data in the destination dataset.

    Args:
        df: DataFrame containing the data to be updated.
        src_meta: Source dataset metadata.
        dst_meta: Destination dataset metadata.

    Returns:
        pd.DataFrame: Updated DataFrame with adjusted indices.
    """

    df["episode_index"] = df["episode_index"] + dst_meta.info["total_episodes"]
    df["index"] = df["index"] + dst_meta.info["total_frames"]

    src_task_names = src_meta.tasks.index.take(df["task_index"].to_numpy())
    df["task_index"] = dst_meta.tasks.loc[src_task_names, "task_index"].to_numpy()

    return df


def update_meta_data(
    df,
    dst_meta,
    meta_idx,
    data_idx,
    videos_idx,
):
    """Updates metadata DataFrame with new chunk, file, and timestamp indices.

    Adjusts all indices and timestamps to account for previously aggregated
    data and videos in the destination dataset.

    Args:
        df: DataFrame containing the metadata to be updated.
        dst_meta: Destination dataset metadata.
        meta_idx: Dictionary containing current metadata chunk and file indices.
        data_idx: Dictionary containing current data chunk and file indices.
        videos_idx: Dictionary containing current video indices and timestamps.

    Returns:
        pd.DataFrame: Updated DataFrame with adjusted indices and timestamps.
    """

    df["meta/episodes/chunk_index"] = df["meta/episodes/chunk_index"] + meta_idx["chunk"]
    df["meta/episodes/file_index"] = df["meta/episodes/file_index"] + meta_idx["file"]
    df["data/chunk_index"] = df["data/chunk_index"] + data_idx["chunk"]
    df["data/file_index"] = df["data/file_index"] + data_idx["file"]
    for key, video_idx in videos_idx.items():
        # CONFLICT RESOLUTION (cherry-pick 90684a96 on top of local 96516e34):
        # Upstream's vectorized rewrite of this loop drops the per-source-file
        # `src_to_offset` logic added by our fork to handle source datasets
        # that span multiple video shards. Keeping the local version preserves
        # correctness; Phase 3 can vectorize the `if src_to_offset` branch
        # separately (see plan).
        # Store original video file indices before updating
        orig_chunk_col = f"videos/{key}/chunk_index"
        orig_file_col = f"videos/{key}/file_index"
        df["_orig_chunk"] = df[orig_chunk_col].copy()
        df["_orig_file"] = df[orig_file_col].copy()

        # Update chunk and file indices to point to destination
        df[orig_chunk_col] = video_idx["chunk"]
        df[orig_file_col] = video_idx["file"]

        # Apply per-source-file timestamp offsets
        src_to_offset = video_idx.get("src_to_offset", {})
        if src_to_offset:
            # Apply offset based on original source file
            for idx in df.index:
                src_key = (df.at[idx, "_orig_chunk"], df.at[idx, "_orig_file"])
                offset = src_to_offset.get(src_key, 0)
                df.at[idx, f"videos/{key}/from_timestamp"] += offset
                df.at[idx, f"videos/{key}/to_timestamp"] += offset
        else:
            # Fallback to simple offset (for backward compatibility)
            df[f"videos/{key}/from_timestamp"] = (
                df[f"videos/{key}/from_timestamp"] + video_idx["latest_duration"]
            )
            df[f"videos/{key}/to_timestamp"] = df[f"videos/{key}/to_timestamp"] + video_idx["latest_duration"]

        # Clean up temporary columns
        df = df.drop(columns=["_orig_chunk", "_orig_file"])

    df["dataset_from_index"] = df["dataset_from_index"] + dst_meta.info["total_frames"]
    df["dataset_to_index"] = df["dataset_to_index"] + dst_meta.info["total_frames"]
    df["episode_index"] = df["episode_index"] + dst_meta.info["total_episodes"]

    return df


def aggregate_datasets(
    repo_ids: list[str],
    aggr_repo_id: str,
    roots: list[Path] | None = None,
    aggr_root: Path | None = None,
    data_files_size_in_mb: float | None = None,
    video_files_size_in_mb: float | None = None,
    chunk_size: int | None = None,
):
    """Aggregates multiple LeRobot datasets into a single unified dataset.

    This is the main function that orchestrates the aggregation process by:
    1. Loading and validating all source dataset metadata
    2. Creating a new destination dataset with unified tasks
    3. Aggregating videos, data, and metadata from all source datasets
    4. Finalizing the aggregated dataset with proper statistics

    Args:
        repo_ids: List of repository IDs for the datasets to aggregate.
        aggr_repo_id: Repository ID for the aggregated output dataset.
        roots: Optional list of root paths for the source datasets.
        aggr_root: Optional root path for the aggregated dataset.
        data_files_size_in_mb: Maximum size for data files in MB (defaults to DEFAULT_DATA_FILE_SIZE_IN_MB)
        video_files_size_in_mb: Maximum size for video files in MB (defaults to DEFAULT_VIDEO_FILE_SIZE_IN_MB)
        chunk_size: Maximum number of files per chunk (defaults to DEFAULT_CHUNK_SIZE)
    """
    logging.info("Start aggregate_datasets")
    # LOCAL-PATCH(visibility): per-run timer + announce config (read at call time)
    _viz_timer = _VizPhaseTimer()
    _viz_inner_reset()  # LOCAL-PATCH(visibility-inner): reset per-run inner accumulators
    _parq_reset()  # LOCAL-PATCH(perf-parquet-stream): defensively close any leaked writers
    _viz_progress_every_call = _viz_progress_every()
    _viz_phase_timing_path_call = _viz_phase_timing_path()
    logging.info(
        "VIZ config: progress_every=%d phase_timing_path=%s",
        _viz_progress_every_call,
        _viz_phase_timing_path_call or "<unset>",
    )
    _viz_flush()
    # end LOCAL-PATCH(visibility)

    if data_files_size_in_mb is None:
        data_files_size_in_mb = DEFAULT_DATA_FILE_SIZE_IN_MB
    if video_files_size_in_mb is None:
        video_files_size_in_mb = DEFAULT_VIDEO_FILE_SIZE_IN_MB
    if chunk_size is None:
        chunk_size = DEFAULT_CHUNK_SIZE

    # LOCAL-PATCH(visibility): time metadata load + validate
    _t0 = _viz_time.monotonic()
    all_metadata = (
        [LeRobotDatasetMetadata(repo_id) for repo_id in repo_ids]
        if roots is None
        else [
            LeRobotDatasetMetadata(repo_id, root=root) for repo_id, root in zip(repo_ids, roots, strict=False)
        ]
    )
    _viz_timer.record("load_source_metadata", _viz_time.monotonic() - _t0)
    logging.info(
        "VIZ load_source_metadata: n_sources=%d wall_s=%.2f",
        len(all_metadata),
        _viz_timer.totals["load_source_metadata"],
    )
    _viz_flush()
    # end LOCAL-PATCH(visibility)
    fps, robot_type, features = _viz_time_call(_viz_timer, "validate_all_metadata", validate_all_metadata, all_metadata)  # LOCAL-PATCH(visibility)
    video_keys = [key for key in features if features[key]["dtype"] == "video"]

    dst_meta = LeRobotDatasetMetadata.create(
        repo_id=aggr_repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        root=aggr_root,
        use_videos=len(video_keys) > 0,
        chunks_size=chunk_size,
        data_files_size_in_mb=data_files_size_in_mb,
        video_files_size_in_mb=video_files_size_in_mb,
    )

    logging.info("Find all tasks")
    # LOCAL-PATCH(visibility): time task discovery
    _t0 = _viz_time.monotonic()
    unique_tasks = pd.concat([m.tasks for m in all_metadata]).index.unique()
    dst_meta.tasks = pd.DataFrame({"task_index": range(len(unique_tasks))}, index=unique_tasks)
    _viz_timer.record("find_all_tasks", _viz_time.monotonic() - _t0)
    logging.info(
        "VIZ find_all_tasks: unique_tasks=%d wall_s=%.2f",
        len(unique_tasks),
        _viz_timer.totals["find_all_tasks"],
    )
    _viz_flush()
    # end LOCAL-PATCH(visibility)

    meta_idx = {"chunk": 0, "file": 0}
    data_idx = {"chunk": 0, "file": 0}
    videos_idx = {
        key: {"chunk": 0, "file": 0, "latest_duration": 0, "episode_duration": 0} for key in video_keys
    }

    dst_meta.episodes = {}

    # LOCAL-PATCH(perf-video-plan): Stage C integration — run planner + executor
    # up front, then drive the per-source loop from per_source_view instead of
    # calling aggregate_videos() per source. Gated by env var; default False so
    # production stays on the online path until validated.
    _pvp_planned = _pvp_use_planned_videos()
    _pvp_per_source_view: dict[str, dict[str, dict]] = {}
    _pvp_exec_summary: dict | None = None
    if _pvp_planned:
        logging.info(
            "VIZ planned-videos: LEROBOT_AGGREGATE_VIDEO_PLAN=true; "
            "running plan_video_shards + execute_video_plan up front "
            "(workers=%d, timeout_s=%.0f)",
            _pvp_workers_env(), _pvp_timeout_env(),
        )
        _viz_flush()
        _pvp_plan, _pvp_per_source_view = _viz_time_call(
            _viz_timer, "plan_video_shards",
            plan_video_shards, all_metadata, dst_meta.root, video_keys,
            video_files_size_in_mb, chunk_size,
        )
        _pvp_n_shards = sum(len(s) for s in _pvp_plan.values())
        logging.info(
            "VIZ planned-videos: planned %d shards across %d cameras (took %.2fs)",
            _pvp_n_shards, len(video_keys),
            _viz_timer.totals.get("plan_video_shards", 0.0),
        )
        _viz_flush()
        _pvp_exec_summary = _viz_time_call(
            _viz_timer, "execute_video_plan",
            execute_video_plan, _pvp_plan, _pvp_workers_env(),
        )
        logging.info(
            "VIZ planned-videos: executed %d/%d shards (took %.2fs, "
            "%d bytes written, methods=%s)",
            _pvp_exec_summary["n_ok"], _pvp_exec_summary["n_shards"],
            _viz_timer.totals.get("execute_video_plan", 0.0),
            _pvp_exec_summary["bytes_written_total"],
            _pvp_exec_summary["method_counts"],
        )
        _viz_flush()
    # end LOCAL-PATCH(perf-video-plan)

    # LOCAL-PATCH(visibility): per-source enumeration + per-call timing + periodic progress logs
    _viz_n = len(all_metadata)
    for _viz_idx, src_meta in enumerate(tqdm.tqdm(all_metadata, desc="Copy data and videos")):
        _viz_src_t0 = _viz_time.monotonic()
        if _pvp_planned:  # LOCAL-PATCH(perf-video-plan)
            # Video work already done up front; just fetch this source's view.
            videos_idx = _pvp_per_source_view[src_meta.repo_id]
        else:
            videos_idx = _viz_time_call(
                _viz_timer, "aggregate_videos",
                aggregate_videos, src_meta, dst_meta, videos_idx, video_files_size_in_mb, chunk_size,
            )
        data_idx = _viz_time_call(
            _viz_timer, "aggregate_data",
            aggregate_data, src_meta, dst_meta, data_idx, data_files_size_in_mb, chunk_size,
        )
        meta_idx = _viz_time_call(
            _viz_timer, "aggregate_metadata",
            aggregate_metadata, src_meta, dst_meta, meta_idx, data_idx, videos_idx,
        )

        dst_meta.info["total_episodes"] += src_meta.total_episodes
        dst_meta.info["total_frames"] += src_meta.total_frames

        _viz_done = _viz_idx + 1
        if _viz_done % _viz_progress_every_call == 0 or _viz_done == _viz_n:
            _viz_per_src = _viz_time.monotonic() - _viz_src_t0
            _viz_running = _viz_timer.summary(_viz_done)
            _viz_phases_str = " ".join(
                f"{p['phase']}={p['wall_s']:.1f}s" for p in _viz_running["phases"]
            )
            logging.info(
                "VIZ PROGRESS src=%d/%d last_src_wall_s=%.2f total_wall_s=%.1f | %s",
                _viz_done, _viz_n, _viz_per_src, _viz_running["wall_s_total"], _viz_phases_str,
            )
            _viz_flush()
    # end LOCAL-PATCH(visibility)

    _viz_time_call(_viz_timer, "finalize_aggregation", finalize_aggregation, dst_meta, all_metadata)  # LOCAL-PATCH(visibility)
    # LOCAL-PATCH(perf-parquet-stream): close all open writers so parquet footers
    # are written before the merger exits and pi-data-sharing validate runs.
    _viz_time_call(_viz_timer, "parq_finalize_all", _parq_finalize_all)
    # end LOCAL-PATCH(perf-parquet-stream)
    logging.info("Aggregation complete.")

    # LOCAL-PATCH(visibility): emit final PHASE_TIMING JSON (always to log, optionally to file)
    _viz_summary = _viz_timer.summary(len(all_metadata))
    _viz_summary["aggr_repo_id"] = aggr_repo_id
    _viz_summary["aggr_root"] = str(aggr_root) if aggr_root is not None else None
    _viz_summary["config"] = {
        "data_files_size_in_mb": data_files_size_in_mb,
        "video_files_size_in_mb": video_files_size_in_mb,
        "chunk_size": chunk_size,
    }
    # LOCAL-PATCH(visibility-inner): attach per-branch breakdown for aggregate_videos
    _viz_summary["videos_inner"] = {
        name: dict(stats) for name, stats in _VIZ_INNER.items()
    }
    # Log a one-liner that's easy to grep alongside VIZ PROGRESS
    _viz_inner_line = " ".join(
        f"{name.split('.', 1)[1]}={s['wall_s']:.1f}s/{s['n_calls']}calls"
        for name, s in _VIZ_INNER.items()
    )
    logging.info("VIZ videos_inner: %s", _viz_inner_line)
    # end LOCAL-PATCH(visibility-inner)
    # LOCAL-PATCH(perf-video-plan): attach executor summary when planned path ran
    _viz_summary["planned_videos_enabled"] = _pvp_planned
    if _pvp_planned and _pvp_exec_summary is not None:
        _viz_summary["execute_video_plan"] = _pvp_exec_summary
    # end LOCAL-PATCH(perf-video-plan)
    # LOCAL-PATCH(perf-parquet-stream): record whether the streaming path was active
    _viz_summary["parquet_stream_enabled"] = _parq_stream_enabled()
    # end LOCAL-PATCH(perf-parquet-stream)
    logging.info("PHASE_TIMING %s", _viz_json.dumps(_viz_summary))
    if _viz_phase_timing_path_call:
        _viz_path = Path(_viz_phase_timing_path_call)
        _viz_path.parent.mkdir(parents=True, exist_ok=True)
        _viz_path.write_text(_viz_json.dumps(_viz_summary, indent=2))
        logging.info("VIZ wrote PHASE_TIMING summary to %s", _viz_path)
    _viz_flush()
    # end LOCAL-PATCH(visibility)


def aggregate_videos(src_meta, dst_meta, videos_idx, video_files_size_in_mb, chunk_size):
    """Aggregates video chunks from a source dataset into the destination dataset.

    Handles video file concatenation and rotation based on file size limits.
    Creates new video files when size limits are exceeded.

    Args:
        src_meta: Source dataset metadata.
        dst_meta: Destination dataset metadata.
        videos_idx: Dictionary tracking video chunk and file indices.
        video_files_size_in_mb: Maximum size for video files in MB (defaults to DEFAULT_VIDEO_FILE_SIZE_IN_MB)
        chunk_size: Maximum number of files per chunk (defaults to DEFAULT_CHUNK_SIZE)

    Returns:
        dict: Updated videos_idx with current chunk and file indices.
    """
    for key in videos_idx:
        videos_idx[key]["episode_duration"] = 0
        # Track offset for each source (chunk, file) pair
        videos_idx[key]["src_to_offset"] = {}

    for key, video_idx in videos_idx.items():
        unique_chunk_file_pairs = {
            (chunk, file)
            for chunk, file in zip(
                src_meta.episodes[f"videos/{key}/chunk_index"],
                src_meta.episodes[f"videos/{key}/file_index"],
                strict=False,
            )
        }
        unique_chunk_file_pairs = sorted(unique_chunk_file_pairs)

        chunk_idx = video_idx["chunk"]
        file_idx = video_idx["file"]
        current_offset = video_idx["latest_duration"]

        for src_chunk_idx, src_file_idx in unique_chunk_file_pairs:
            src_path = src_meta.root / DEFAULT_VIDEO_PATH.format(
                video_key=key,
                chunk_index=src_chunk_idx,
                file_index=src_file_idx,
            )

            dst_path = dst_meta.root / DEFAULT_VIDEO_PATH.format(
                video_key=key,
                chunk_index=chunk_idx,
                file_index=file_idx,
            )

            # LOCAL-PATCH(visibility-inner): time the PyAV duration probe
            _viz_t0 = _viz_time.monotonic()
            src_duration = get_video_duration_in_s(src_path)
            _viz_inner_record("videos.probe", _viz_time.monotonic() - _viz_t0)
            # end LOCAL-PATCH(visibility-inner)

            if not dst_path.exists():
                # Store offset before incrementing
                videos_idx[key]["src_to_offset"][(src_chunk_idx, src_file_idx)] = current_offset
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                # LOCAL-PATCH(visibility-inner): time first-in-shard copy
                _viz_t0 = _viz_time.monotonic()
                _viz_src_bytes = src_path.stat().st_size
                shutil.copy(str(src_path), str(dst_path))
                _viz_inner_record("videos.copy_first", _viz_time.monotonic() - _viz_t0, _viz_src_bytes)
                # end LOCAL-PATCH(visibility-inner)
                videos_idx[key]["episode_duration"] += src_duration
                current_offset += src_duration
                continue

            # Check file sizes before appending
            src_size = get_file_size_in_mb(src_path)
            dst_size = get_file_size_in_mb(dst_path)

            if dst_size + src_size >= video_files_size_in_mb:
                # Rotate to a new file, this source becomes start of new destination
                # So its offset should be 0
                videos_idx[key]["src_to_offset"][(src_chunk_idx, src_file_idx)] = 0
                chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, chunk_size)
                dst_path = dst_meta.root / DEFAULT_VIDEO_PATH.format(
                    video_key=key,
                    chunk_index=chunk_idx,
                    file_index=file_idx,
                )
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                # LOCAL-PATCH(visibility-inner): time size-rotation copy
                _viz_t0 = _viz_time.monotonic()
                _viz_src_bytes = src_path.stat().st_size
                shutil.copy(str(src_path), str(dst_path))
                _viz_inner_record("videos.copy_rotate", _viz_time.monotonic() - _viz_t0, _viz_src_bytes)
                # end LOCAL-PATCH(visibility-inner)
                # Reset offset for next file
                current_offset = src_duration
            else:
                # Append to existing video file - use current accumulated offset
                videos_idx[key]["src_to_offset"][(src_chunk_idx, src_file_idx)] = current_offset
                # LOCAL-PATCH(visibility-inner): time ffmpeg concat (bytes_in = src + existing dst, both are read)
                _viz_t0 = _viz_time.monotonic()
                _viz_bytes_in = src_path.stat().st_size + dst_path.stat().st_size
                concatenate_video_files(
                    [dst_path, src_path],
                    dst_path,
                )
                _viz_inner_record("videos.concat", _viz_time.monotonic() - _viz_t0, _viz_bytes_in)
                # end LOCAL-PATCH(visibility-inner)
                current_offset += src_duration

            videos_idx[key]["episode_duration"] += src_duration

        videos_idx[key]["chunk"] = chunk_idx
        videos_idx[key]["file"] = file_idx

    return videos_idx


def aggregate_data(src_meta, dst_meta, data_idx, data_files_size_in_mb, chunk_size):
    """Aggregates data chunks from a source dataset into the destination dataset.

    Reads source data files, updates indices to match the aggregated dataset,
    and writes them to the destination with proper file rotation.

    Args:
        src_meta: Source dataset metadata.
        dst_meta: Destination dataset metadata.
        data_idx: Dictionary tracking data chunk and file indices.

    Returns:
        dict: Updated data_idx with current chunk and file indices.
    """
    unique_chunk_file_ids = {
        (c, f)
        for c, f in zip(
            src_meta.episodes["data/chunk_index"], src_meta.episodes["data/file_index"], strict=False
        )
    }

    unique_chunk_file_ids = sorted(unique_chunk_file_ids)

    for src_chunk_idx, src_file_idx in unique_chunk_file_ids:
        src_path = src_meta.root / DEFAULT_DATA_PATH.format(
            chunk_index=src_chunk_idx, file_index=src_file_idx
        )
        df = pd.read_parquet(src_path)
        df = update_data_df(df, src_meta, dst_meta)

        data_idx = append_or_create_parquet_file(
            df,
            src_path,
            data_idx,
            data_files_size_in_mb,
            chunk_size,
            DEFAULT_DATA_PATH,
            contains_images=len(dst_meta.image_keys) > 0,
            aggr_root=dst_meta.root,
        )

    return data_idx


def aggregate_metadata(src_meta, dst_meta, meta_idx, data_idx, videos_idx):
    """Aggregates metadata from a source dataset into the destination dataset.

    Reads source metadata files, updates all indices and timestamps,
    and writes them to the destination with proper file rotation.

    Args:
        src_meta: Source dataset metadata.
        dst_meta: Destination dataset metadata.
        meta_idx: Dictionary tracking metadata chunk and file indices.
        data_idx: Dictionary tracking data chunk and file indices.
        videos_idx: Dictionary tracking video indices and timestamps.

    Returns:
        dict: Updated meta_idx with current chunk and file indices.
    """
    chunk_file_ids = {
        (c, f)
        for c, f in zip(
            src_meta.episodes["meta/episodes/chunk_index"],
            src_meta.episodes["meta/episodes/file_index"],
            strict=False,
        )
    }

    chunk_file_ids = sorted(chunk_file_ids)
    for chunk_idx, file_idx in chunk_file_ids:
        src_path = src_meta.root / DEFAULT_EPISODES_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        df = pd.read_parquet(src_path)
        df = update_meta_data(
            df,
            dst_meta,
            meta_idx,
            data_idx,
            videos_idx,
        )

        meta_idx = append_or_create_parquet_file(
            df,
            src_path,
            meta_idx,
            DEFAULT_DATA_FILE_SIZE_IN_MB,
            DEFAULT_CHUNK_SIZE,
            DEFAULT_EPISODES_PATH,
            contains_images=False,
            aggr_root=dst_meta.root,
        )

    # Increment latest_duration by the total duration added from this source dataset
    for k in videos_idx:
        videos_idx[k]["latest_duration"] += videos_idx[k]["episode_duration"]

    return meta_idx


# LOCAL-PATCH(perf-parquet-stream): Patch A — streaming parquet writer.
# Replaces append_or_create_parquet_file's O(N^2) read-modify-write of the
# destination parquet (pd.read_parquet -> pd.concat -> to_parquet on every
# source) with a pyarrow.ParquetWriter that's opened once per destination
# file and accumulates row groups as sources arrive. Total cost drops from
# O(N^2) to O(N) in source count.
#
# Gated by LEROBOT_AGGREGATE_PARQUET_STREAM env var; default off so the
# online RMW path remains the fallback. Open writers are tracked in a
# module-level dict keyed by destination Path; _parq_reset() is called at
# the top of aggregate_datasets() to defend against leaked state from a
# prior call, and _parq_finalize_all() is called at the end so every
# writer's footer is properly written.
#
# 500-ep cloud measurement (commit c12233bd4 Stage C only) showed
# aggregate_data at 202s with O(N^2) per-100-source growth (14.9s ->
# 26.0s -> 42.9s -> 61.0s). Linear extrapolation to 5000 ep is ~5.9h,
# uncomfortably close to the 4h Batch cap. Patch A projects ~30-40 min
# linear for 5000 ep, comfortably within cap with margin for >10k ep.
import pyarrow as _parq_pa
import pyarrow.parquet as _parq_papq


_PARQ_WRITERS: dict = {}


def _parq_stream_enabled() -> bool:
    """Read LEROBOT_AGGREGATE_PARQUET_STREAM at call time. Default False."""
    raw = _viz_os.environ.get("LEROBOT_AGGREGATE_PARQUET_STREAM", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _parq_reset() -> None:
    """Close any leaked writers from a previous call (defensive)."""
    for entry in list(_PARQ_WRITERS.values()):
        try:
            entry["writer"].close()
        except Exception:
            pass
    _PARQ_WRITERS.clear()


def _parq_finalize_all() -> None:
    """Close every open writer. Must be called at end of aggregate_datasets()
    so each destination parquet gets a valid footer."""
    for entry in _PARQ_WRITERS.values():
        entry["writer"].close()
    _PARQ_WRITERS.clear()


def _append_or_create_parquet_file_streaming(
    df: pd.DataFrame,
    src_path: Path,
    idx: dict[str, int],
    max_mb: float,
    chunk_size: int,
    default_path: str,
    aggr_root: Path,
) -> dict[str, int]:
    """Streaming variant of append_or_create_parquet_file().

    Maintains an open pyarrow.ParquetWriter per destination path in
    _PARQ_WRITERS. Each call writes one row group (the source's rows)
    and updates a cumulative size estimate. When the estimate would
    exceed max_mb, closes the current writer and opens a new one at
    the rotated (chunk, file) index.
    """
    table = _parq_pa.Table.from_pandas(df, preserve_index=False)
    src_size_mb = get_parquet_file_size_in_mb(src_path)

    dst_path = aggr_root / default_path.format(chunk_index=idx["chunk"], file_index=idx["file"])

    entry = _PARQ_WRITERS.get(dst_path)
    if entry is None:
        # First write to this destination.
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        writer = _parq_papq.ParquetWriter(str(dst_path), table.schema)
        writer.write_table(table)
        _PARQ_WRITERS[dst_path] = {
            "writer": writer,
            "schema": table.schema,
            "size_estimate_mb": src_size_mb,
        }
        return idx

    if entry["size_estimate_mb"] + src_size_mb >= max_mb:
        # Rotate: close current writer, open a new one at the next index.
        entry["writer"].close()
        del _PARQ_WRITERS[dst_path]
        idx["chunk"], idx["file"] = update_chunk_file_indices(
            idx["chunk"], idx["file"], chunk_size
        )
        new_path = aggr_root / default_path.format(
            chunk_index=idx["chunk"], file_index=idx["file"]
        )
        new_path.parent.mkdir(parents=True, exist_ok=True)
        writer = _parq_papq.ParquetWriter(str(new_path), table.schema)
        writer.write_table(table)
        _PARQ_WRITERS[new_path] = {
            "writer": writer,
            "schema": table.schema,
            "size_estimate_mb": src_size_mb,
        }
        return idx

    # Append to the open writer for this destination. Schemas can drift
    # across sources because `update_meta_data` mutates `from_timestamp` /
    # `to_timestamp` via `df.at[i, c] += offset`, which pandas treats as
    # in-place truncation when the column dtype is integer (or vice versa
    # when the value falls cleanly on an int). The online pd.concat path
    # silently coerces; we replicate the same behavior via pyarrow.cast
    # with safe=False so the merger doesn't crash on naturally-mixed-type
    # source data. Genuinely incompatible types (e.g., string vs int) will
    # still raise from cast().
    if table.schema != entry["schema"]:
        try:
            table = table.cast(entry["schema"], safe=False)
        except (_parq_pa.ArrowInvalid, _parq_pa.ArrowNotImplementedError,
                _parq_pa.ArrowTypeError) as exc:
            raise ValueError(
                f"streaming parquet append: schema mismatch for {dst_path} "
                f"and cast to writer schema failed: {exc}"
            ) from exc
    entry["writer"].write_table(table)
    entry["size_estimate_mb"] += src_size_mb
    return idx
# end LOCAL-PATCH(perf-parquet-stream)


def append_or_create_parquet_file(
    df: pd.DataFrame,
    src_path: Path,
    idx: dict[str, int],
    max_mb: float,
    chunk_size: int,
    default_path: str,
    contains_images: bool = False,
    aggr_root: Path = None,
):
    """Appends data to an existing parquet file or creates a new one based on size constraints.

    Manages file rotation when size limits are exceeded to prevent individual files
    from becoming too large. Handles both regular parquet files and those containing images.

    Args:
        df: DataFrame to write to the parquet file.
        src_path: Path to the source file (used for size estimation).
        idx: Dictionary containing current 'chunk' and 'file' indices.
        max_mb: Maximum allowed file size in MB before rotation.
        chunk_size: Maximum number of files per chunk before incrementing chunk index.
        default_path: Format string for generating file paths.
        contains_images: Whether the data contains images requiring special handling.
        aggr_root: Root path for the aggregated dataset.

    Returns:
        dict: Updated index dictionary with current chunk and file indices.
    """
    # LOCAL-PATCH(perf-parquet-stream): dispatch to the streaming writer when
    # enabled. The image-bearing path uses to_parquet_with_hf_images which is
    # not supported by ParquetWriter; fall through to the online path for
    # contains_images=True so image datasets continue to work.
    if _parq_stream_enabled() and not contains_images:
        return _append_or_create_parquet_file_streaming(
            df, src_path, idx, max_mb, chunk_size, default_path, aggr_root,
        )
    # end LOCAL-PATCH(perf-parquet-stream)

    dst_path = aggr_root / default_path.format(chunk_index=idx["chunk"], file_index=idx["file"])

    if not dst_path.exists():
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if contains_images:
            to_parquet_with_hf_images(df, dst_path)
        else:
            df.to_parquet(dst_path)
        return idx

    src_size = get_parquet_file_size_in_mb(src_path)
    dst_size = get_parquet_file_size_in_mb(dst_path)

    if dst_size + src_size >= max_mb:
        idx["chunk"], idx["file"] = update_chunk_file_indices(idx["chunk"], idx["file"], chunk_size)
        new_path = aggr_root / default_path.format(chunk_index=idx["chunk"], file_index=idx["file"])
        new_path.parent.mkdir(parents=True, exist_ok=True)
        final_df = df
        target_path = new_path
    else:
        existing_df = pd.read_parquet(dst_path)
        final_df = pd.concat([existing_df, df], ignore_index=True)
        target_path = dst_path

    if contains_images:
        to_parquet_with_hf_images(final_df, target_path)
    else:
        final_df.to_parquet(target_path)

    return idx


def finalize_aggregation(aggr_meta, all_metadata):
    """Finalizes the dataset aggregation by writing summary files and statistics.

    Writes the tasks file, info file with total counts and splits, and
    aggregated statistics from all source datasets.

    Args:
        aggr_meta: Aggregated dataset metadata.
        all_metadata: List of all source dataset metadata objects.
    """
    logging.info("write tasks")
    write_tasks(aggr_meta.tasks, aggr_meta.root)

    logging.info("write info")
    aggr_meta.info.update(
        {
            "total_tasks": len(aggr_meta.tasks),
            "total_episodes": sum(m.total_episodes for m in all_metadata),
            "total_frames": sum(m.total_frames for m in all_metadata),
            "splits": {"train": f"0:{sum(m.total_episodes for m in all_metadata)}"},
        }
    )
    write_info(aggr_meta.info, aggr_meta.root)

    logging.info("write stats")
    aggr_meta.stats = aggregate_stats([m.stats for m in all_metadata])
    write_stats(aggr_meta.stats, aggr_meta.root)
