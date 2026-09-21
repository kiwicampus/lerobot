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

import contextlib
import copy
import logging
import os
import shutil
import subprocess
import tempfile
import time
from concurrent import futures
from dataclasses import dataclass, field
from pathlib import Path

import datasets
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm

from lerobot.configs import VIDEO_ENCODER_INFO_KEYS

from .compute_stats import aggregate_stats
from .dataset_metadata import LeRobotDatasetMetadata
from .feature_utils import features_equal_for_merge, get_hf_features_from_features
from .io_utils import (
    get_file_size_in_mb,
    get_parquet_file_size_in_mb,
    to_parquet_one_row_group_per_episode,
    to_parquet_with_hf_images,
    write_info,
    write_stats,
    write_tasks,
)
from .utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_DATA_PATH,
    DEFAULT_EPISODES_PATH,
    DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    DEFAULT_VIDEO_PATH,
    update_chunk_file_indices,
)
from .video_utils import concatenate_video_files, get_video_duration_in_s


class IncompatibleSourceError(ValueError):
    """Raised by plan_video_shards() when sources for a camera disagree on
    codec / pix_fmt / resolution / fps. Aborts the merge before any I/O."""


@dataclass(frozen=True)
class SourceEntry:
    src_path: Path
    src_chunk: int
    src_file: int
    src_size_bytes: int
    src_duration_s: float
    offset_in_dst_s: float


@dataclass
class ShardPlan:
    key: str
    dst_chunk: int
    dst_file: int
    dst_path: Path
    sources: list = field(default_factory=list)  # list[SourceEntry]

    @property
    def total_size_bytes(self) -> int:
        return sum(s.src_size_bytes for s in self.sources)

    @property
    def total_duration_s(self) -> float:
        return sum(s.src_duration_s for s in self.sources)


def _pvp_unique_src_pairs(src_meta, key) -> list[tuple[int, int]]:
    """Sorted unique (chunk, file) pairs for `key` in this source's episodes
    metadata. Same logic the online aggregate_videos uses."""
    return sorted(
        {
            (chunk, file)
            for chunk, file in zip(
                src_meta.episodes[f"videos/{key}/chunk_index"],
                src_meta.episodes[f"videos/{key}/file_index"],
                strict=False,
            )
        }
    )


def _pvp_validate_camera_consistency(all_metadata, key: str) -> None:
    """All source datasets must agree on codec / pix_fmt / resolution / fps
    for the given camera, otherwise ffmpeg -c copy concat will fail at
    runtime. We catch this up-front so the merge aborts cleanly."""
    if not all_metadata:
        return
    ref_feat = all_metadata[0].features.get(key)
    if ref_feat is None:
        raise IncompatibleSourceError(f"camera '{key}' missing from first source's features")
    ref_info = ref_feat.get("info", {}) or {}
    ref_shape = ref_feat.get("shape")
    keys_to_check = ("video.codec", "video.pix_fmt", "video.fps", "video.height", "video.width")
    for src_meta in all_metadata[1:]:
        f = src_meta.features.get(key)
        if f is None:
            raise IncompatibleSourceError(f"camera '{key}' missing from source {src_meta.repo_id}")
        info = f.get("info", {}) or {}
        if f.get("shape") != ref_shape:
            raise IncompatibleSourceError(
                f"camera '{key}' shape mismatch: ref={ref_shape} vs {src_meta.repo_id}={f.get('shape')}"
            )
        for k in keys_to_check:
            if info.get(k) != ref_info.get(k):
                raise IncompatibleSourceError(
                    f"camera '{key}' {k} mismatch: ref={ref_info.get(k)} vs {src_meta.repo_id}={info.get(k)}"
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
                "episode_duration": 0.0,  # see "contract trick" in design doc
                "src_to_offset": {},
            }
            # Track what we record into the view per (src_chunk, src_file).
            last_dst_chunk = chunk_idx
            last_dst_file = file_idx

            for src_chunk, src_file in _pvp_unique_src_pairs(src_meta, key):
                src_path = src_meta.root / DEFAULT_VIDEO_PATH.format(
                    video_key=key,
                    chunk_index=src_chunk,
                    file_index=src_file,
                )
                if not src_path.is_file():
                    raise FileNotFoundError(
                        f"missing source mp4: {src_path} (camera={key}, src_meta={src_meta.repo_id})"
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
                            dst_path=aggr_root
                            / DEFAULT_VIDEO_PATH.format(
                                video_key=key,
                                chunk_index=chunk_idx,
                                file_index=file_idx,
                            ),
                        )
                    offset_in_dst_s = 0.0
                    dst_size_estimate_bytes = src_size_bytes
                    dst_duration_estimate_s = src_duration_s
                elif dst_size_estimate_bytes + src_size_bytes >= size_cap_bytes:
                    # Rotate: close current shard, start a new one.
                    shards.append(current_shard)
                    chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, chunk_size)
                    current_shard = ShardPlan(
                        key=key,
                        dst_chunk=chunk_idx,
                        dst_file=file_idx,
                        dst_path=aggr_root
                        / DEFAULT_VIDEO_PATH.format(
                            video_key=key,
                            chunk_index=chunk_idx,
                            file_index=file_idx,
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

                current_shard.sources.append(
                    SourceEntry(
                        src_path=src_path,
                        src_chunk=src_chunk,
                        src_file=src_file,
                        src_size_bytes=src_size_bytes,
                        src_duration_s=src_duration_s,
                        offset_in_dst_s=offset_in_dst_s,
                    )
                )

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
                raise AssertionError(f"duplicate shard key in plan: {tup} — planner bug")
            seen.add(tup)

    return plan, per_source_view


# ─────── Stage B: executor (runs each ShardPlan to disk) ───────


@dataclass(frozen=True)
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
    raw = os.environ.get("LEROBOT_AGGREGATE_VIDEO_PLAN", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _pvp_workers_env() -> int:
    """Read LEROBOT_AGGREGATE_VIDEO_PLAN_WORKERS at call time. Default
    min(8, cpu_count()) capped at 1+."""
    raw = os.environ.get("LEROBOT_AGGREGATE_VIDEO_PLAN_WORKERS")
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return min(8, os.cpu_count() or 4)


def _pvp_timeout_env() -> float:
    """Read LEROBOT_AGGREGATE_VIDEO_PLAN_TIMEOUT_S at call time. Default 1200."""
    raw = os.environ.get("LEROBOT_AGGREGATE_VIDEO_PLAN_TIMEOUT_S")
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
    subprocess / tempfile). The function is intentionally pure
    of any non-picklable closure state.
    """
    t0 = time.monotonic()

    if not shard.sources:
        return ShardResult(ok=False, detail="empty source list", wall_s=0.0)

    shard.dst_path.parent.mkdir(parents=True, exist_ok=True)

    # Single-source fast path: byte-identical with shutil.copy
    if len(shard.sources) == 1:
        try:
            shutil.copy(str(shard.sources[0].src_path), str(shard.dst_path))
        except Exception as exc:  # noqa: BLE001
            return ShardResult(
                ok=False,
                detail=f"copy failed: {exc!r}",
                wall_s=time.monotonic() - t0,
                method="copy",
            )
        return ShardResult(
            ok=True,
            method="copy",
            wall_s=time.monotonic() - t0,
            bytes_written=shard.dst_path.stat().st_size,
        )

    # Multi-source: ffmpeg concat demuxer with stream-copy
    list_path = None
    tmp_dst = shard.dst_path.with_suffix(".mp4.tmp")
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            suffix=".ffconcat",
            delete=False,
            dir=str(shard.dst_path.parent),
        ) as f:
            f.write("ffconcat version 1.0\n")
            for src in shard.sources:
                f.write(f"file '{src.src_path.resolve()}'\n")
            list_path = f.name

        proc = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                list_path,
                "-c",
                "copy",
                "-movflags",
                "faststart",
                "-f",
                "mp4",  # force muxer; tmp suffix is .mp4.tmp
                str(tmp_dst),
            ],
            capture_output=True,
            text=True,
            timeout=_pvp_timeout_env(),
        )
        if proc.returncode != 0:
            return ShardResult(
                ok=False,
                detail=f"ffmpeg rc={proc.returncode}: {proc.stderr[-2000:]}",
                wall_s=time.monotonic() - t0,
                method="ffmpeg",
            )
        shutil.move(str(tmp_dst), str(shard.dst_path))
        return ShardResult(
            ok=True,
            method="ffmpeg",
            wall_s=time.monotonic() - t0,
            bytes_written=shard.dst_path.stat().st_size,
        )
    except subprocess.TimeoutExpired:
        return ShardResult(
            ok=False,
            detail=f"ffmpeg timed out after {_pvp_timeout_env()}s",
            wall_s=time.monotonic() - t0,
            method="ffmpeg",
        )
    except Exception as exc:  # noqa: BLE001
        return ShardResult(
            ok=False,
            detail=f"ffmpeg invocation failed: {exc!r}",
            wall_s=time.monotonic() - t0,
            method="ffmpeg",
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
            "n_shards": 0,
            "n_ok": 0,
            "wall_s_total": 0.0,
            "bytes_written_total": 0,
            "method_counts": {},
            "max_workers": 0,
            "per_shard": [],
        }

    workers = max_workers if max_workers is not None else _pvp_workers_env()
    workers = min(workers, len(tasks))  # don't over-allocate

    t0 = time.monotonic()
    results: list[tuple[ShardPlan, ShardResult]] = []

    if workers <= 1:
        # Sequential path; preferred for tests and small plans.
        for s in tasks:
            results.append((s, _execute_shard(s)))
    else:
        # Parallel path; one ffmpeg subprocess per shard, up to `workers`
        # concurrently. Order of completion is non-deterministic.
        with futures.ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_shard = {pool.submit(_execute_shard, s): s for s in tasks}
            for fut in futures.as_completed(future_to_shard):
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
        "wall_s_total": time.monotonic() - t0,
        "bytes_written_total": sum(r.bytes_written for _, r in results if r.ok),
        "method_counts": method_counts,
        "max_workers": workers,
        "per_shard": [
            {
                "key": s.key,
                "dst_chunk": s.dst_chunk,
                "dst_file": s.dst_file,
                "dst_path": str(s.dst_path),
                "n_sources": len(s.sources),
                "ok": r.ok,
                "method": r.method,
                "wall_s": r.wall_s,
                "bytes_written": r.bytes_written,
                "detail": r.detail,
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
        raise ShardExecutionError(f"{len(failures)}/{len(results)} shards failed:\n{head}")

    return summary


# end LOCAL-PATCH(perf-video-plan)


def merge_video_feature_info_for_aggregate(all_metadata: list[LeRobotDatasetMetadata]) -> dict[str, dict]:
    """Create a merged video feature info dictionary for aggregation. The video encoder info is merged field-by-field: each key is kept only when every source agrees; otherwise that key is set to ``null`` (or ``{}`` for ``video.extra_options``) and a warning is logged.

    Args:
        all_metadata: List of LeRobotDatasetMetadata objects to merge.

    Returns:
        dict: A dictionary of merged video feature info.
    """
    merged_info = copy.deepcopy(all_metadata[0].features)
    video_keys = [k for k in merged_info if merged_info[k].get("dtype") == "video"]

    for vk in video_keys:
        video_infos = [m.features.get(vk, {}).get("info") or {} for m in all_metadata]
        base_video_info = video_infos[0]

        merged_encoder_info: dict = {}
        fallback_keys: list[str] = []
        for info_key in VIDEO_ENCODER_INFO_KEYS:
            values = [info.get(info_key, None) for info in video_infos]
            first_value = values[0]
            all_match = all(v == first_value for v in values[1:])

            if all_match:
                merged_encoder_info[info_key] = first_value
            else:
                fallback_keys.append(info_key)
                merged_encoder_info[info_key] = {} if info_key == "video.extra_options" else None

        if fallback_keys:
            logging.warning(
                f"Merging heterogeneous or incomplete video encoder metadata for feature {vk}. "
                f"Setting these keys to null: {fallback_keys}.",
            )

        merged_info[vk]["info"] = {**base_video_info, **merged_encoder_info}
        # TODO(CarolinePascal): make this variable once we have support for other video backends.
        merged_info[vk]["info"]["video.video_backend"] = "pyav"

    return merged_info


def validate_all_metadata(all_metadata: list[LeRobotDatasetMetadata]):
    """Validates that all dataset metadata have consistent properties.

    Ensures all datasets have the same fps, robot_type, and features to guarantee
    compatibility when aggregating them into a single dataset.
    Video encoder info is not considered for validation but is merged during aggregation in ``merge_video_feature_info_for_aggregate``.

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
        if not features_equal_for_merge(features, meta.features):
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

    df["episode_index"] = df["episode_index"] + dst_meta.info.total_episodes
    df["index"] = df["index"] + dst_meta.info.total_frames

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

    For data file indices, uses the 'src_to_dst' mapping from aggregate_data()
    to correctly map source file indices to their destination locations.

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

    # Update data file indices using source-to-destination mapping
    # This is critical for handling datasets that are already results of a merge
    data_src_to_dst = data_idx.get("src_to_dst", {})
    if data_src_to_dst:
        # Store original indices for lookup
        df["_orig_data_chunk"] = df["data/chunk_index"].copy()
        df["_orig_data_file"] = df["data/file_index"].copy()

        # Vectorized mapping from (src_chunk, src_file) to (dst_chunk, dst_file)
        # This is much faster than per-row iteration for large metadata tables
        mapping_index = pd.MultiIndex.from_tuples(
            list(data_src_to_dst.keys()),
            names=["chunk_index", "file_index"],
        )
        mapping_values = list(data_src_to_dst.values())
        mapping_df = pd.DataFrame(
            mapping_values,
            index=mapping_index,
            columns=["dst_chunk", "dst_file"],
        )

        # Construct a MultiIndex for each row based on original data indices
        row_index = pd.MultiIndex.from_arrays(
            [df["_orig_data_chunk"], df["_orig_data_file"]],
            names=["chunk_index", "file_index"],
        )

        # Align mapping to rows; missing keys fall back to the default destination
        reindexed = mapping_df.reindex(row_index)
        reindexed[["dst_chunk", "dst_file"]] = reindexed[["dst_chunk", "dst_file"]].fillna(
            {"dst_chunk": data_idx["chunk"], "dst_file": data_idx["file"]}
        )

        # Assign mapped destination indices back to the DataFrame
        df["data/chunk_index"] = reindexed["dst_chunk"].to_numpy()
        df["data/file_index"] = reindexed["dst_file"].to_numpy()

        # Clean up temporary columns
        df = df.drop(columns=["_orig_data_chunk", "_orig_data_file"])
    else:
        # Fallback to simple offset (backward compatibility for single-file sources)
        df["data/chunk_index"] = df["data/chunk_index"] + data_idx["chunk"]
        df["data/file_index"] = df["data/file_index"] + data_idx["file"]
    # Per-episode sources store videos/*/from_timestamp as int64, and adding a
    # fractional second offset to an int64 column floors it silently, shifting
    # every merged episode's video start. Offsets are seconds, so coerce first.
    for key in videos_idx:
        for col in (f"videos/{key}/from_timestamp", f"videos/{key}/to_timestamp"):
            if col in df.columns:
                df[col] = df[col].astype("float64")
    for key, video_idx in videos_idx.items():
        # Store original video file indices before updating
        orig_chunk_col = f"videos/{key}/chunk_index"
        orig_file_col = f"videos/{key}/file_index"
        df["_orig_chunk"] = df[orig_chunk_col].copy()
        df["_orig_file"] = df[orig_file_col].copy()

        # Get mappings for this video key
        src_to_offset = video_idx.get("src_to_offset", {})
        src_to_dst = video_idx.get("src_to_dst", {})

        # Apply per-source-file mappings
        if src_to_dst:
            # Map each episode to its correct destination file and apply offset
            for idx in df.index:
                src_key = (df.at[idx, "_orig_chunk"], df.at[idx, "_orig_file"])

                # Get destination chunk/file for this source file
                dst_chunk, dst_file = src_to_dst.get(src_key, (video_idx["chunk"], video_idx["file"]))
                df.at[idx, orig_chunk_col] = dst_chunk
                df.at[idx, orig_file_col] = dst_file

                # Apply timestamp offset
                offset = src_to_offset.get(src_key, 0)
                df.at[idx, f"videos/{key}/from_timestamp"] += offset
                df.at[idx, f"videos/{key}/to_timestamp"] += offset
        elif src_to_offset:
            # Fallback: use same destination for all, but apply per-file offsets
            df[orig_chunk_col] = video_idx["chunk"]
            df[orig_file_col] = video_idx["file"]
            for idx in df.index:
                src_key = (df.at[idx, "_orig_chunk"], df.at[idx, "_orig_file"])
                offset = src_to_offset.get(src_key, 0)
                df.at[idx, f"videos/{key}/from_timestamp"] += offset
                df.at[idx, f"videos/{key}/to_timestamp"] += offset
        else:
            # Fallback to simple offset (for backward compatibility)
            df[orig_chunk_col] = video_idx["chunk"]
            df[orig_file_col] = video_idx["file"]
            df[f"videos/{key}/from_timestamp"] = (
                df[f"videos/{key}/from_timestamp"] + video_idx["latest_duration"]
            )
            df[f"videos/{key}/to_timestamp"] = df[f"videos/{key}/to_timestamp"] + video_idx["latest_duration"]

        # Clean up temporary columns
        df = df.drop(columns=["_orig_chunk", "_orig_file"])

    df["dataset_from_index"] = df["dataset_from_index"] + dst_meta.info.total_frames
    df["dataset_to_index"] = df["dataset_to_index"] + dst_meta.info.total_frames
    df["episode_index"] = df["episode_index"] + dst_meta.info.total_episodes

    return df


def aggregate_datasets(
    repo_ids: list[str],
    aggr_repo_id: str,
    roots: list[Path] | None = None,
    aggr_root: Path | None = None,
    data_files_size_in_mb: int | None = None,
    video_files_size_in_mb: int | None = None,
    chunk_size: int | None = None,
    concatenate_videos: bool = True,
    concatenate_data: bool = True,
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
        concatenate_videos: When False, keep one mp4 per source file instead of packing into shards.
        concatenate_data: When False, keep one parquet per source file instead of packing into shards.
    """
    logging.info("Start aggregate_datasets")

    if data_files_size_in_mb is None:
        data_files_size_in_mb = DEFAULT_DATA_FILE_SIZE_IN_MB
    if video_files_size_in_mb is None:
        video_files_size_in_mb = DEFAULT_VIDEO_FILE_SIZE_IN_MB
    if chunk_size is None:
        chunk_size = DEFAULT_CHUNK_SIZE

    all_metadata = (
        [LeRobotDatasetMetadata(repo_id) for repo_id in repo_ids]
        if roots is None
        else [
            LeRobotDatasetMetadata(repo_id, root=root) for repo_id, root in zip(repo_ids, roots, strict=False)
        ]
    )
    fps, robot_type, _ = validate_all_metadata(all_metadata)
    features = merge_video_feature_info_for_aggregate(all_metadata)
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
    unique_tasks = pd.concat([m.tasks for m in all_metadata]).index.unique()
    dst_meta.tasks = pd.DataFrame(
        {"task_index": range(len(unique_tasks))}, index=pd.Index(unique_tasks, name="task")
    )

    meta_idx = {"chunk": 0, "file": 0}
    data_idx = {"chunk": 0, "file": 0}
    videos_idx = {
        key: {"chunk": 0, "file": 0, "latest_duration": 0, "episode_duration": 0} for key in video_keys
    }

    dst_meta.episodes = {}

    _parq_reset()

    # Plan every destination shard up front and build them in parallel, instead
    # of concatenating per source. Needs shard packing, so it stays off when
    # concatenate_videos is False.
    use_plan = _pvp_use_planned_videos() and concatenate_videos and len(video_keys) > 0
    per_source_view: dict[str, dict[str, dict]] = {}
    if use_plan:
        logging.info(
            "Planned videos enabled: planning shards (workers=%d, timeout_s=%.0f)",
            _pvp_workers_env(),
            _pvp_timeout_env(),
        )
        plan, per_source_view = plan_video_shards(
            all_metadata, dst_meta.root, video_keys, video_files_size_in_mb, chunk_size
        )
        exec_summary = execute_video_plan(plan, _pvp_workers_env())
        logging.info(
            "Planned videos: executed %d/%d shards, %d bytes written, methods=%s",
            exec_summary["n_ok"],
            exec_summary["n_shards"],
            exec_summary["bytes_written_total"],
            exec_summary["method_counts"],
        )

    for src_meta in tqdm.tqdm(all_metadata, desc="Copy data and videos"):
        if use_plan:
            videos_idx = per_source_view[src_meta.repo_id]
        else:
            videos_idx = aggregate_videos(
                src_meta, dst_meta, videos_idx, video_files_size_in_mb, chunk_size, concatenate_videos
            )
        data_idx = aggregate_data(
            src_meta, dst_meta, data_idx, data_files_size_in_mb, chunk_size, concatenate_data
        )

        meta_idx = aggregate_metadata(src_meta, dst_meta, meta_idx, data_idx, videos_idx)

        # Clear the src_to_dst mapping after processing each source dataset
        # to avoid interference between different source datasets
        data_idx.pop("src_to_dst", None)

        dst_meta.info.total_episodes += src_meta.total_episodes
        dst_meta.info.total_frames += src_meta.total_frames

    _parq_finalize_all()

    finalize_aggregation(dst_meta, all_metadata)
    logging.info("Aggregation complete.")


def aggregate_videos(
    src_meta, dst_meta, videos_idx, video_files_size_in_mb, chunk_size, concatenate_videos=True
):
    """Aggregates video chunks from a source dataset into the destination dataset.

    Handles video file concatenation and rotation based on file size limits.
    Creates new video files when size limits are exceeded.

    Args:
        src_meta: Source dataset metadata.
        dst_meta: Destination dataset metadata.
        videos_idx: Dictionary tracking video chunk and file indices.
        video_files_size_in_mb: Maximum size for video files in MB (defaults to DEFAULT_VIDEO_FILE_SIZE_IN_MB)
        chunk_size: Maximum number of files per chunk (defaults to DEFAULT_CHUNK_SIZE)
        concatenate_videos: When False, keep one mp4 per source file instead of packing into shards.
    Returns:
        dict: Updated videos_idx with current chunk and file indices.
    """
    for key in videos_idx:
        videos_idx[key]["episode_duration"] = 0
        # Track offset for each source (chunk, file) pair
        videos_idx[key]["src_to_offset"] = {}
        # Track destination (chunk, file) for each source (chunk, file) pair
        videos_idx[key]["src_to_dst"] = {}
        # Initialize dst_file_durations if not present
        # dst_file_durations tracks duration of each destination file
        if "dst_file_durations" not in videos_idx[key]:
            videos_idx[key]["dst_file_durations"] = {}

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
        dst_file_durations = video_idx["dst_file_durations"]

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

            src_duration = get_video_duration_in_s(src_path)
            dst_key = (chunk_idx, file_idx)

            if not dst_path.exists():
                # New destination file: offset is 0
                videos_idx[key]["src_to_offset"][(src_chunk_idx, src_file_idx)] = 0
                videos_idx[key]["src_to_dst"][(src_chunk_idx, src_file_idx)] = dst_key
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(src_path), str(dst_path))
                # Track duration of this destination file
                dst_file_durations[dst_key] = src_duration
                videos_idx[key]["episode_duration"] += src_duration
                continue

            # Check file sizes before appending
            src_size = get_file_size_in_mb(src_path)
            dst_size = get_file_size_in_mb(dst_path)

            if not concatenate_videos or dst_size + src_size >= video_files_size_in_mb:
                # Rotate to a new file - offset is 0
                chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, chunk_size)
                dst_key = (chunk_idx, file_idx)
                videos_idx[key]["src_to_offset"][(src_chunk_idx, src_file_idx)] = 0
                videos_idx[key]["src_to_dst"][(src_chunk_idx, src_file_idx)] = dst_key
                dst_path = dst_meta.root / DEFAULT_VIDEO_PATH.format(
                    video_key=key,
                    chunk_index=chunk_idx,
                    file_index=file_idx,
                )
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(src_path), str(dst_path))
                # Track duration of this new destination file
                dst_file_durations[dst_key] = src_duration
            else:
                # Append to existing destination file
                # Offset is the current duration of this destination file
                current_dst_duration = dst_file_durations.get(dst_key, 0)
                videos_idx[key]["src_to_offset"][(src_chunk_idx, src_file_idx)] = current_dst_duration
                videos_idx[key]["src_to_dst"][(src_chunk_idx, src_file_idx)] = dst_key
                # TODO(CarolinePascal): Move the check before the loop to avoid failing in the middle + add possibility to re-encode the video if the check fails
                concatenate_video_files(
                    [dst_path, src_path],
                    dst_path,
                    compatibility_check=True,
                )
                # Update duration of this destination file
                dst_file_durations[dst_key] = current_dst_duration + src_duration

            videos_idx[key]["episode_duration"] += src_duration

        videos_idx[key]["chunk"] = chunk_idx
        videos_idx[key]["file"] = file_idx

    return videos_idx


def aggregate_data(src_meta, dst_meta, data_idx, data_files_size_in_mb, chunk_size, concatenate_data=True):
    """Aggregates data chunks from a source dataset into the destination dataset.

    Reads source data files, updates indices to match the aggregated dataset,
    and writes them to the destination with proper file rotation.

    Tracks a `src_to_dst` mapping from source (chunk, file) to destination (chunk, file)
    which is critical for correctly updating episode metadata when source datasets
    have multiple data files (e.g., from a previous merge operation).

    Args:
        src_meta: Source dataset metadata.
        dst_meta: Destination dataset metadata.
        data_idx: Dictionary tracking data chunk and file indices.
        data_files_size_in_mb: Maximum size for data files in MB.
        chunk_size: Maximum number of files per chunk.
        concatenate_data: When False, keep one parquet per source file instead of packing into shards.

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
    contains_images = len(dst_meta.image_keys) > 0

    # retrieve features schema for proper image typing in parquet
    hf_features = get_hf_features_from_features(dst_meta.features) if contains_images else None

    # Track source to destination file mapping for metadata update
    # This is critical for handling datasets that are already results of a merge
    src_to_dst: dict[tuple[int, int], tuple[int, int]] = {}

    for src_chunk_idx, src_file_idx in unique_chunk_file_ids:
        src_path = src_meta.root / DEFAULT_DATA_PATH.format(
            chunk_index=src_chunk_idx, file_index=src_file_idx
        )
        if contains_images:
            # Use HuggingFace datasets to read source data to preserve image format
            src_ds = datasets.Dataset.from_parquet(str(src_path))
            df = src_ds.to_pandas()
        else:
            df = pd.read_parquet(src_path)
        df = update_data_df(df, src_meta, dst_meta)

        # Write data and get the actual destination file it was written to
        # This avoids duplicating the rotation logic here
        data_idx, (dst_chunk, dst_file) = append_or_create_parquet_file(
            df,
            src_path,
            data_idx,
            data_files_size_in_mb,
            chunk_size,
            DEFAULT_DATA_PATH,
            contains_images=contains_images,
            aggr_root=dst_meta.root,
            hf_features=hf_features,
            concatenate=concatenate_data,
            one_row_group_per_episode=True,
        )

        # Record the mapping from source to actual destination
        src_to_dst[(src_chunk_idx, src_file_idx)] = (dst_chunk, dst_file)

    # Add the mapping to data_idx for use in metadata update
    data_idx["src_to_dst"] = src_to_dst

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

        meta_idx, _ = append_or_create_parquet_file(
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


_PARQ_WRITERS: dict = {}


def _parq_stream_enabled() -> bool:
    """Read LEROBOT_AGGREGATE_PARQUET_STREAM at call time. Default False."""
    raw = os.environ.get("LEROBOT_AGGREGATE_PARQUET_STREAM", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _parq_reset() -> None:
    """Close any leaked writers from a previous call (defensive)."""
    for entry in list(_PARQ_WRITERS.values()):
        with contextlib.suppress(Exception):
            entry["writer"].close()
    _PARQ_WRITERS.clear()


def _parq_finalize_all() -> None:
    """Close every open writer. Must be called at end of aggregate_datasets()
    so each destination parquet gets a valid footer."""
    for entry in _PARQ_WRITERS.values():
        entry["writer"].close()
    _PARQ_WRITERS.clear()


def _normalize_timestamp_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with every ``videos/*/{from,to}_timestamp`` column as float64.

    LOCAL-FIX(timestamp-dtype): the streaming writer pins the schema of the
    first table it sees and casts later tables to it with ``safe=False``. If
    that first source carried an int64 ``from_timestamp`` (see
    ``update_meta_data``), every later float offset was floored on write.
    Normalising the timestamp columns up front — on the DataFrame, so the
    parquet's pandas metadata agrees with the arrow schema — makes the pinned
    schema float regardless of which source arrives first.
    """
    cols = [
        c
        for c in df.columns
        if c.startswith("videos/")
        and (c.endswith("/from_timestamp") or c.endswith("/to_timestamp"))
        and not pd.api.types.is_float_dtype(df[c])
    ]
    if not cols:
        return df
    df = df.copy()
    for c in cols:
        df[c] = df[c].astype("float64")
    return df


def _append_or_create_parquet_file_streaming(
    df: pd.DataFrame,
    src_path: Path,
    idx: dict[str, int],
    max_mb: float,
    chunk_size: int,
    default_path: str,
    aggr_root: Path,
    concatenate: bool = True,
) -> tuple[dict[str, int], tuple[int, int]]:
    """Streaming variant of append_or_create_parquet_file().

    Maintains an open pyarrow.ParquetWriter per destination path in
    _PARQ_WRITERS. Each call writes one row group (the source's rows)
    and updates a cumulative size estimate. When the estimate would
    exceed max_mb, closes the current writer and opens a new one at
    the rotated (chunk, file) index.

    Returns (updated_idx, (dst_chunk, dst_file)) to match
    append_or_create_parquet_file().
    """
    table = pa.Table.from_pandas(_normalize_timestamp_columns(df), preserve_index=False)
    src_size_mb = get_parquet_file_size_in_mb(src_path)

    dst_path = aggr_root / default_path.format(chunk_index=idx["chunk"], file_index=idx["file"])

    entry = _PARQ_WRITERS.get(dst_path)
    if entry is None:
        # First write to this destination.
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(str(dst_path), table.schema)
        writer.write_table(table)
        _PARQ_WRITERS[dst_path] = {
            "writer": writer,
            "schema": table.schema,
            "size_estimate_mb": src_size_mb,
        }
        return idx, (idx["chunk"], idx["file"])

    if not concatenate or entry["size_estimate_mb"] + src_size_mb >= max_mb:
        # Rotate: close current writer, open a new one at the next index.
        entry["writer"].close()
        del _PARQ_WRITERS[dst_path]
        idx["chunk"], idx["file"] = update_chunk_file_indices(idx["chunk"], idx["file"], chunk_size)
        new_path = aggr_root / default_path.format(chunk_index=idx["chunk"], file_index=idx["file"])
        new_path.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(str(new_path), table.schema)
        writer.write_table(table)
        _PARQ_WRITERS[new_path] = {
            "writer": writer,
            "schema": table.schema,
            "size_estimate_mb": src_size_mb,
        }
        return idx, (idx["chunk"], idx["file"])

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
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError) as exc:
            raise ValueError(
                f"streaming parquet append: schema mismatch for {dst_path} "
                f"and cast to writer schema failed: {exc}"
            ) from exc
    entry["writer"].write_table(table)
    entry["size_estimate_mb"] += src_size_mb
    return idx, (idx["chunk"], idx["file"])


def append_or_create_parquet_file(
    df: pd.DataFrame,
    src_path: Path,
    idx: dict[str, int],
    max_mb: float,
    chunk_size: int,
    default_path: str,
    contains_images: bool = False,
    aggr_root: Path = None,
    hf_features: datasets.Features | None = None,
    concatenate: bool = True,
    one_row_group_per_episode: bool = False,
) -> tuple[dict[str, int], tuple[int, int]]:
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
        hf_features: Optional HuggingFace Features schema for proper image typing.
        concatenate: When False, always rotate to a new file instead of appending to the current one.
        one_row_group_per_episode: True for DATA parquet (emit one row group per episode); False for
            the episodes-metadata parquet (already one row per episode).

    Returns:
        tuple: (updated_idx, (dst_chunk, dst_file)) where updated_idx is the index dict
               and (dst_chunk, dst_file) is the actual destination file the data was written to.
    """
    if _parq_stream_enabled() and not contains_images and not one_row_group_per_episode:
        return _append_or_create_parquet_file_streaming(
            df, src_path, idx, max_mb, chunk_size, default_path, aggr_root, concatenate
        )

    dst_chunk, dst_file = idx["chunk"], idx["file"]
    dst_path = aggr_root / default_path.format(chunk_index=dst_chunk, file_index=dst_file)

    if not dst_path.exists():
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if contains_images:
            to_parquet_with_hf_images(df, dst_path, features=hf_features)
        elif one_row_group_per_episode:
            to_parquet_one_row_group_per_episode(df, dst_path)
        else:
            df.to_parquet(dst_path)
        return idx, (dst_chunk, dst_file)

    src_size = get_parquet_file_size_in_mb(src_path)
    dst_size = get_parquet_file_size_in_mb(dst_path)

    if not concatenate or dst_size + src_size >= max_mb:
        idx["chunk"], idx["file"] = update_chunk_file_indices(idx["chunk"], idx["file"], chunk_size)
        dst_chunk, dst_file = idx["chunk"], idx["file"]
        new_path = aggr_root / default_path.format(chunk_index=dst_chunk, file_index=dst_file)
        new_path.parent.mkdir(parents=True, exist_ok=True)
        final_df = df
        target_path = new_path
    else:
        if contains_images:
            # Use HuggingFace datasets to read existing data to preserve image format
            existing_ds = datasets.Dataset.from_parquet(str(dst_path))
            existing_df = existing_ds.to_pandas()
        else:
            existing_df = pd.read_parquet(dst_path)
        final_df = pd.concat([existing_df, df], ignore_index=True)
        target_path = dst_path

    if contains_images:
        to_parquet_with_hf_images(final_df, target_path, features=hf_features)
    elif one_row_group_per_episode:
        to_parquet_one_row_group_per_episode(final_df, target_path)
    else:
        final_df.to_parquet(target_path)

    return idx, (dst_chunk, dst_file)


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
    aggr_meta.info.total_tasks = len(aggr_meta.tasks)
    aggr_meta.info.total_episodes = sum(m.total_episodes for m in all_metadata)
    aggr_meta.info.total_frames = sum(m.total_frames for m in all_metadata)
    aggr_meta.info.splits = {"train": f"0:{sum(m.total_episodes for m in all_metadata)}"}
    write_info(aggr_meta.info, aggr_meta.root)

    logging.info("write stats")
    aggr_meta.stats = aggregate_stats([m.stats for m in all_metadata])
    write_stats(aggr_meta.stats, aggr_meta.root)
