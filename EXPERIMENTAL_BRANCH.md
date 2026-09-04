# The `experimental` branch

**Who should read this:** anyone who builds or runs the robot-os dataset merge
pipeline (`ml/data_pipeline/lerobot_merge`), or who is about to change which
lerobot commit robot-os points at.

## In one paragraph

This branch changes one thing in LeRobot: the code that combines many small
datasets into one big one (`aggregate_datasets`, what `lerobot-edit-dataset
merge` runs). Our robots record with the `stable-base` branch and do not need
any of this. Our merge pipeline, which turns 16,000+ single-episode datasets
into one 1 TB dataset that we share with Physical Intelligence, cannot run
correctly without it.

## What this branch adds

1. **Video files are assembled directly with ffmpeg, from a plan made up front.**
   The stock code copies and re-stitches video for every episode as it goes.
   For thousands of episodes that takes many hours and a lot of memory. Turned
   on with `LEROBOT_AGGREGATE_VIDEO_PLAN=true`.

2. **The merged tables are written in pieces, not held in memory.**
   The stock code keeps every episode's table in memory until the end. At our
   scale that runs out of memory. Turned on with
   `LEROBOT_AGGREGATE_PARQUET_STREAM=true`.

3. **Per-episode video start and end times are always stored as decimals.**
   Our recorders store the start time of a single-episode dataset as the whole
   number `0`. The stock code kept that integer type when adding offsets, so
   every episode's start time was rounded down to a whole second. The video
   window of almost every merged episode was then wrong by up to one second,
   and the error was only noticed after a dataset had been shared. This branch
   forces the timestamp columns to decimals before any offset is applied.

4. **Progress and timing logs** (`VIZ ...` lines) so a ten-hour merge can be
   watched and its slow parts identified.

## Why it matters

Items 1 and 2 are the difference between a merge that finishes in a few hours
and one that crashes. Item 3 is the difference between a correct dataset and one
where the pictures do not match the robot's joint positions. The robot-os merge
pipeline has a check that rejects a dataset with the item 3 problem, and its
tests exercise items 1 to 3 directly
(`ml/data_pipeline/lerobot_merge/tests/unit/test_fork_aggregate_timestamps.py`,
`tests/test_plan_video_shards.py`, `tests/test_execute_video_plan.py`).

## How to use it

The robot-os repository intentionally points its `thirdparty/lerobot` submodule
at the same commit as the robots (`stable-base`), so that merging a pipeline
branch never changes what runs on a robot. The merge image is built from the
working tree, so **before building the merge image or running a merge locally**:

```bash
git -C thirdparty/lerobot fetch origin
git -C thirdparty/lerobot checkout experimental
```

Do not commit the submodule change this creates. The image build script refuses
to build if the checkout is not this branch.

## Relationship with `stable-base`

The two branches split in May 2026. `stable-base` has since gained recording
features (compressed-frame input, optional rerun, upstream updates) that this
branch does not have; this branch has the merge changes above that
`stable-base` does not have. Bringing them back together is future work. Anyone
doing it should run the merge pipeline's tests and one local end-to-end merge
before changing the pipeline's checkout.

## Where the changes live

Everything is in `src/lerobot/datasets/aggregate.py` (plus small helpers next
to it). The commits on this branch that are not on `stable-base` are listed by:

```bash
git log --oneline experimental ^stable-base
```
