import pandas as pd
import pyarrow.parquet as pq
import pytest

from lerobot.datasets import aggregate as agg


@pytest.fixture
def stream_env(monkeypatch):
    monkeypatch.setenv("LEROBOT_AGGREGATE_PARQUET_STREAM", "true")
    agg._parq_reset()
    yield
    agg._parq_reset()


def _src_parquet(tmp_path, name, df):
    path = tmp_path / f"{name}.parquet"
    df.to_parquet(path)
    return path


def _append(df, src, idx, tmp_path, max_mb=1000.0):
    return agg.append_or_create_parquet_file(
        df,
        src,
        idx,
        max_mb,
        1000,
        "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        aggr_root=tmp_path / "out",
    )


def test_streaming_append_reorders_columns_to_writer_schema(tmp_path, stream_env):
    first = pd.DataFrame({"observation.state": [1.0], "action": [10.0], "index": [0]})
    second = pd.DataFrame({"action": [20.0], "observation.state": [2.0], "index": [1]})
    idx = {"chunk": 0, "file": 0}
    _append(first, _src_parquet(tmp_path, "a", first), idx, tmp_path)
    _append(second, _src_parquet(tmp_path, "b", second), idx, tmp_path)
    agg._parq_finalize_all()

    out = pq.read_table(tmp_path / "out/data/chunk-000/file-000.parquet").to_pandas()
    assert list(out.columns) == ["observation.state", "action", "index"]
    assert out["action"].tolist() == [10.0, 20.0]
    assert out["observation.state"].tolist() == [1.0, 2.0]


def test_streaming_rotation_keeps_first_column_order(tmp_path, stream_env):
    first = pd.DataFrame({"observation.state": [1.0], "action": [10.0], "index": [0]})
    second = pd.DataFrame({"action": [20.0], "observation.state": [2.0], "index": [1]})
    idx = {"chunk": 0, "file": 0}
    _append(first, _src_parquet(tmp_path, "a", first), idx, tmp_path, max_mb=0.0)
    _append(second, _src_parquet(tmp_path, "b", second), idx, tmp_path, max_mb=0.0)
    agg._parq_finalize_all()

    assert idx == {"chunk": 0, "file": 1}
    out = pq.read_table(tmp_path / "out/data/chunk-000/file-001.parquet").to_pandas()
    assert list(out.columns) == ["observation.state", "action", "index"]
    assert out["action"].tolist() == [20.0]


def test_streaming_append_rejects_different_column_sets(tmp_path, stream_env):
    first = pd.DataFrame({"observation.state": [1.0], "index": [0]})
    second = pd.DataFrame({"action": [20.0], "index": [1]})
    idx = {"chunk": 0, "file": 0}
    _append(first, _src_parquet(tmp_path, "a", first), idx, tmp_path)
    with pytest.raises(ValueError, match="schema mismatch"):
        _append(second, _src_parquet(tmp_path, "b", second), idx, tmp_path)
