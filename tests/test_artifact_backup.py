"""Range-download recovery and safetensors integrity checks without network."""

import importlib.util
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("artifact_backup", Path(__file__).resolve().parents[1] / "scripts/backup_training.py")
backup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backup)


def test_range_download_retries_from_last_written_byte(tmp_path, monkeypatch):
    header = json.dumps({"x": {"dtype": "F32", "shape": [3], "data_offsets": [0, 12]}}).encode()
    payload = struct.pack("<Q", len(header)) + header + bytes(12)
    offsets = []
    class Response:
        status_code = 206
        def __init__(self, offset):
            self.offset = offset
            self.headers = {"Content-Range": f"bytes {offset}-{len(payload)-1}/{len(payload)}"}
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def raise_for_status(self):
            pass
        def iter_content(self, chunk_size):
            if len(offsets) == 1:
                yield payload[self.offset:self.offset + 5]
                raise backup.requests.ConnectionError("simulated interruption")
            yield payload[self.offset:]
    def get(url, *, headers, **kwargs):
        offset = int(headers["Range"].split("=")[1].split("-")[0])
        offsets.append(offset)
        return Response(offset)
    monkeypatch.setattr(backup.requests, "get", get)
    monkeypatch.setattr(backup.time, "sleep", lambda seconds: None)
    path = backup.download(SimpleNamespace(file_name="pa2/outputs/sft/best/model.safetensors", url="private"), tmp_path)
    assert offsets == [0, 5]
    assert path.read_bytes() == payload
    assert not path.with_suffix(".safetensors.part").exists()
    backup.verify_weights(path)
    path.write_bytes(payload[:-1])
    with pytest.raises(ValueError, match="Truncated"):
        backup.verify_weights(path)


def test_artifact_path_escape_is_rejected_before_request(tmp_path):
    with pytest.raises(ValueError, match="Unsafe"):
        backup.download(SimpleNamespace(file_name="../outside", url="private"), tmp_path)


def test_finished_restart_download_recovers_after_crash_before_rename(tmp_path, monkeypatch):
    import zipfile
    target = tmp_path / "latest.pt"
    part = target.with_suffix(".pt.part")
    with zipfile.ZipFile(part, "w") as archive:
        archive.writestr("archive/data.pkl", b"private restart structure; never unpickled")
    class Response:
        status_code = 416
        headers = {"Content-Range": f"bytes */{part.stat().st_size}"}
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    monkeypatch.setattr(backup.requests, "get", lambda *a, **kw: Response())
    recovered = backup.download(SimpleNamespace(file_name="latest.pt", url="private"), tmp_path)
    backup.verify_file(recovered, ".pt")
    assert recovered == target and not part.exists()
