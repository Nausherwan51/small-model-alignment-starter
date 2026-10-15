"""Stream and verify private Kaggle training artifacts without exposing URLs.

Use the session-output API: the committed-files API can be empty after timeout.
Downloads use bounded memory, resumable HTTP ranges, retries, and atomic rename.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import struct
import time
import zipfile

import requests


def verify_weights(path):
    with path.open("rb") as handle:
        header_length = struct.unpack("<Q", handle.read(8))[0]
        if not 0 < header_length < 100_000_000:
            raise ValueError("Invalid safetensors header")
        tensors = json.loads(handle.read(header_length))
    offsets = [entry["data_offsets"][1] for key, entry in tensors.items() if key != "__metadata__"]
    if not offsets or path.stat().st_size != 8 + header_length + max(offsets):
        raise ValueError("Truncated safetensors artifact")


def verify_file(path, suffix):
    if suffix == ".safetensors":
        verify_weights(path)
    elif suffix == ".pt":
        with zipfile.ZipFile(path) as archive:
            if not any(name.endswith("/data.pkl") for name in archive.namelist()):
                raise ValueError("Incomplete PyTorch restart archive")


def download(item, root):
    name, url = item.file_name, item.url
    target = (root / name).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("Unsafe artifact path")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size:
        try:
            if target.suffix in (".safetensors", ".pt"):
                verify_file(target, target.suffix)
            else:
                # Small config/metrics/tokenizer artifacts are downloaded afresh.
                raise ValueError("refresh")
            return target
        except (ValueError, OSError, struct.error, zipfile.BadZipFile):
            pass
    part = target.with_suffix(target.suffix + ".part")
    attempt = 0
    while True:
        offset = part.stat().st_size if part.exists() else 0
        try:
            # Bounded ranges also avoid proxies buffering an entire 2 GB object
            # before yielding its first byte. Each completed chunk is reusable.
            headers = {"Range": f"bytes={offset}-{offset + 64 * 1024 * 1024 - 1}"}
            with requests.get(url, headers=headers, stream=True, timeout=(30, 60)) as response:
                if response.status_code == 416 and response.headers.get("Content-Range") == f"bytes */{offset}":
                    # Previous process finished the body but died before rename.
                    verify_file(part, target.suffix)
                    part.replace(target)
                    return target
                response.raise_for_status()
                append = response.status_code == 206
                if append:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if not match or int(match[1]) != offset:
                        raise ValueError("Invalid range response")
                    expected_end, total = int(match[2]) + 1, int(match[3])
                else:
                    total = int(response.headers["Content-Length"])
                    expected_end = total
                with part.open("ab" if append else "wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        handle.write(chunk)
                if part.stat().st_size != expected_end:
                    raise ValueError("Incomplete range response")
            attempt = 0
            if part.stat().st_size < total:
                continue
            verify_file(part, target.suffix)
            part.replace(target)
            return target
        except (requests.RequestException, OSError, ValueError, struct.error, zipfile.BadZipFile):
            if attempt == 3:
                # Do not include exception text: HTTP exceptions can contain signed URLs.
                raise RuntimeError(f"Artifact download failed after retries: {name}") from None
            time.sleep(2 ** attempt)
            attempt += 1


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", default="nausherwan123/pa2-sft-dpo-20261015")
    parser.add_argument("--output", default="outputs/backups/main")
    parser.add_argument("--required-tracks", nargs="*", default=["sft", "dpo_hh"],
                        help="Tracks whose complete/best/final artifacts must exist; empty permits paused runs")
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    owner, slug, _ = api.parse_kernel_string(args.kernel)
    items, token, log = [], None, None
    with api.build_kaggle_client() as client:
        while True:
            request = ApiListKernelSessionOutputRequest()
            request.user_name, request.kernel_slug = owner, slug
            api._set_paging(request, 200, token)
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            for item in response.files or []:
                name = item.file_name.replace("\\", "/")
                if any(part.endswith((".tmp", ".part", ".pending")) for part in Path(name).parts):
                    continue  # interrupted/uncommitted writes are not restart candidates
                if "/outputs/" in name and any(f"/outputs/{track}/" in name for track in ("sft", "dpo_hh", "dpo_uf")):
                    items.append(item)
                elif name.endswith(("/provenance.json", "/failure.txt", "/pipeline_complete.json")):
                    items.append(item)
            log = log or response.log
            token = response.next_page_token
            if not token:
                break
    if log:
        (root / "training.log").write_text(log, encoding="utf-8")
    if not items:
        raise RuntimeError("Session output not available yet; no backup claimed")
    # Three streams bound memory and allow small metadata to proceed alongside weights.
    items.sort(key=lambda item: item.file_name.endswith(".safetensors"))
    with ThreadPoolExecutor(max_workers=3) as pool:
        paths = list(pool.map(lambda item: download(item, root), items))
    records = {}
    for path in paths:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        records[path.relative_to(root).as_posix()] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    report = {"kernel": args.kernel, "files": records, "verified": True}
    for track in args.required_tracks:
        base = root / "pa2/outputs" / track
        for suffix in ("complete.json", "best/model.safetensors", "final/model.safetensors"):
            if not (base / suffix).exists():
                raise RuntimeError(f"Required backup missing: {track}/{suffix}")
    (root / "backup_verified.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"verified": True, "files": len(records), "bytes": sum(r["bytes"] for r in records.values()),
                      "manifest": str(root / "backup_verified.json")}))


if __name__ == "__main__":
    main()
