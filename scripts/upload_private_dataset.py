"""PA2 private dataset uploader: no public switch; verify server visibility."""
import argparse
import json
import os
from pathlib import Path


def private_metadata(api, identity, destination):
    path = api.dataset_metadata(identity, str(destination))
    response = json.loads(Path(path).read_text(encoding="utf-8"))
    info = response.get("info", response)
    if info.get("isPrivate") is not True:
        raise ValueError("Dataset is not confirmed private; do not launch a job")
    return {"dataset": identity, "private": True}


def upload(api, folder, mode, notes, receipt_folder):
    folder = Path(folder).resolve()
    metadata = json.loads((folder / "dataset-metadata.json").read_text())
    identity = metadata["id"]
    if not identity.startswith("nausherwan123/"):
        raise ValueError("Unexpected PA2 dataset owner")
    if mode == "version":
        private_metadata(api, identity, receipt_folder)
    previous = Path.cwd()
    try:
        os.chdir(folder)  # Kaggle Windows upload cache requires a simple path.
        if mode == "create":
            response = api.dataset_create_new(".", public=False, quiet=True, convert_to_csv=False)
        else:
            response = api.dataset_create_version(".", notes, quiet=True, convert_to_csv=False, delete_old_versions=False)
        if getattr(response, "error", None):
            raise RuntimeError("Kaggle rejected the private dataset upload")
    finally:
        os.chdir(previous)
    result = private_metadata(api, identity, receipt_folder)
    (receipt_folder / "private_upload_verified.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--mode", choices=["create", "version"], required=True)
    parser.add_argument("--notes", default="Private reproducible PA2 inputs")
    args = parser.parse_args()
    folder = Path(args.path).resolve()
    receipts = folder.parent / "privacy_receipt"
    receipts.mkdir(parents=True, exist_ok=True)
    api = KaggleApi(); api.authenticate()
    print(json.dumps(upload(api, folder, args.mode, args.notes, receipts)))


if __name__ == "__main__": main()
