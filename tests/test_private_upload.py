import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts.upload_private_dataset import upload


def test_new_dataset_explicitly_private_and_verified(tmp_path):
    folder=tmp_path/"input"; folder.mkdir()
    receipt=tmp_path/"receipt"; receipt.mkdir()
    (folder/"dataset-metadata.json").write_text(json.dumps({"id":"nausherwan123/test-input"}))
    class Api:
        calls=[]
        def dataset_create_new(self, path, **kwargs):
            self.calls.append(kwargs)
            assert Path.cwd()==folder and path=="."
            return SimpleNamespace(error=None)
        def dataset_metadata(self, identity, destination):
            path=Path(destination)/"metadata.json"
            path.write_text(json.dumps({"info":{"isPrivate":True}}))
            return path
    api=Api(); before=Path.cwd()
    assert upload(api,folder,"create","notes",receipt)["private"]
    assert api.calls[0]["public"] is False and Path.cwd()==before


def test_version_refuses_public_dataset_before_upload(tmp_path):
    folder=tmp_path/"input"; folder.mkdir()
    (folder/"dataset-metadata.json").write_text(json.dumps({"id":"nausherwan123/test-input"}))
    class Api:
        def dataset_metadata(self, identity, destination):
            path=tmp_path/"metadata.json"
            path.write_text(json.dumps({"info":{"isPrivate":False}}))
            return path
        def dataset_create_version(self,*args,**kwargs):
            pytest.fail("Public dataset must not be updated")
    with pytest.raises(ValueError,match="not confirmed private"):
        upload(Api(),folder,"version","notes",tmp_path)
