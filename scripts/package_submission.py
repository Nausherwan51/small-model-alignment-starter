"""Package only reviewed, complete evidence; exclude environments and credentials."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
from pypdf import PdfReader
from prepare_kaggle import source_paths


def main():
    root = Path(__file__).resolve().parents[1]
    bundle = root / "submission/27100384"
    report = root / "output/pdf/27100384_PA2_Report.pdf"
    pages = len(PdfReader(report).pages)
    assert 1 <= pages <= 8
    assert json.loads((bundle / "evaluation/report_analysis.json").read_text())["reviewed"]
    assert json.loads((bundle / "evidence/judging_audit.json").read_text())["primary_results_verified"]
    review=json.loads((bundle / "evidence/report_visual_review.json").read_text())
    assert review["all_pages_inspected"] and review["page_count"] == pages
    assert review["pdf_sha256"] == hashlib.sha256(report.read_bytes()).hexdigest()
    for track in ("uf","hh","persona"):
        summary=json.loads((bundle / f"evaluation/{track}_summary.json").read_text())
        assert summary["counts"]["pending"] == 0
        assert summary["judge_model"] == "google/gemini-3.6-flash"
    for path in (root / "outputs/downloads/main/backup_verified.json",
                 root / "outputs/downloads/uf_recovery/backup_verified.json"):
        assert json.loads(path.read_text())["verified"]
        dest=bundle / "evidence/backups" / path.parent.name
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path,dest/path.name)
    shutil.copy2(report,bundle/report.name)
    prepared=root / "outputs/prepared"
    data=bundle / "evidence/data"
    data.mkdir(exist_ok=True)
    for name in ("manifest.json","alpaca_eval_300_seed42.json","hh_eval_300_seed42.json"):
        shutil.copy2(prepared/name,data/name)
    for name in ("generation_sanity_inspection.json","evaluation_execution.json"):
        shutil.copy2(root / "outputs" / name,bundle / "evidence" / name)
    records={p.relative_to(bundle).as_posix():{"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in bundle.rglob("*") if p.is_file() and p.name != "file_manifest.json"}
    (bundle/"file_manifest.json").write_text(json.dumps(records,indent=2))
    target=root / "submission/27100384_PA2_Submission.zip"
    with zipfile.ZipFile(target,"w",zipfile.ZIP_DEFLATED) as archive:
        for path in bundle.rglob("*"):
            if path.is_file(): archive.write(path,"27100384/"+path.relative_to(bundle).as_posix())
        for path in source_paths(root): archive.write(path,"27100384/source/"+path.relative_to(root).as_posix())
    print(json.dumps({"report_pages":pages,"files":len(records),"zip":str(target),"bytes":target.stat().st_size}))


if __name__ == "__main__": main()
