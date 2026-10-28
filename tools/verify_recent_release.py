"""Verify recent archived sources and results without external assets or a GPU."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest=json.loads((ROOT/'provenance/recent_20261028/source_manifest.json').read_text())
    rows=manifest['exports']
    assert len({r['path'] for r in rows})==len(rows)
    for row in rows:
        target=(ROOT/row['path']).resolve()
        assert ROOT in target.parents,row['path']
        assert target.is_file() and target.stat().st_size==row['bytes'],row['path']
        assert digest(target)==row['export_sha256']==row['original_sha256'],row['path']
        if target.suffix=='.py':compile(target.read_text(),str(target),'exec')
        if target.suffix=='.json':json.loads(target.read_text())
    for row in manifest['shared_dependencies']:
        assert digest(ROOT/row['path'])==row['export_sha256'],row['path']
    count=sum(r['kind']=='historical_source' for r in rows)
    assert count==manifest['source_files']
    checkpoints=json.loads((ROOT/'provenance/recent_20261028/checkpoints.json').read_text())
    assert len(checkpoints)==24 and all(len(r['final_sha256'])==64 for r in checkpoints)
    print(f'PASS: {count} unchanged Python sources; {len(rows)-count} results/protocols; 24 checkpoint hashes.')
    print('Shared baseline portability adjustments recorded; no data/weights/GPU execution required.')


if __name__=='__main__':main()
