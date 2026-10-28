"""Export an explicit code/result allowlist from the research workspace.

Never copies datasets, weights, feature arrays, logs or experiment directories.
Existing portable baseline modules are preserved. This records an archive;
historical protocols still require their original external assets to execute.
"""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_DIRS = (
    'egoexo_relations', 'egoexo_tokenmerge', 'egoexo_semantic_processing',
    'egoexo_block10', 'egoexo_layer_compressibility',
)
RESULTS = {
    'reports/egoexo_relations_20261026/seed17_results.json': 'relations_seed17.json',
    'reports/egoexo_relations_20261026/seed17_结论与机制核查.txt': 'relations_conclusion.txt',
    'reports/egoexo_tokenmerge_20261026/results.json': 'tokenmerge.json',
    'reports/egoexo_tokenmerge_20261026/本轮结论.txt': 'tokenmerge_conclusion.txt',
    'reports/egoexo_semantic_processing_20261027/v2_VAL结果.json': 'semantic_val.json',
    'reports/egoexo_semantic_processing_20261027/v2_TEST确认结果.json': 'semantic_test.json',
    'reports/egoexo_semantic_processing_20261027/v2最终实验报告.txt': 'semantic_report.txt',
    'reports/egoexo_semantic_processing_20261027/退化原因诊断.json': 'degradation.json',
    'reports/egoexo_semantic_processing_20261027/退化原因_结论.txt': 'degradation_conclusion.txt',
    'reports/egoexo_block10_20261028/结果汇总.json': 'block10_val.json',
    'reports/egoexo_block10_20261028/结果汇总.txt': 'block10_report.txt',
    'reports/egoexo_layer_compressibility_20261028/逐层96槽结果.json': 'layer96.json',
    'reports/egoexo_layer_compressibility_20261028/逐层96槽结果.csv': 'layer96.csv',
    'reports/egoexo_layer_compressibility_20261028/逐层96槽结果.txt': 'layer96_report.txt',
    'reports/egoexo_layer_compressibility_20261028/K_V压缩理论收益上限.json': 'kv_mac_budget.json',
    'experiments/egoexo_layer_compressibility_20261028/benchmark_gpu3.json': 'layer96_benchmark_raw.json',
}
PROTOCOLS = {
    'experiments/egoexo_semantic_processing_20261027/protocol_v2_frozen.json': 'semantic_v2_frozen.json',
    'experiments/egoexo_semantic_processing_20261027/runtime_amendment.json': 'semantic_runtime_amendment.json',
    'experiments/egoexo_semantic_processing_20261027/post_training/degradation_protocol.json': 'degradation_protocol.json',
    'experiments/egoexo_block10_20261028/protocol_v2_frozen.json': 'block10_frozen.json',
    'experiments/egoexo_block10_20261028/runtime_choices.json': 'block10_runtime_choices.json',
    'experiments/egoexo_layer_compressibility_20261028/protocol_frozen.json': 'layer96_frozen.json',
    'experiments/egoexo_layer_compressibility_20261028/preflight_gpu2_b32.json': 'layer96_preflight.json',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    assert source != ROOT, 'Source and export must be separate directories'
    mappings = []
    for directory in CODE_DIRS:
        paths = sorted((source / 'scripts' / directory).glob('*.py'))
        assert paths, directory
        for path in paths:
            relative = path.relative_to(source)
            mappings.append((relative, relative, 'historical_source'))
    for relative, name in RESULTS.items():
        mappings.append((Path(relative), Path('results/recent_20261028') / name, 'aggregate_result'))
    for relative, name in PROTOCOLS.items():
        mappings.append((Path(relative), Path('provenance/recent_20261028/protocols') / name, 'historical_protocol'))
    # Validate the complete allowlist before copying any files.
    for relative, destination, kind in mappings:
        assert (source / relative).is_file(), relative
        assert (source / relative).stat().st_size < 2 * 2**20, relative
        assert destination.suffix in ('.py', '.json', '.csv', '.txt'), destination
    manifest = []
    for relative, destination, kind in mappings:
        src, dst = source / relative, ROOT / destination
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        assert digest(src) == digest(dst)
        manifest.append(dict(path=str(destination), source_path=str(relative),
            original_sha256=digest(src), export_sha256=digest(dst), bytes=dst.stat().st_size, kind=kind))
    # Record shared imports without overwriting prior portability adjustments.
    shared = []
    for relative in (
        'scripts/egoexo_textalign/common.py', 'scripts/egoexo_textalign/backend.py',
        'scripts/egoexo_textalign/model.py', 'scripts/time_origin_reeval/run.py',
        'scripts/sampling_corrected/bounded_decode.py',
        'repos/EgoExoLearn/association_benchmark/models/timesformer.py',
    ):
        shared.append(dict(path=relative, original_sha256=digest(source/relative),
            export_sha256=digest(ROOT/relative),
            identical=digest(source/relative)==digest(ROOT/relative)))
    write(ROOT/'provenance/recent_20261028/source_manifest.json', dict(
        exported_at=datetime.now(timezone.utc).isoformat(),
        purpose='Historical source archive with aggregate results; not a fresh training rerun',
        source_files=sum(r['kind']=='historical_source' for r in manifest),
        exports=manifest, shared_dependencies=shared,
        historical_frozen_files_modified=False,
        external_assets_included=False))
    checkpoints = []
    for study, arms in [('egoexo_semantic_processing_20261027', ('R0','R1','R2','R3','R4','R5')),
                        ('egoexo_block10_20261028', ('S1','S2'))]:
        for arm in arms:
            for seed in (17,29,43):
                folder=source/'experiments'/study/'fits'/arm/f'seed{seed}'
                done=json.loads((folder/'complete.json').read_text())
                assert done['passed']
                checkpoints.append(dict(study=study,arm=arm,seed=seed,epoch=4,
                    final_sha256=done['final_sha256'],included=False))
    write(ROOT/'provenance/recent_20261028/checkpoints.json',checkpoints)
    print(json.dumps(dict(copied=len(manifest),source_files=sum(r['kind']=='historical_source' for r in manifest),
        total_bytes=sum(r['bytes'] for r in manifest),checkpoint_hashes=len(checkpoints)),indent=2))


if __name__ == '__main__':
    main()
