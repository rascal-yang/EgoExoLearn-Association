"""CPU-only design audit and score-independent diagnostic sampling.

Reads existing TRAIN/VAL metadata, never TEST or GPU/model results. Does not
implement or validate the proposed model, and does not modify the v1 protocol.
"""
import collections
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/egoexo_semantic_processing_20261027"
BASE = ROOT / "experiments/egoexo_textalign_20261025"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def graph_summary(questions, all_candidates=False):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    for row in questions.values():
        query = row["query"]["video_uid"]
        candidates = row["choices"].values() if all_candidates else [row["choices"][str(row["answer"])]]
        for candidate in candidates:
            parent[find(query)] = find(candidate["video_uid"])
    sizes = collections.Counter(find(row["query"]["video_uid"]) for row in questions.values())
    return {"components": len(sizes), "question_sizes": sorted(sizes.values(), reverse=True)}


def main():
    annotation = ROOT / "data/association_annotations/association_val_hard.json"
    metadata_path = ROOT / "repos/EgoExoLearn/video_meta_downsampled.json"
    holdout_path = ROOT / "experiments/egoexo_relations_20261026/selection_holdout.npy"
    protocol_path = ROOT / "experiments/egoexo_semantic_processing_20261027/design_protocol_v2_draft.json"
    plan_path = REPORT / "深入实验方案_v2.txt"
    questions = read(annotation)
    metadata = {row["video_uid"]: row for row in read(metadata_path)}
    records = read(BASE / "records.json")
    phrases = read(BASE / "phrases.json")
    splits = np.load(BASE / "splits.npz")
    train = list(map(int, splits["train"]))
    holdout = list(map(int, np.load(holdout_path)))
    source = collections.Counter(row["query"]["video_uid"] for row in questions.values())
    correct = collections.Counter(row["choices"][str(row["answer"])]["video_uid"] for row in questions.values())
    clips = {(clip["video_uid"], float(clip["start_sec"]), float(clip["end_sec"]))
             for row in questions.values() for clip in [row["query"], *row["choices"].values()]}
    val_videos = {key[0] for key in clips}
    train_intervals = collections.defaultdict(list)
    for i in train:
        video, start, end = records[i]["key"]
        train_intervals[video].append((start, end))
    train_clips = {tuple(records[i]["key"]) for i in train}
    train_videos = set(train_intervals)
    holdout_videos = {records[i]["key"][0] for i in holdout}
    overlap = sum(any(max(a, c) < min(b, d) for c, d in train_intervals.get(video, []))
                  for video, a, b in clips)
    vocab = {kind: {phrases[j]["label"] for i in train for j in records[i]["phrases"]
                    if phrases[j]["type"] == kind} for kind in ("noun", "verb")}
    directions = collections.Counter(metadata[row["query"]["video_uid"]]["view"] for row in questions.values())
    tasks = collections.Counter(metadata[row["query"]["video_uid"]]["task"] for row in questions.values())
    dishes = collections.Counter(row["query"]["dish"] for row in questions.values())
    task_cross = collections.defaultdict(collections.Counter)
    dish_to_task = {"hgr": "task1", "jj": "task2", "dsx": "task3", "mxr": "task4", "xhs": "task5"}
    conflicts = []
    strata = collections.defaultdict(list)
    for uid, row in questions.items():
        meta = metadata[row["query"]["video_uid"]]
        assert len(row["choices"]) == 20
        assert all(metadata[c["video_uid"]]["view"] != meta["view"] for c in row["choices"].values())
        task_cross[row["query"]["dish"]][meta["task"]] += 1
        if dish_to_task[row["query"]["dish"]] != meta["task"]:
            conflicts.append(uid)
        strata[(meta["view"], meta["task"])].append(uid)
    rng = np.random.default_rng(20261027)
    selected, quota_report = [], []
    source_uses = collections.Counter()
    for direction in sorted(directions):
        keys = sorted(key for key in strata if key[0] == direction)
        raw = {key: 128 * len(strata[key]) / directions[direction] for key in keys}
        original_quota = {key: math.floor(raw[key]) for key in keys}
        for key in sorted(keys, key=lambda k: (-(raw[k] - original_quota[k]), k))[:128 - sum(original_quota.values())]:
            original_quota[key] += 1
        capacities = {}
        for key in keys:
            counts = collections.Counter(questions[uid]["query"]["video_uid"] for uid in strata[key])
            capacities[key] = sum(min(4, n) for n in counts.values())
        assert sum(capacities.values()) >= 128
        free, quota, remaining = set(keys), {}, 128
        while free:
            weight = sum(len(strata[key]) for key in free)
            targets = {key: remaining * len(strata[key]) / weight for key in free}
            saturated = {key for key in free if targets[key] > capacities[key]}
            if saturated:
                for key in saturated:
                    quota[key] = capacities[key]
                    remaining -= capacities[key]
                free -= saturated
                continue
            for key in free:
                quota[key] = math.floor(targets[key])
            extras = remaining - sum(quota[key] for key in free)
            ranked = sorted(free, key=lambda k: (-(targets[k] - quota[k]), k))
            for key in ranked[:extras]:
                quota[key] += 1
            break
        assert sum(quota.values()) == 128
        for key in keys:
            available = strata[key]
            capacity = capacities[key]
            assert quota[key] <= capacity, (key, quota[key], capacity)
            picked = []
            for index in rng.permutation(len(available)):
                uid = available[int(index)]
                video = questions[uid]["query"]["video_uid"]
                if source_uses[video] >= 4:
                    continue
                source_uses[video] += 1
                picked.append(uid)
                if len(picked) == quota[key]:
                    break
            assert len(picked) == quota[key]
            selected.extend(picked)
            quota_report.append({"direction": key[0], "task": key[1], "population": len(available),
                                 "original_proportional_quota": original_quota[key],
                                 "quota": quota[key], "capacity_under_video_cap": capacity})
    assert len(selected) == len(set(selected)) == 256
    assert max(source_uses.values()) <= 4
    diagnostic = {
        "status": "score_independent_planning_queue_not_model_evidence",
        "annotation_sha256": digest(annotation),
        "metadata_sha256": digest(metadata_path),
        "seed": 20261027,
        "max_questions_per_source_video": 4,
        "source_videos": len(source_uses),
        "strata": quota_report,
        "uids": selected,
    }
    diagnostic_path = REPORT / "v2_固定256题诊断队列.json"
    diagnostic_path.write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2) + "\n")
    protocol = read(protocol_path)
    assert protocol["training"]["fits"] == len(protocol["groups"]) * len(protocol["training"]["seeds"]) == 18
    assert protocol["training"]["clip_exposures_per_fit"] == len(train) * 4
    assert protocol["training"]["primary_checkpoint"] == "epoch4_final"
    assert protocol["groups"][-1]["represented_patch_mass_per_frame"] == 48
    assert not train_videos & holdout_videos
    report = {
        "scope": "CPU metadata/protocol feasibility audit only; no GPU, no model accuracy, no TEST read",
        "annotation_sha256": digest(annotation),
        "metadata_sha256": digest(metadata_path),
        "text_plan_sha256": digest(plan_path),
        "draft_protocol_sha256": digest(protocol_path),
        "diagnostic_queue_sha256": digest(diagnostic_path),
        "source_files": {str(path.relative_to(ROOT)): digest(path) for path in [
            BASE / "records.json", BASE / "phrases.json", BASE / "splits.npz", holdout_path]},
        "questions": len(questions),
        "directions": dict(directions),
        "metadata_tasks": dict(tasks),
        "annotation_dishes": dict(dishes),
        "dish_to_metadata_task": dict(task_cross),
        "task_field_conflict_count": len(conflicts),
        "task_field_conflict_uids": conflicts,
        "source_video_clusters": len(source),
        "correct_candidate_video_clusters": len(correct),
        "query_correct_shared_physical_videos": len(set(source) & set(correct)),
        "source_cluster_size_min_max": [min(source.values()), max(source.values())],
        "largest_source_clusters": source.most_common(5),
        "largest_source_question_fraction": max(source.values()) / len(questions),
        "cluster_weight_concentration_not_effective_sample_size": len(questions) ** 2 / sum(n * n for n in source.values()),
        "query_correct_graph": graph_summary(questions),
        "all_candidate_graph": graph_summary(questions, True),
        "unique_validation_clips": len(clips),
        "validation_videos": len(val_videos),
        "train_validation_video_overlap": len(train_videos & val_videos),
        "exact_train_validation_clip_overlap": len(train_clips & clips),
        "positive_duration_train_validation_interval_overlap_clips": overlap,
        "interval_overlap_scope": "Current TRAIN records and half-open annotation time intervals, not a decoded-frame overlap audit",
        "train_clips": len(train),
        "train_videos": len(train_videos),
        "selection_holdout_clips": len(holdout),
        "selection_holdout_videos": len(holdout_videos),
        "train_selection_video_overlap": len(train_videos & holdout_videos),
        "train_vocabulary": {kind: len(labels) for kind, labels in vocab.items()},
        "diagnostic_questions": len(selected),
        "diagnostic_source_videos": len(source_uses),
        "diagnostic_max_per_source_video": max(source_uses.values()),
        "illustrative_independent_query_MDE_80pct_pp": {
            str(d): 100 * (2.2414027276 + 0.8416212336) * math.sqrt(d / len(questions))
            for d in [0.02, 0.05, 0.10]},
        "MDE_caveat": "Normal approximation with assumed paired disagreement; not measured power and not cluster-adjusted",
        "attention_score_elements_per_head_per_spatial_block": {
            "dense": 16 * 196 * 197 + 3137, "compact_max": 16 * 196 * 49 + 769},
        "parameter_draft_checks_passed": True,
        "native_model_verified": False,
        "training_launched": False,
        "human_labels_available_from_this_audit": False,
    }
    output = REPORT / "v2_设计核验.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"audit": str(output), "diagnostic_queue": str(diagnostic_path),
                      "diagnostic_source_videos": len(source_uses), "task_conflicts": len(conflicts),
                      "draft_checks_passed": True, "training_launched": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
