#!/usr/bin/env python3
"""Generate CompSpoof V2 protocol CSVs.

Functions:
  main() -- CLI: build protocol CSVs for all splits
  process_split(name, meta, root, out, strategy, ...) -- one split's CSV
"""

import argparse
import os
import sys

import pandas as pd

# ── Label mappings ────────────────────────────────────────────────────────────

LABEL_MAPS = {
    "any_fake": {
        "original": "real",
        "bonafide_bonafide": "real",
        "bonafide_spoof": "fake",
        "spoof_bonafide": "fake",
        "spoof_spoof": "fake",
    },
    "speech_only": {
        "original": "real",
        "bonafide_bonafide": "real",
        "bonafide_spoof": "real",  # speech is bona fide
        "spoof_bonafide": "fake",
        "spoof_spoof": "fake",
    },
    "any_real": {
        "original": "real",
        "bonafide_bonafide": "real",
        "bonafide_spoof": "real",  # speech is bona fide
        "spoof_bonafide": "real",  # environment is bona fide
        "spoof_spoof": "fake",
    },
}

# ── Per-split source paths (relative to data_root) ────────────────────────────

SPLITS = {
    "train": {"primary": "development/metadata/train.csv"},
    "valid": {"primary": "development/metadata/val.csv"},
    "eval": {
        "primary": "eval_source/metadata/eval.csv",
        "fallback": "eval_label.csv",
    },
    "test": {
        "primary": "test_source/metadata/test.csv",
        "fallback": "test_label.csv",
    },
}


def process_split(
    split_name,
    meta_csv,
    data_root,
    output_dir,
    label_strategy,
    drop_original_splits=frozenset(),
):
    """Convert one split's metadata CSV into a protocol CSV."""
    df = pd.read_csv(meta_csv, low_memory=False)
    print(f"  Loaded {len(df)} rows  |  columns: {list(df.columns)}")

    # ── Label ─────────────────────────────────────────────────────────────────
    label_col = next((c for c in ("label", "Label") if c in df.columns), None)
    if label_col is None:
        sys.exit(f"ERROR: no label column in {meta_csv}")

    raw_labels = df[label_col].astype(str).str.strip().str.lower()

    # ── Optionally drop the un-mixed "original" class ────────────────────────
    if split_name in drop_original_splits:
        keep = raw_labels != "original"
        dropped = (~keep).sum()
        df = df[keep].copy()
        raw_labels = raw_labels[keep]
        print(
            f"  drop_original: removed {dropped} 'original' rows"
            f" from {split_name}"
        )

    if label_strategy == "multiclass":
        df["Label"] = raw_labels
    else:
        mapping = LABEL_MAPS[label_strategy]
        df["Label"] = raw_labels.map(mapping)
        unmapped = df["Label"].isna().sum()
        if unmapped:
            unknown = set(raw_labels[df["Label"].isna()])
            print(
                f"  WARNING: {unmapped} rows with unknown labels"
                f" {unknown} -> dropped"
            )
            df = df[df["Label"].notna()].copy()

    # ── Path ──────────────────────────────────────────────────────────────────
    path_col = next(
        (
            c
            for c in ("audio_path", "Audio_path", "path", "Path")
            if c in df.columns
        ),
        None,
    )
    if path_col:
        df["Path"] = df[path_col].apply(
            lambda p: _resolve_audio_path(p, data_root)
        )
    else:
        fname_col = next(
            (
                c
                for c in ("filename", "file_name", "file", "wav")
                if c in df.columns
            ),
            None,
        )
        if fname_col:
            df["Path"] = df[fname_col].apply(
                lambda f: _resolve_audio_path(f"audio/{f}", data_root)
            )
            print(f"  No audio_path column; built Path from '{fname_col}'")
        else:
            df["Path"] = ""
            print("  WARNING: no audio path column found; Path will be empty")

    # ── ID ────────────────────────────────────────────────────────────────────
    id_col = next((c for c in ("ID", "id") if c in df.columns), None)
    if id_col:
        df["ID"] = df[id_col].astype(str)
    elif path_col:
        df["ID"] = df[path_col].apply(_stem)
    else:
        df["ID"] = [f"{split_name}_{i}" for i in range(len(df))]

    # ── Optional standard columns ─────────────────────────────────────────────
    rename = {}
    if "duration" in df.columns:
        rename["duration"] = "Duration"
    if "sample_rate" in df.columns:
        rename["sample_rate"] = "SampleRate"
    if "speech_generation_model" in df.columns:
        rename["speech_generation_model"] = "Attack"
    elif "attack" in df.columns:
        rename["attack"] = "Attack"
    if rename:
        df = df.rename(columns=rename)

    # ── Column order: core first, then remaining CompSpoof metadata ───────────
    core = ["ID", "Label", "Path"]
    optional_core = [
        c for c in ("Duration", "SampleRate", "Attack") if c in df.columns
    ]
    used = set(
        core
        + optional_core
        + list(rename.keys())
        + [label_col, path_col or "", id_col or ""]
    )
    extra = [
        c for c in df.columns if c not in used and c not in core + optional_core
    ]
    out_cols = core + optional_core + extra
    df = df[[c for c in out_cols if c in df.columns]]

    # ── Write ─────────────────────────────────────────────────────────────────
    out_path = os.path.join(output_dir, f"{split_name}.csv")
    df.to_csv(out_path, index=False)

    print(f"  → {out_path}  ({len(df)} rows)")
    for lbl, cnt in sorted(df["Label"].value_counts().items()):
        print(f"     {lbl:<25s}  {cnt:>6d}  ({100 * cnt / len(df):.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate JSPS-EXP protocol CSVs from CompSpoof V2"
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help="Root directory of CompSpoof V2 "
        "(the folder containing development/, eval_source/, test_source/, ...)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for train.csv / valid.csv / eval.csv / test.csv",
    )
    parser.add_argument(
        "--label_strategy",
        default="any_fake",
        choices=["any_fake", "speech_only", "any_real", "multiclass"],
        help=(
            "any_fake    : any spoofed component → fake (default)\n"
            "speech_only : label follows speech only; bonafide_spoof → real\n"
            "any_real    : real if either component is bona fide;"
            " spoof_spoof → fake\n"
            "multiclass  : keep original 5-class labels"
        ),
    )
    parser.add_argument(
        "--drop_original",
        action="store_true",
        help="Drop the 'original' class from train and valid splits "
        "(eval/test left intact for comparability).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    drop_original_splits = (
        frozenset(("train", "valid")) if args.drop_original else frozenset()
    )

    print(f"data_root      : {args.data_root}")
    print(f"output_dir     : {args.output_dir}")
    print(f"label_strategy : {args.label_strategy}")
    print(
        f"drop_original  : {args.drop_original}"
        f"  (splits: {sorted(drop_original_splits) or 'none'})"
    )
    print()

    for split_name, split_cfg in SPLITS.items():
        meta_csv = _find_meta_csv(args.data_root, split_cfg)
        if meta_csv is None:
            tried = [
                os.path.join(args.data_root, split_cfg[k])
                for k in ("primary", "fallback")
                if k in split_cfg
            ]
            print(f"SKIP {split_name}: none of {tried} found\n")
            continue
        print(f"── {split_name} ──")
        process_split(
            split_name,
            meta_csv,
            args.data_root,
            args.output_dir,
            args.label_strategy,
            drop_original_splits=drop_original_splits,
        )
        print()

    print("Done.")


def _find_meta_csv(data_root, split_cfg):
    """Return the first existing metadata CSV, else None."""
    for key in ("primary", "fallback"):
        if key not in split_cfg:
            continue
        path = os.path.join(data_root, split_cfg[key])
        if os.path.exists(path):
            return path
    return None


def _resolve_audio_path(raw_path, data_root):
    """Return absolute audio path from root."""
    raw_path = str(raw_path).strip()
    if not raw_path:
        return ""
    if os.path.isabs(raw_path):
        return raw_path
    return os.path.join(data_root, raw_path)


def _stem(path_str):
    return os.path.splitext(os.path.basename(str(path_str)))[0]


if __name__ == "__main__":
    main()
