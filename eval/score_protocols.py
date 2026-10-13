#!/usr/bin/env python
"""Score per-clip detection for offline analysis.

Functions:
  main() -- score protocols across experiments, write per-clip CSVs
  score_protocol(exp, df) -- clean-score a protocol, add score cols
  merge_with_protocol(df, pred, name, tag) -- join preds onto metadata
  resolve_experiment(arg, base) -- resolve .ckpt, folder, or bare name
  resolve_protocol(arg, base) -- resolve protocol arg to a CSV path
  parse_args() -- CLI: experiments, protocols, out-dir, options
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import torch
from eval_utils import (
    assign_components,
    load_experiment,
    parse_score_vector,
    predict_frame,
)

warnings.filterwarnings("ignore")

# ── path resolution ───────────────────────────────────────────────────────────


def resolve_experiment(arg, log_base):
    """Resolve exp arg: .ckpt, folder, or bare name under log_base."""
    if os.path.isfile(arg) and arg.endswith(".ckpt"):
        return arg
    if os.path.isdir(arg) or not log_base:
        folder = arg
    else:
        folder = os.path.join(log_base, arg)
    if not os.path.isfile(os.path.join(folder, "hyperparams.yaml")):
        raise FileNotFoundError(
            f"no hyperparams.yaml under experiment: {folder}"
        )
    return folder


def resolve_protocol(arg, proto_base):
    """Resolve protocol arg to a CSV path."""
    cands = [arg]
    if proto_base:
        cands.append(os.path.join(proto_base, arg))
    for cand in cands:
        if os.path.isfile(cand):
            return cand
        if os.path.isdir(cand):
            test_csv = os.path.join(cand, "test.csv")
            if os.path.isfile(test_csv):
                return test_csv
    raise FileNotFoundError(f"could not resolve protocol: {arg}")


# ── scoring ──────────────────────────────────────────────────────────────────


def score_protocol(exp, protocol_df):
    """Score protocol_df clean; return metrics and predictions."""
    metrics, pred_df = predict_frame(exp, protocol_df)
    pred_df["Score_real"] = pred_df["score_vector"].apply(
        lambda v: _score_value(v, exp.real_id)
    )
    pred_df["Score_fake"] = pred_df["score_vector"].apply(
        lambda v: _score_value(v, exp.fake_id)
    )
    return metrics, pred_df


def merge_with_protocol(protocol_df, pred_df, exp_name, protocol_tag):
    """Left-join protocol metadata onto predictions, keyed on ID."""
    proto = protocol_df.copy()
    proto["_id"] = proto["ID"].astype(str)
    pred = pred_df.copy()
    pred["_id"] = pred["sample_id"].astype(str)
    merged = proto.merge(pred.drop(columns=["sample_id"]), on="_id", how="left")
    merged = merged.drop(columns=["_id"])
    merged.insert(0, "experiment", exp_name)
    merged.insert(1, "protocol", protocol_tag)
    return merged


# ── main ─────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description="Dump per-clip evaluation score CSVs for offline analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "-e",
        "--exp",
        nargs="+",
        required=True,
        help="Experiment folders, model.ckpt files, or bare names "
        "resolved under --log-base.",
    )
    p.add_argument(
        "-p",
        "--protocol",
        nargs="+",
        required=True,
        help="Protocol CSV paths, directories (-> <dir>/test.csv), "
        "or bare names resolved under --proto-base.",
    )
    p.add_argument(
        "-o", "--out-dir", required=True, help="Directory for the score CSVs."
    )
    p.add_argument(
        "--log-base",
        default=None,
        help="Base dir for resolving bare experiment names "
        "(optional; not needed when --exp are full paths).",
    )
    p.add_argument(
        "--proto-base",
        default=None,
        help="Base dir for resolving bare protocol names "
        "(optional; not needed when --protocol are full paths).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Per-model inference batch size.",
    )
    p.add_argument(
        "--checkpoint",
        default="latest",
        help="Which checkpoint to score: latest (default), best_valid "
        "(lowest validation loss), or an int step number. Ignored for "
        ".ckpt files.",
    )
    p.add_argument(
        "--data-root",
        default=None,
        help="Audio root for relative or $ROOT protocol paths "
        "(default: the data_root stored in the trained config).",
    )
    p.add_argument(
        "--device", default=None, help="cuda / cpu (default: auto-detect)."
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Score only the first N rows of each protocol (quick smoke test).",
    )
    p.add_argument(
        "--components",
        action="store_true",
        help="Add CompSpoof speech/event/Class columns when the protocol "
        "carries the *_generation_mothed source columns.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    exp_folders = [resolve_experiment(e, args.log_base) for e in args.exp]
    protocols = [resolve_protocol(pr, args.proto_base) for pr in args.protocol]
    print(f"device: {device}")
    print(f"experiments ({len(exp_folders)}):")
    for f in exp_folders:
        print(f"  - {f}")
    print(f"protocols ({len(protocols)}):")
    for f in protocols:
        print(f"  - {f}")

    # Load each protocol once (and optionally add component columns) up front.
    proto_frames = {}
    for proto_path in protocols:
        df = pd.read_csv(proto_path, low_memory=False)
        if "Label" in df.columns:
            df["Label"] = df["Label"].astype(str).str.strip().str.lower()
        if args.components and {
            "speech_generation_mothed",
            "env_generation_mothed",
        } <= set(df.columns):
            df = assign_components(df)
        if args.limit is not None:
            df = df.head(args.limit).reset_index(drop=True)
        proto_frames[proto_path] = df
    proto_tags = _protocol_tags(protocols)

    # Loop models on the outer level so each checkpoint loads exactly once.
    for folder in exp_folders:
        print(f"\n── loading {folder}  (checkpoint={args.checkpoint})")
        exp = load_experiment(
            folder,
            device,
            batch_size=args.batch_size,
            checkpoint=args.checkpoint,
            data_root=args.data_root,
        )
        exp_name = exp.name
        try:
            for proto_path, proto_tag in zip(protocols, proto_tags):
                df = proto_frames[proto_path]
                print(f"   scoring {len(df)} clips on {proto_tag} …")
                metrics, pred_df = score_protocol(exp, df)
                merged = merge_with_protocol(df, pred_df, exp_name, proto_tag)

                n_scored = merged["score_vector"].notna().sum()
                out_path = os.path.join(
                    args.out_dir, f"{exp_name}__{proto_tag}.csv"
                )
                merged.to_csv(out_path, index=False)
                print(f"     scored {n_scored}/{len(merged)} -> {out_path}")
                print(f"     metrics: {metrics}")
        finally:
            exp.unload()


# ── private helpers ───────────────────────────────────────────────────────────


def _score_value(vec, idx):
    """Extract one score from a score_vector."""
    vec = parse_score_vector(vec)
    return np.nan if vec is None else float(vec[idx])


def _protocol_tags(proto_paths):
    """Parent-folder tag per protocol; add file stem when shared."""
    parents = [os.path.basename(os.path.dirname(p)) for p in proto_paths]
    tags = []
    for path, parent in zip(proto_paths, parents):
        stem = os.path.splitext(os.path.basename(path))[0]
        if not parent:
            tags.append(stem)
        elif parents.count(parent) > 1:
            # e.g. AnyFakeNoOriginal/{test,eval}.csv must not overwrite.
            tags.append(f"{parent}_{stem}")
        else:
            tags.append(parent)
    return tags


if __name__ == "__main__":
    main()
