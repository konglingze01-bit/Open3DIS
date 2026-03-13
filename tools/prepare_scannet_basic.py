#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def setup_logger(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(levelname)s] %(message)s")


def run_cmd(cmd, cwd=None, strict=True):
    logging.debug("RUN: %s", " ".join(map(str, cmd)))
    result = subprocess.run(
        list(map(str, cmd)),
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0 and strict:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(map(str, cmd))}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def validate_scene(
    raw_scene_dir: Path,
    need_sens: bool = False,
    need_ply: bool = False,
    need_gt: bool = False,
) -> dict:
    scene = raw_scene_dir.name
    required = {}

    if need_sens:
        required["sens"] = raw_scene_dir / f"{scene}.sens"

    if need_ply:
        required["ply"] = raw_scene_dir / f"{scene}_vh_clean_2.ply"

    # 只有在生成 groundtruth 时，才检查这些 3D 标注文件
    if need_gt:
        required["ply"] = raw_scene_dir / f"{scene}_vh_clean_2.ply"
        required["segs"] = raw_scene_dir / f"{scene}_vh_clean_2.0.010000.segs.json"
        required["agg"] = raw_scene_dir / f"{scene}.aggregation.json"
        required["meta"] = raw_scene_dir / f"{scene}.txt"

    return {k: str(v) for k, v in required.items() if not v.exists()}


def zero_pad_dir(d: Path, ext: str) -> None:
    if not d.exists():
        return
    files = sorted(d.glob(f"*{ext}"))
    tmp_files = []
    for p in files:
        if p.stem.isdigit():
            tmp = d / f"__tmp__{p.stem}{ext}"
            if tmp.exists():
                tmp.unlink()
            p.rename(tmp)
            tmp_files.append(tmp)
    for p in sorted(tmp_files):
        stem = p.stem.replace("__tmp__", "")
        p.rename(d / f"{int(stem):05d}{ext}")


def normalize_export(scene_dir: Path) -> None:
    zero_pad_dir(scene_dir / "color", ".jpg")
    zero_pad_dir(scene_dir / "depth", ".png")
    zero_pad_dir(scene_dir / "pose", ".txt")

    intrinsic_dir = scene_dir / "intrinsic"
    intrinsic_color = intrinsic_dir / "intrinsic_color.txt"
    intrinsic_depth = intrinsic_dir / "intrinsic_depth.txt"

    if intrinsic_color.exists():
        shutil.copy2(intrinsic_color, scene_dir / "intrinsic.txt")
    if intrinsic_depth.exists():
        shutil.copy2(intrinsic_depth, scene_dir / "intrinsic_depth.txt")


def export_sens(python_exec: str, sens_reader: Path, sens_file: Path, out_scene_dir: Path, overwrite: bool):
    if not sens_reader.exists():
        raise FileNotFoundError(f"SensReader not found: {sens_reader}")
    ensure_dir(out_scene_dir)

    if not overwrite:
        if (out_scene_dir / "color").exists() and (out_scene_dir / "depth").exists() and (out_scene_dir / "pose").exists():
            logging.info("Skip existing export: %s", out_scene_dir.name)
            return

    cmd = [
        python_exec, sens_reader,
        "--filename", sens_file,
        "--output_path", out_scene_dir,
        "--export_depth_images",
        "--export_color_images",
        "--export_poses",
        "--export_intrinsics",
    ]
    run_cmd(cmd, strict=True)
    normalize_export(out_scene_dir)


def copy_original_ply(raw_root: Path, scene: str, ply_dst_root: Path, overwrite: bool):
    src = raw_root / scene / f"{scene}_vh_clean_2.ply"
    dst = ply_dst_root / f"{scene}.ply"
    if dst.exists() and not overwrite:
        return
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def write_split(split_file: Path, scenes):
    ensure_dir(split_file.parent)
    split_file.write_text("\n".join(scenes) + "\n", encoding="utf-8")


def build_groundtruth(
    python_exec: str,
    preprocess_script: Path,
    dataset_root: Path,
    label_map_file: Path,
    scenes,
    gt_dst_root: Path,
    overwrite: bool,
):
    if not preprocess_script.exists():
        raise FileNotFoundError(f"preprocess_scannet200.py not found: {preprocess_script}")
    if not label_map_file.exists():
        raise FileNotFoundError(f"label_map_file not found: {label_map_file}")

    ensure_dir(gt_dst_root)

    with tempfile.TemporaryDirectory(prefix="scannet3_split_") as tmp:
        tmp = Path(tmp)
        (tmp / "scannetv2_train.txt").write_text("", encoding="utf-8")
        (tmp / "scannetv2_val.txt").write_text("\n".join(scenes) + "\n", encoding="utf-8")

        out_root = tmp / "out_gt"
        ensure_dir(out_root)

        cmd = [
            python_exec,
            preprocess_script,
            "--dataset_root", dataset_root,
            "--output_root", out_root,
            "--label_map_file", label_map_file,
            "--train_val_splits_path", tmp,
        ]
        run_cmd(cmd, strict=True)

        for scene in scenes:
            src = out_root / "val" / f"{scene}_inst_nostuff.pth"
            dst = gt_dst_root / f"{scene}.pth"
            if dst.exists() and not overwrite:
                continue
            shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(description="Prepare basic ScanNet data for Open3DIS")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--open3dis-root", required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--sens-reader", required=True)
    parser.add_argument("--python-exec", default=sys.executable)
    parser.add_argument("--export-sens", action="store_true")
    parser.add_argument("--copy-ply", action="store_true")
    parser.add_argument("--write-split", action="store_true")
    parser.add_argument("--build-gt", action="store_true")
    parser.add_argument("--isbnet-preprocess", default="")
    parser.add_argument("--label-map-file", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logger(args.verbose)

    raw_root = Path(args.raw_root).resolve()
    repo = Path(args.open3dis_root).resolve()
    sens_reader = Path(args.sens_reader).resolve()

    data_2d_root = repo / "data" / "Scannet200" / "Scannet200_2D_custom" / "val"
    data_3d_root = repo / "data" / "Scannet200" / "Scannet200_3D" / "val"
    ply_root = data_3d_root / "original_ply_files"
    gt_root = data_3d_root / "groundtruth"
    split_file = repo / "open3dis" / "dataset" / "scannet3_val.txt"

    ensure_dir(data_2d_root)
    ensure_dir(ply_root)
    ensure_dir(gt_root)

    report = {"scenes": {}, "paths": {
        "data_2d_root": str(data_2d_root),
        "ply_root": str(ply_root),
        "gt_root": str(gt_root),
        "split_file": str(split_file),
    }}

    for scene in args.scenes:
        report["scenes"][scene] = {"ok": True, "steps": []}
        raw_scene_dir = raw_root / scene

        try:
            missing = validate_scene(
                raw_scene_dir,
                need_sens=args.export_sens,
                need_ply=args.copy_ply or args.build_gt,
                need_gt=args.build_gt,
            )
            if missing:
                raise FileNotFoundError(json.dumps(missing, indent=2, ensure_ascii=False))

            if args.export_sens:
                export_sens(
                    python_exec=args.python_exec,
                    sens_reader=sens_reader,
                    sens_file=raw_scene_dir / f"{scene}.sens",
                    out_scene_dir=data_2d_root / scene,
                    overwrite=args.overwrite,
                )
                report["scenes"][scene]["steps"].append("export_sens")

            if args.copy_ply:
                copy_original_ply(raw_root, scene, ply_root, args.overwrite)
                report["scenes"][scene]["steps"].append("copy_ply")

        except Exception as e:
            report["scenes"][scene]["ok"] = False
            report["scenes"][scene]["error"] = f"{type(e).__name__}: {e}"
            logging.error("Scene failed: %s -> %s", scene, report["scenes"][scene]["error"])

    if args.write_split:
        write_split(split_file, args.scenes)
        logging.info("Wrote split: %s", split_file)

    if args.build_gt:
        build_groundtruth(
            python_exec=args.python_exec,
            preprocess_script=Path(args.isbnet_preprocess).resolve(),
            dataset_root=raw_root,
            label_map_file=Path(args.label_map_file).resolve(),
            scenes=args.scenes,
            gt_dst_root=gt_root,
            overwrite=args.overwrite,
        )
        logging.info("Built groundtruth: %s", gt_root)

    report_path = repo / "data" / "Scannet200" / "prepare_basic_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("Report written: %s", report_path)


if __name__ == "__main__":
    main()