#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare raw ScanNet scenes into Open3DIS-compatible layout.

What it can do:
1) Validate raw ScanNet scene files
2) Export RGB / depth / poses / intrinsics from .sens via ScanNet SensReader
3) Normalize frame names to 00000.jpg/png/txt
4) Create intrinsic.txt / intrinsic_depth.txt at scene root
5) Copy original ply into Open3DIS 3D tree
6) Write a custom split file
7) Optionally build groundtruth via ISBNet preprocess_scannet200.py
8) Optionally build superpoints via segmentator

Tested logic target:
- Open3DIS Scannet200-style folder layout
- Python 3 launcher
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ----------------------------
# Logging
# ----------------------------
def setup_logger(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(message)s",
    )


# ----------------------------
# Data model
# ----------------------------
@dataclass
class SceneStatus:
    scene_id: str
    raw_ok: bool = False
    sens_export_ok: bool = False
    rename_ok: bool = False
    ply_copy_ok: bool = False
    gt_ok: bool = False
    superpoint_ok: bool = False
    error: Optional[str] = None


# ----------------------------
# Helpers
# ----------------------------
def run_cmd(cmd: List[str], cwd: Optional[Path] = None, strict: bool = True) -> subprocess.CompletedProcess:
    logging.debug("RUN: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0 and strict:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_copy(src: Path, dst: Path, overwrite: bool = False) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")
    if dst.exists() and not overwrite:
        return
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def discover_scenes(raw_root: Path, requested: Optional[List[str]]) -> List[str]:
    if requested:
        return requested
    scenes = []
    for p in sorted(raw_root.iterdir()):
        if p.is_dir() and p.name.startswith("scene"):
            scenes.append(p.name)
    return scenes


def find_required_raw_files(scene_dir: Path) -> Dict[str, Path]:
    scene = scene_dir.name
    candidates = {
        "sens": scene_dir / f"{scene}.sens",
        "ply": scene_dir / f"{scene}_vh_clean_2.ply",
        "segs": scene_dir / f"{scene}_vh_clean_2.0.010000.segs.json",
        "agg": scene_dir / f"{scene}.aggregation.json",
        "meta": scene_dir / f"{scene}.txt",
    }
    return candidates


def validate_raw_scene(scene_dir: Path) -> Tuple[bool, Dict[str, str]]:
    files = find_required_raw_files(scene_dir)
    missing = {k: str(v) for k, v in files.items() if not v.exists()}
    return len(missing) == 0, missing


def zero_pad_frames(scene_root: Path) -> None:
    for sub, ext in [("color", ".jpg"), ("depth", ".png"), ("pose", ".txt")]:
        d = scene_root / sub
        if not d.exists():
            continue

        files = sorted(d.glob(f"*{ext}"))
        temp_pairs = []

        # 两段重命名，避免目标名冲突
        for p in files:
            stem = p.stem
            if stem.isdigit():
                tmp = d / f"__tmp__{stem}{ext}"
                if tmp.exists():
                    tmp.unlink()
                p.rename(tmp)
                temp_pairs.append(tmp)

        for tmp in sorted(temp_pairs):
            stem = tmp.stem.replace("__tmp__", "")
            final = d / f"{int(stem):05d}{ext}"
            if final.exists():
                final.unlink()
            tmp.rename(final)


def materialize_intrinsics(scene_root: Path) -> None:
    intrinsic_dir = scene_root / "intrinsic"
    if not intrinsic_dir.exists():
        return

    intrinsic_color = intrinsic_dir / "intrinsic_color.txt"
    intrinsic_depth = intrinsic_dir / "intrinsic_depth.txt"

    # Open3DIS loader 直接读 scene 根目录 intrinsic.txt
    if intrinsic_color.exists():
        shutil.copy2(intrinsic_color, scene_root / "intrinsic.txt")
    if intrinsic_depth.exists():
        shutil.copy2(intrinsic_depth, scene_root / "intrinsic_depth.txt")

    # 保险：如果 intrinsic/ 下是逐帧 txt，也保留原目录不动


def export_sens_scene(
    python_exec: str,
    sens_reader: Path,
    sens_file: Path,
    out_scene_dir: Path,
    overwrite: bool = False,
) -> None:
    ensure_dir(out_scene_dir)

    if not overwrite:
        color_dir = out_scene_dir / "color"
        depth_dir = out_scene_dir / "depth"
        pose_dir = out_scene_dir / "pose"
        if color_dir.exists() and depth_dir.exists() and pose_dir.exists():
            logging.info("Skip export (already exists): %s", out_scene_dir.name)
            return

    cmd = [
        python_exec,
        str(sens_reader),
        "--filename", str(sens_file),
        "--output_path", str(out_scene_dir),
        "--export_depth_images",
        "--export_color_images",
        "--export_poses",
        "--export_intrinsics",
    ]
    run_cmd(cmd, strict=True)


def write_split_file(split_file: Path, scenes: List[str]) -> None:
    ensure_dir(split_file.parent)
    split_file.write_text("\n".join(scenes) + "\n", encoding="utf-8")


def build_groundtruth_with_isbnet(
    python_exec: str,
    isbnet_preprocess: Path,
    dataset_root: Path,
    label_map_file: Path,
    scenes: List[str],
    gt_dst_dir: Path,
    overwrite: bool = False,
) -> None:
    if not isbnet_preprocess.exists():
        raise FileNotFoundError(f"ISBNet preprocess script not found: {isbnet_preprocess}")
    if not label_map_file.exists():
        raise FileNotFoundError(f"Label map file not found: {label_map_file}")

    ensure_dir(gt_dst_dir)

    with tempfile.TemporaryDirectory(prefix="scannet_split_") as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / "scannetv2_train.txt").write_text("", encoding="utf-8")
        (tmpdir / "scannetv2_val.txt").write_text("\n".join(scenes) + "\n", encoding="utf-8")

        out_root = tmpdir / "out_gt"
        ensure_dir(out_root)

        cmd = [
            python_exec,
            str(isbnet_preprocess),
            "--dataset_root", str(dataset_root),
            "--output_root", str(out_root),
            "--label_map_file", str(label_map_file),
            "--train_val_splits_path", str(tmpdir),
        ]
        run_cmd(cmd, strict=True)

        for scene in scenes:
            src = out_root / "val" / f"{scene}_inst_nostuff.pth"
            dst = gt_dst_dir / f"{scene}.pth"
            safe_copy(src, dst, overwrite=overwrite)


def build_superpoints(
    scenes: List[str],
    raw_root: Path,
    spp_dst_dir: Path,
    overwrite: bool = False,
) -> None:
    ensure_dir(spp_dst_dir)

    try:
        import numpy as np
        import open3d as o3d
        import torch
        import segmentator
    except Exception as e:
        raise RuntimeError(
            "Failed to import open3d / torch / segmentator for superpoints. "
            "Please make sure these are installed in the open3dis environment."
        ) from e

    for scene in scenes:
        dst = spp_dst_dir / f"{scene}.pth"
        if dst.exists() and not overwrite:
            logging.info("Skip superpoints (already exists): %s", scene)
            continue

        mesh_file = raw_root / scene / f"{scene}_vh_clean_2.ply"
        if not mesh_file.exists():
            raise FileNotFoundError(f"Mesh not found for superpoints: {mesh_file}")

        mesh = o3d.io.read_triangle_mesh(str(mesh_file))
        vertices = torch.from_numpy(np.asarray(mesh.vertices).astype("float32"))
        faces = torch.from_numpy(np.asarray(mesh.triangles).astype("int64"))
        spp = segmentator.segment_mesh(vertices, faces).cpu()
        torch.save(spp, str(dst))


def save_manifest(report_path: Path, scene_statuses: List[SceneStatus], meta: Dict) -> None:
    payload = {
        "meta": meta,
        "scenes": [asdict(s) for s in scene_statuses],
    }
    ensure_dir(report_path.parent)
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# ----------------------------
# Main
# ----------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare raw ScanNet scenes for Open3DIS")
    parser.add_argument("--raw-root", required=True, help="Raw ScanNet scans root, e.g. /workspace/code/data/ScanNet/scans")
    parser.add_argument("--open3dis-root", required=True, help="Open3DIS repo root, e.g. /workspace/code/Open3DIS")
    parser.add_argument("--scenes", nargs="*", default=None, help="Scene ids, e.g. scene0000_00 scene0001_00")
    parser.add_argument("--split-name", default="scannet3_val.txt", help="Output split filename under open3dis/dataset/")
    parser.add_argument("--tag-2d", default="custom", help="2D folder suffix: Scannet200_2D_<tag>/val")
    parser.add_argument("--python-exec", default=sys.executable, help="Python executable used to run helper scripts")
    parser.add_argument("--sens-reader", default="", help="Path to ScanNet SensReader/python/reader.py")
    parser.add_argument("--export-sens", action="store_true", help="Export RGB/depth/pose/intrinsics from .sens")
    parser.add_argument("--copy-ply", action="store_true", help="Copy *_vh_clean_2.ply to original_ply_files/{scene}.ply")
    parser.add_argument("--write-split", action="store_true", help="Write custom split file")
    parser.add_argument("--build-gt", action="store_true", help="Build groundtruth via ISBNet preprocess_scannet200.py")
    parser.add_argument("--isbnet-preprocess", default="", help="Path to preprocess_scannet200.py")
    parser.add_argument("--label-map-file", default="", help="Path to scannetv2-labels.combined.tsv")
    parser.add_argument("--build-superpoints", action="store_true", help="Build superpoints via segmentator")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--strict", action="store_true", help="Fail fast on first scene error")
    parser.add_argument("--verbose", action="store_true", help="Verbose logs")
    args = parser.parse_args()

    setup_logger(args.verbose)

    raw_root = Path(args.raw_root).resolve()
    open3dis_root = Path(args.open3dis_root).resolve()

    if not raw_root.exists():
        logging.error("Raw root not found: %s", raw_root)
        return 2
    if not open3dis_root.exists():
        logging.error("Open3DIS root not found: %s", open3dis_root)
        return 2

    scenes = discover_scenes(raw_root, args.scenes)
    if not scenes:
        logging.error("No scenes found.")
        return 2

    logging.info("Scenes: %s", ", ".join(scenes))

    split_file = open3dis_root / "open3dis" / "dataset" / args.split_name
    data_2d_root = open3dis_root / "data" / "Scannet200" / f"Scannet200_2D_{args.tag_2d}" / "val"
    data_3d_root = open3dis_root / "data" / "Scannet200" / "Scannet200_3D" / "val"
    gt_dir = data_3d_root / "groundtruth"
    ply_dir = data_3d_root / "original_ply_files"
    spp_dir = data_3d_root / "superpoints"

    ensure_dir(data_2d_root)
    ensure_dir(gt_dir)
    ensure_dir(ply_dir)
    ensure_dir(spp_dir)

    scene_statuses: List[SceneStatus] = []

    # 1) validate raw files
    for scene in scenes:
        status = SceneStatus(scene_id=scene)
        scene_statuses.append(status)

        try:
            scene_dir = raw_root / scene
            ok, missing = validate_raw_scene(scene_dir)
            status.raw_ok = ok
            if not ok:
                raise FileNotFoundError(f"Missing raw files: {json.dumps(missing, ensure_ascii=False)}")

            # 2) export sens
            if args.export_sens:
                if not args.sens_reader:
                    raise ValueError("--sens-reader is required when --export-sens is enabled")
                sens_reader = Path(args.sens_reader).resolve()
                sens_file = scene_dir / f"{scene}.sens"
                out_scene_dir = data_2d_root / scene
                export_sens_scene(
                    python_exec=args.python_exec,
                    sens_reader=sens_reader,
                    sens_file=sens_file,
                    out_scene_dir=out_scene_dir,
                    overwrite=args.overwrite,
                )
                materialize_intrinsics(out_scene_dir)
                zero_pad_frames(out_scene_dir)
                status.sens_export_ok = True
                status.rename_ok = True

            # 3) copy ply
            if args.copy_ply:
                src_ply = scene_dir / f"{scene}_vh_clean_2.ply"
                dst_ply = ply_dir / f"{scene}.ply"
                safe_copy(src_ply, dst_ply, overwrite=args.overwrite)
                status.ply_copy_ok = True

        except Exception as e:
            status.error = f"{type(e).__name__}: {e}"
            logging.error("Scene failed: %s -> %s", scene, status.error)
            logging.debug(traceback.format_exc())
            if args.strict:
                break

    # 4) split
    if args.write_split:
        try:
            write_split_file(split_file, scenes)
            logging.info("Wrote split: %s", split_file)
        except Exception as e:
            logging.error("Failed to write split file: %s", e)
            if args.strict:
                return 1

    # 5) groundtruth
    if args.build_gt:
        try:
            if not args.isbnet_preprocess:
                raise ValueError("--isbnet-preprocess is required when --build-gt is enabled")
            if not args.label_map_file:
                raise ValueError("--label-map-file is required when --build-gt is enabled")

            build_groundtruth_with_isbnet(
                python_exec=args.python_exec,
                isbnet_preprocess=Path(args.isbnet_preprocess).resolve(),
                dataset_root=raw_root,
                label_map_file=Path(args.label_map_file).resolve(),
                scenes=scenes,
                gt_dst_dir=gt_dir,
                overwrite=args.overwrite,
            )
            for s in scene_statuses:
                s.gt_ok = True
            logging.info("Groundtruth built: %s", gt_dir)
        except Exception as e:
            logging.error("Failed to build groundtruth: %s", e)
            if args.strict:
                return 1

    # 6) superpoints
    if args.build_superpoints:
        try:
            build_superpoints(
                scenes=scenes,
                raw_root=raw_root,
                spp_dst_dir=spp_dir,
                overwrite=args.overwrite,
            )
            for s in scene_statuses:
                s.superpoint_ok = True
            logging.info("Superpoints built: %s", spp_dir)
        except Exception as e:
            logging.error("Failed to build superpoints: %s", e)
            if args.strict:
                return 1

    report_path = open3dis_root / "data" / "Scannet200" / "prepare_report.json"
    save_manifest(
        report_path=report_path,
        scene_statuses=scene_statuses,
        meta={
            "raw_root": str(raw_root),
            "open3dis_root": str(open3dis_root),
            "split_file": str(split_file),
            "data_2d_root": str(data_2d_root),
            "data_3d_root": str(data_3d_root),
            "scenes": scenes,
        },
    )

    failed = [s for s in scene_statuses if s.error]
    logging.info("Done. report=%s", report_path)
    logging.info("Success scenes: %d / %d", len(scene_statuses) - len(failed), len(scene_statuses))
    if failed:
        logging.warning("Failed scenes: %s", ", ".join(s.scene_id for s in failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())