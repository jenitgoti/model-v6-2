#!/usr/bin/env python3
"""Launch Model_v6.3 training on the WSL box with the RTX PRO 1000.

Loads model_v6_2.py.

    python train_wsl.py selftest     # run this FIRST, takes ~1 minute
    python train_wsl.py prepare      # once, if the letterboxed arrays are absent
    python train_wsl.py transfer     # warm start from the V5 checkpoint
    python train_wsl.py scratch      # random initialization, no V5 weights
    python train_wsl.py report       # print progress, safe while training runs

Paths come from the environment so nothing has to be edited:

    export PCB_DATASET_ROOT="/home/u117134c/Data_preprocessing/45000 images/Split_Data"
    export PCB_MODEL_OUTPUT_DIR="$HOME/Models/Model_v6_2_scratch"
    export PCB_V5_CHECKPOINT="/home/u117134c/Models/Model_v5_instance/best_model_v5_instance.keras"
    export PCB_TRANSFER_OUTPUT_DIR="$HOME/Models/Model_v6_2_from_v5_transfer"

Progress, rewritten every epoch, with per-class precision/recall/F1/AP:

    cat <output dir>/TRAINING_REPORT.md
    ls  <output dir>/reports/          # milestone snapshot every 50 epochs

Resume after any interruption: rerun the same command. BackupAndRestore
continues from the last completed epoch.
"""
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

MODES = ("selftest", "prepare", "transfer", "scratch", "report")
mode = (sys.argv[1] if len(sys.argv) > 1 else "transfer").strip().lower()
if mode not in MODES:
    sys.exit(f"usage: python train_wsl.py [{'|'.join(MODES)}]")

os.environ.setdefault(
    "PCB_DATASET_ROOT",
    "/home/u117134c/Data_preprocessing/45000 images/Split_Data",
)
os.environ.setdefault(
    "PCB_MODEL_OUTPUT_DIR",
    str(Path.home() / "Models" / "Model_v6_2_scratch"),
)

# model_v6_2.py is the only maintained model file. model_v6_3.py is a stale
# copy whose sole differences are path defaults this launcher already sets
# from the environment; it was being preferred here, which would have trained
# the pre-optimisation architecture on the GPU box. Delete it.
MODEL_FILE = HERE / "model_v6_2.py"
if not MODEL_FILE.is_file():
    sys.exit(f"{MODEL_FILE} is not next to this script.")
spec = importlib.util.spec_from_file_location(MODEL_FILE.stem, MODEL_FILE)
m = importlib.util.module_from_spec(spec)
sys.modules[MODEL_FILE.stem] = m
spec.loader.exec_module(m)

# --- RTX PRO 1000, 8 GB ------------------------------------------------------
# These are the file's committed defaults; they are restated here so this
# launcher is self-describing and so the Mac launcher cannot be mistaken for it.
m.REQUIRE_GPU = mode not in ("selftest", "prepare", "report")
m.USE_MIXED_PRECISION = True          # real speedup on CUDA, unlike on CPU
m.BATCH_SIZE = 2
m.GRADIENT_ACCUMULATION_STEPS = 2     # effective batch 4
m.DATA_LOADER_WORKERS = 8             # the GPU must never wait on cv2
m.DATA_LOADER_QUEUE_SIZE = 16

if v5 := os.environ.get("PCB_V5_CHECKPOINT"):
    m.TRANSFER_SOURCE_MODEL = Path(v5)
if out := os.environ.get("PCB_TRANSFER_OUTPUT_DIR"):
    m.TRANSFER_OUTPUT_DIR = Path(out)
    m.TRANSFERRED_MODEL_FOR_INFERENCE = (
        m.TRANSFER_OUTPUT_DIR / "best_transfer_model_v6_2_instance.keras"
    )


def main() -> None:
    m.validate_spatial_configuration()
    m.validate_augmentation_configuration()

    if mode == "selftest":
        m.run_selftest()
        return
    if mode == "prepare":
        m.prepare_dataset()
        m.save_dataset_preview()
        return
    if mode == "report":
        for candidate in (m.TRANSFER_OUTPUT_DIR, m.MODEL_OUTPUT_DIR):
            if (candidate / "training_log.csv").is_file():
                print(m.build_training_report(candidate))
                m.write_training_report(candidate)
                return
        sys.exit("No training_log.csv found in either output directory.")

    transfer = mode == "transfer"
    output_dir = m.TRANSFER_OUTPUT_DIR if transfer else m.MODEL_OUTPUT_DIR

    if transfer:
        # Fail here, not forty minutes into a run, and never silently fall
        # back to random initialization when a warm start was requested.
        if not m.TRANSFER_SOURCE_MODEL.is_file():
            sys.exit(
                f"V5 checkpoint not found: {m.TRANSFER_SOURCE_MODEL}\n"
                "Set PCB_V5_CHECKPOINT to its real path, or run "
                "'python train_wsl.py scratch'."
            )
        epochs = m.TRANSFER_EPOCHS
    else:
        epochs = m.EPOCHS

    print("=" * 72)
    print(f"Model_v6.2 - {'V5 warm start' if transfer else 'scratch'}")
    print("  dataset  :", m.DATASET_ROOT)
    print("  arrays   :", m.ARRAY_DIR)
    print("  output   :", output_dir)
    if transfer:
        print("  V5 source:", m.TRANSFER_SOURCE_MODEL, "(opened read-only)")
        print("  unfreeze : deep@%d  p5@%d  p4@%d  full@%d" % (
            m.TRANSFER_DEEP_UNFREEZE_EPOCH, m.TRANSFER_P5_UNFREEZE_EPOCH,
            m.TRANSFER_P4_UNFREEZE_EPOCH, m.TRANSFER_FULL_FINE_TUNE_EPOCH))
    print("  batch    : %d x %d accumulation (effective %d)" % (
        m.BATCH_SIZE, m.GRADIENT_ACCUMULATION_STEPS,
        m.BATCH_SIZE * m.GRADIENT_ACCUMULATION_STEPS))
    print("  precision:", "mixed_float16" if m.USE_MIXED_PRECISION else "float32")
    print("  epochs   :", epochs)
    print("  report   :", output_dir / "TRAINING_REPORT.md")
    print("=" * 72, flush=True)

    m.train(fine_tune=False, transfer=transfer)


if __name__ == "__main__":
    main()
