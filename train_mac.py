"""Launch scratch training of model_v6_2 on this Mac.

Keeps model_v6_2.py's committed configuration pointing at the RTX box and
overrides only what this machine requires. Measured on an M3 Air, 16 GB:

    CPU  batch 1 : 3.03 s/image   <- best
    CPU  batch 2 : 3.28 s/image
    CPU  batch 4 : 4.83 s/image   (unified-memory pressure)
    Metal GPU    : 7.54 s/image   (tensorflow-metal is slower than the CPU
                                   for this model; do not enable it)

Data loading is 0.012 s/image, 0.4% of a step, so the model is the whole
cost and only less compute makes this faster.

Resume: just run it again. BackupAndRestore picks up from the last epoch.
Progress:  cat runs/scratch_mac/TRAINING_REPORT.md
"""
import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("PCB_DATASET_ROOT", str(HERE / "Split_Data"))
os.environ.setdefault("PCB_MODEL_OUTPUT_DIR", str(HERE / "runs" / "scratch_mac"))

spec = importlib.util.spec_from_file_location("model_v6_2", HERE / "model_v6_2.py")
m = importlib.util.module_from_spec(spec)
sys.modules["model_v6_2"] = m
spec.loader.exec_module(m)

# --- this machine ------------------------------------------------------------
m.REQUIRE_GPU = False
# float16 has no fast path on CPU and costs casts on every layer.
m.USE_MIXED_PRECISION = False
# Effective batch stays 4, matching the RTX config's 2 x 2, but at the
# per-image cost of batch 1 rather than batch 4.
m.BATCH_SIZE = 1
m.GRADIENT_ACCUMULATION_STEPS = 4
# The model saturates all 8 cores; loader threads would only compete with it.
m.DATA_LOADER_WORKERS = 2
m.DATA_LOADER_QUEUE_SIZE = 8

# The cosine schedule anneals over EPOCHS, so stopping a 300-epoch run early
# leaves the learning rate high and the model un-annealed. At the measured
# ~3.9 h/epoch, 300 epochs is ~50 days; 120 completes the whole schedule in
# ~20 days and still covers the 95-mode augmentation cycle with margin.
# Set this back to 300 if the machine can be left alone for two months.
m.EPOCHS = 120

if __name__ == "__main__":
    m.validate_spatial_configuration()
    m.validate_augmentation_configuration()
    print("=" * 70)
    print("Model_v6.2 scratch training")
    print("  dataset :", m.DATASET_ROOT)
    print("  output  :", m.MODEL_OUTPUT_DIR)
    print("  batch   :", m.BATCH_SIZE, "x", m.GRADIENT_ACCUMULATION_STEPS,
          "accumulation steps")
    print("  epochs  :", m.EPOCHS, f"(~3.9 h/epoch measured on this machine)")
    print("  report  :", m.MODEL_OUTPUT_DIR / "TRAINING_REPORT.md")
    print("=" * 70, flush=True)
    m.train(fine_tune=False, transfer=False)
