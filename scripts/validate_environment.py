
import argparse
import importlib
import inspect
import platform
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "src/fusion/adaptive_gate.py",
    "src/fusion/fusion_model.py",
    "src/fusion/fusion_losses.py",
    "scripts/train_fusion.py",
    "scripts/test_fusion.py",
    "src/preprocessing/flir_dataset.py",
    "src/preprocessing/pair_flir_images.py",
    "src/preprocessing/coco_converter.py",
    "scripts/build_fused_dataset.py",
    "src/detection/train_detector.py",
    "src/detection/inference.py",
    "scripts/run_video_inference.py",
    "src/evaluation/metrics.py",
    "src/evaluation/visualization.py",
]

REQUIRED_PACKAGES = [
    "torch",
    "torchvision",
    "cv2",
    "PIL",
    "numpy",
    "pandas",
    "matplotlib",
    "rfdetr",
]


def check_project_files():
    print("\n1. PROJECT FILE CHECK")
    missing = []

    for relative_path in REQUIRED_FILES:
        path = PROJECT_ROOT / relative_path

        if path.exists():
            print(f"[OK] {relative_path}")
        else:
            print(f"[MISSING] {relative_path}")
            missing.append(relative_path)

    print(f"\nFiles found: {len(REQUIRED_FILES) - len(missing)}"
          f"/{len(REQUIRED_FILES)}")

    return missing


def check_python_environment():
    print("\n2. PYTHON ENVIRONMENT")
    print("Python version:", platform.python_version())
    print("Platform:", platform.platform())

    missing_packages = []

    for package in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "version unavailable")
            print(f"[OK] {package}: {version}")
        except Exception as error:
            print(f"[ERROR] {package}: {error}")
            missing_packages.append(package)

    return missing_packages


def check_gpu():
    print("\n3. GPU CHECK")

    try:
        import torch

        if torch.cuda.is_available():
            print("[OK] CUDA is available")
            print("GPU:", torch.cuda.get_device_name(0))
            print("CUDA version:", torch.version.cuda)
        else:
            print("[INFO] CUDA is not available.")
            print("The script can still run CPU-side checks.")

    except Exception as error:
        print("[ERROR] Could not check GPU:", error)


def check_rfdetr_api():
    print("\n4. RF-DETR API CHECK")

    try:
        from rfdetr import RFDETRLarge

        print("[OK] RFDETRLarge imported")

        try:
            print("Constructor signature:")
            print(inspect.signature(RFDETRLarge))
        except (TypeError, ValueError):
            print("[INFO] Constructor signature unavailable")

        for method_name in ["train", "predict", "optimize_for_inference"]:
            method = getattr(RFDETRLarge, method_name, None)

            if method is None:
                print(f"[NOT FOUND] {method_name}")
                continue

            try:
                print(f"{method_name}{inspect.signature(method)}")
            except (TypeError, ValueError):
                print(f"[FOUND] {method_name}, signature unavailable")

    except Exception as error:
        print("[ERROR] RF-DETR API could not be inspected:", error)


def check_optional_paths(dataset_root, checkpoint, video_root):
    print("\n5. OPTIONAL DATA PATH CHECK")

    paths = {
        "Dataset root": dataset_root,
        "Fusion checkpoint": checkpoint,
        "Video root": video_root,
    }

    for label, value in paths.items():
        if not value:
            print(f"[SKIPPED] {label}: not provided")
            continue

        path = Path(value).expanduser()

        if path.exists():
            print(f"[OK] {label}: {path}")
        else:
            print(f"[NOT FOUND] {label}: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate the RGB-T fusion and RF-DETR project."
    )

    parser.add_argument("--dataset", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--videos", default=None)

    args = parser.parse_args()

    print("=" * 65)
    print("ADAPTIVE RGB-THERMAL FUSION + RF-DETR PROJECT VALIDATION")
    print("=" * 65)

    missing_files = check_project_files()
    missing_packages = check_python_environment()

    check_gpu()
    check_rfdetr_api()

    check_optional_paths(
        args.dataset,
        args.checkpoint,
        args.videos
    )

    print("\n6. VALIDATION SUMMARY")
    print("Missing project files:", len(missing_files))
    print("Unavailable packages:", len(missing_packages))

    if missing_files or missing_packages:
        print("\n[WARNING] Resolve the issues listed above before training.")
    else:
        print("\n[OK] Basic project and dependency checks passed.")

    print("\nIMPORTANT:")
    print("Passing these checks does not prove that training or inference")
    print("will work. Dataset compatibility and model APIs still need")
    print("runtime testing.")


if __name__ == "__main__":
    main()
