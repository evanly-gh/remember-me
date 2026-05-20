"""
Push a trained beauty checkpoint to the Hugging Face Hub.

Optional helper — only needed if you want the face-service to pull
weights from the Hub instead of bundling them into the Docker image.

After running, set `BEAUTY_HF_REPO_ID=<your-repo-id>` in the
face-service environment. `BeautyAnalyzer` will resolve the checkpoint
via `huggingface_hub.hf_hub_download` on first load.
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="path to beauty_regressor.pt")
    parser.add_argument("--hf-repo-id", required=True,
                        help="e.g. your-username/scut-fbp5500-resnet50")
    parser.add_argument("--remote-filename", default="beauty_regressor.pt",
                        help="filename inside the HF repo")
    parser.add_argument("--private", action="store_true",
                        help="create a private repo (default: public)")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    api = HfApi()
    create_repo(
        repo_id=args.hf_repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    print(f"Uploading {ckpt} → {args.hf_repo_id}/{args.remote_filename}")
    api.upload_file(
        path_or_fileobj=str(ckpt),
        path_in_repo=args.remote_filename,
        repo_id=args.hf_repo_id,
        repo_type="model",
    )

    # Also push a small README so the model card isn't empty.
    readme_text = (
        "# SCUT-FBP5500 Beauty Regressor\n\n"
        "Single-output regression head on top of a timm ResNet-50 "
        "backbone, fine-tuned on the SCUT-FBP5500 dataset.\n\n"
        "Output: float in [1.0, 5.0], where higher = more conventionally "
        "attractive per the dataset's averaged human ratings.\n\n"
        "Inference: see "
        "https://github.com/HCIILAB/SCUT-FBP5500-Database-Release for "
        "the dataset license and per-image distribution restrictions.\n"
    )
    api.upload_file(
        path_or_fileobj=readme_text.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.hf_repo_id,
        repo_type="model",
    )
    print("Done. Set BEAUTY_HF_REPO_ID=" + args.hf_repo_id + " in the face-service env.")


if __name__ == "__main__":
    main()
