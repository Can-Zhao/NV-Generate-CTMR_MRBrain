import os, subprocess, shutil
import argparse
import time
from tqdm.auto import tqdm
from monai.apps import download_url
from pathlib import Path
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from typing import List, Dict, Optional

def fetch_to_hf_path_cmd(
    items: List[Dict[str, str]],
    root_dir: str = "./",  # (kept for signature compatibility; not required)
    revision: str = "main",
    overwrite: bool = False,
    token: Optional[str] = None,  # or rely on env HF_TOKEN / HUGGINGFACE_HUB_TOKEN
) -> list[str]:
    """
    items: list of {"repo_id": "...", "filename": "path/in/repo.ext", "path": "local/target.ext"}
    Returns list of saved local paths (in the same order as items).

    Pure Python implementation (CI-safe): no `huggingface-cli` dependency.
    """
    saved = []

    for it in items:
        repo_id = it["repo_id"]
        repo_file = it["filename"]
        dst = Path(it["path"])
        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists() and not overwrite:
            saved.append(str(dst))
            continue

        # Download into HF cache, then copy to requested destination
        # Retry logic for rate limiting (429 errors)
        max_retries = 5
        retry_delay = 60  # Start with 60 seconds
        cached_path = None
        
        for attempt in range(max_retries):
            try:
                cached_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=repo_file,
                    revision=revision,
                    token=token,  # if None, huggingface_hub will use env / cached auth if present
                )
                break  # Success, exit retry loop
            except HfHubHTTPError as e:
                # Check if it's a 429 rate limit error
                is_rate_limit = False
                if hasattr(e, 'response') and e.response is not None:
                    is_rate_limit = e.response.status_code == 429
                elif "429" in str(e) or "Too Many Requests" in str(e) or "rate limit" in str(e).lower():
                    is_rate_limit = True
                
                if is_rate_limit and attempt < max_retries - 1:
                    # Rate limited - wait and retry
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    print(f"\n⚠️  Rate limited (429). Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                    if not token:
                        print(f"💡 Tip: Get a free token at https://huggingface.co/settings/tokens")
                        print(f"   Then set: export HF_TOKEN=your_token_here")
                        print(f"   This will prevent rate limits (public repos work without token, but tokens get higher limits)")
                    time.sleep(wait_time)
                    continue
                else:
                    # Other error or max retries reached
                    raise

        # Copy/move into place
        if dst.exists() and overwrite:
            dst.unlink()

        shutil.copy2(cached_path, dst)
        saved.append(str(dst))

    return saved



def download_model_data(generate_version, root_dir, model_only=False, token=None):
    # TODO: remove the `files` after the files are uploaded to the NGC
    if generate_version == "ddpm-ct" or generate_version == "rflow-ct":
        files = [
            {
                "path": "models/autoencoder_v1.pt",
                "repo_id": "nvidia/NV-Generate-CT",
                "filename":"models/autoencoder_v1.pt",
            },
            {
                "path": "models/mask_generation_autoencoder.pt",
                "repo_id": "nvidia/NV-Generate-CT",
                "filename": "models/mask_generation_autoencoder.pt",
            },
            {
                "path": "models/mask_generation_diffusion_unet.pt",
                "repo_id": "nvidia/NV-Generate-CT",
                "filename": "models/mask_generation_diffusion_unet.pt",
            }]
        if not model_only:
            files += [
                {
                    "path": "datasets/all_anatomy_size_conditions.json",
                    "repo_id": "nvidia/NV-Generate-CT",
                    "filename": "datasets/all_anatomy_size_conditions.json",
                },
                {
                    "path": "datasets/all_masks_flexible_size_and_spacing_4000.zip",
                    "repo_id": "nvidia/NV-Generate-CT",
                    "filename": "datasets/all_masks_flexible_size_and_spacing_4000.zip",
                },
            ]
    elif generate_version == "rflow-mr":
        files = [
            {
                "path": "models/autoencoder_v2.pt",
                "repo_id": "nvidia/NV-Generate-MR",
                "filename": "models/autoencoder_v2.pt",
            },
            {
                "path": "models/diff_unet_3d_rflow-mr.pt",
                "repo_id": "nvidia/NV-Generate-MR",
                "filename": "models/diff_unet_3d_rflow-mr.pt",
            }
        ]
    else:
        raise ValueError(f"generate_version has to be chosen from ['ddpm-ct', 'rflow-ct', 'rflow-mr'], yet got {generate_version}.")
    if generate_version == "ddpm-ct":
        files += [
            {
                "path": "models/diff_unet_3d_ddpm-ct.pt",
                "repo_id": "nvidia/NV-Generate-CT",
                "filename": "models/diff_unet_3d_ddpm-ct.pt",
            },
            {
                "path": "models/controlnet_3d_ddpm-ct.pt",
                "repo_id": "nvidia/NV-Generate-CT",
                "filename": "models/controlnet_3d_ddpm-ct.pt",
            }]
        if not model_only:
            files += [
                {
                    "path": "datasets/candidate_masks_flexible_size_and_spacing_3000.json",
                    "repo_id": "nvidia/NV-Generate-CT",
                    "filename": "datasets/candidate_masks_flexible_size_and_spacing_3000.json",
                },
            ]
    elif generate_version == "rflow-ct":
        files += [
            {
                "path": "models/diff_unet_3d_rflow-ct.pt",
                "repo_id": "nvidia/NV-Generate-CT",
                "filename": "models/diff_unet_3d_rflow-ct.pt",
            },
            {
                "path": "models/controlnet_3d_rflow-ct.pt",
                "repo_id": "nvidia/NV-Generate-CT",
                "filename": "models/controlnet_3d_rflow-ct.pt",
            }]
        if not model_only:
            files += [
                {
                    "path": "datasets/candidate_masks_flexible_size_and_spacing_4000.json",
                    "repo_id": "nvidia/NV-Generate-CT",
                    "filename": "datasets/candidate_masks_flexible_size_and_spacing_4000.json",
                },
            ]
    
    # Get token from environment if not provided
    # Note: Public repos don't require a token, but using one (even free) avoids rate limits
    if token is None:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if token:
            print("Using Hugging Face token from environment variable (helps avoid rate limits).")
        else:
            print("Note: No HF_TOKEN found. Downloads will work but may hit rate limits.")
            print("To avoid rate limits, get a free token at https://huggingface.co/settings/tokens")
            print("Then set: export HF_TOKEN=your_token_here")
    
    for file in files:
        file["path"] = file["path"] if "datasets/" not in file["path"] else os.path.join(root_dir, file["path"])
        if "repo_id" in file.keys():
            path = fetch_to_hf_path_cmd([file], root_dir=root_dir, revision="main", token=token)
            print("saved to:", path)
        else:
            download_url(url=file["url"], filepath=file["path"])
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model downloading")
    parser.add_argument(
        "--version",
        type=str,
        default="rflow-ct",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="./",
    )
    parser.add_argument("--model_only", dest="model_only", action="store_true", help="Download model only, not any dataset")
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token (optional, but helps avoid rate limits). Can also set HF_TOKEN env var. Get free token at https://huggingface.co/settings/tokens"
    )

    args = parser.parse_args()
    download_model_data(args.version, args.root_dir, args.model_only, token=args.token)
