"""
This script downloads the artifacts required to reproduce the results.
"""

from huggingface_hub import snapshot_download


def main():
    repo_id = "cmpatino/utility_driven_prediction"
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        ignore_patterns=["*.md", "*.gitattributes"],
        local_dir="./",
    )


if __name__ == "__main__":
    main()
