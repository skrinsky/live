from huggingface_hub import snapshot_download

HF_TOKEN = "YOUR_HF_TOKEN_HERE"

print("Downloading EARS vocals...")
snapshot_download(
    repo_id="facebook/ears",
    repo_type="dataset",
    local_dir="data/clean_vocals",
    token=HF_TOKEN,
)

print("Downloading DNS noise...")
snapshot_download(
    repo_id="microsoft/DNS-Challenge",
    repo_type="dataset",
    local_dir="data/noise",
    token=HF_TOKEN,
)

print("Done.")
