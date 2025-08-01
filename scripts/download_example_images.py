import os
import urllib.request

# GitHub Repo Base URL for a repository containing example images from ImageNet
REPO_BASE = "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master"

# Some example images selected from the repo 
# to use for testing purposes (WordNet‑ID, class-label, filename)
SELECTED = [
    ("n01443537", "goldfish", "goldfish.JPEG"),
    ("n02085936", "Maltese_dog", "dog.JPEG"),     
    ("n02123394", "Persian_cat", "cat.JPEG"),
    ("n02493793", "spider_monkey", "monkey.JPEG"),
]

# Downloads selected example images from the repository
# and saves them to the specified directory
def download_selected_images(save_dir="examples/original_images"):
    os.makedirs(save_dir, exist_ok=True)
    for synset, label, fname in SELECTED:
        url = f"{REPO_BASE}/{synset}_{label}.JPEG"
        out_path = os.path.join(save_dir, fname)
        if os.path.exists(out_path):
            print(f"✅ {fname} already exists.")
        else:
            print(f"⬇️ Downloading {fname} for class '{label}'...")
            try:
                urllib.request.urlretrieve(url, out_path)
                print(f"✅ Saved to {out_path}")
            except Exception as e:
                print(f"❌ Failed to download {label}: {e}")

if __name__ == "__main__":
    download_selected_images()
