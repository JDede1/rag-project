"""
sync_colab_url.py
------------------------------------
Syncs the latest Ngrok URL from your Google Drive (rag_llm_url.txt)
to your local generator.py file used by the RAG backend.

Usage:
    python src/utils/sync_colab_url.py
"""

import os
from pathlib import Path
import re

# ====================================================
# 1️⃣ Locate Drive file (saved from Colab)
# ====================================================
drive_path_options = [
    Path.home() / "Google Drive" / "MyDrive" / "rag_llm_url.txt",  # Mac/Windows
    Path.home() / "MyDrive" / "rag_llm_url.txt",                   # Linux
    Path("/mnt/drive/MyDrive/rag_llm_url.txt"),                    # Colab mount path (optional)
]

url_file = None
for path in drive_path_options:
    if path.exists():
        url_file = path
        break

if not url_file:
    print("❌ Could not find 'rag_llm_url.txt' in Google Drive.")
    print("Make sure your Colab notebook saved it there.")
    exit(1)

# ====================================================
# 2️⃣ Read the Ngrok URL
# ====================================================
with open(url_file) as f:
    new_url = f.read().strip()

if not new_url.startswith("http"):
    print(f"❌ Invalid URL in {url_file}: {new_url}")
    exit(1)

print(f"🔗 Found latest Ngrok URL:\n{new_url}\n")

# ====================================================
# 3️⃣ Locate and update generator.py
# ====================================================
project_root = Path(__file__).resolve().parents[2]
generator_path = project_root / "src" / "generation" / "generator.py"

if not generator_path.exists():
    print(f"❌ generator.py not found at {generator_path}")
    exit(1)

with open(generator_path, "r") as f:
    content = f.read()

# Replace any existing REMOTE_LLM_URL line
new_content, count = re.subn(
    r'REMOTE_LLM_URL\s*=\s*["\']https?://[^"\']+["\']',
    f'REMOTE_LLM_URL = "{new_url}"',
    content,
)

if count == 0:
    print("⚠️ No REMOTE_LLM_URL found in generator.py — adding new one.")
    new_content = f'REMOTE_LLM_URL = "{new_url}"\n\n' + content

with open(generator_path, "w") as f:
    f.write(new_content)

print(f"✅ Updated generator.py with the new endpoint:\n{new_url}")
print("🎯 You can now run `make run-api` to start your backend.")
