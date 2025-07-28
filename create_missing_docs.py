import os
import re

with open("mkdocs.yml", "r") as f:
    for line in f:
        match = re.match(r"^\s*[-]*\s*(.+):\s*(.*\.md)", line)
        if match:
            title, path = match.groups()
            full_path = os.path.join("docs", path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            if not os.path.exists(full_path):
                with open(full_path, "w") as md:
                    md.write(f"# {title.strip()}\n")
                print(f"✅ Created: {full_path}")
            else:
                print(f"✔️ Already exists: {full_path}")