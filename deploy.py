# deploy.py
import subprocess

commands = [
    "source venv/bin/activate",
    "mkdocs build --clean",
    "ghp-import -p -f site",  # <- no -n here!
    "deactivate"
]

script = " && ".join(commands)
subprocess.call(script, shell=True, executable="/bin/bash")