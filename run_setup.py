import os
import subprocess
import sys

def run_cmd(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error ({result.returncode}):\n{result.stderr}")
    else:
        print(f"Success:\n{result.stdout}")

print("Setting up virtual environment...")
run_cmd(f"{sys.executable} -m venv .venv")

venv_python = os.path.join(".venv", "Scripts", "python.exe")

print("Upgrading pip...")
run_cmd(f"{venv_python} -m pip install -q -U pip")

print("Installing requirements...")
run_cmd(f"{venv_python} -m pip install -q -r requirements.txt")

print("Upgrading dill for Python 3.14+...")
run_cmd(f"{venv_python} -m pip install -q --upgrade \"dill>=0.4,<1.0\"")

print("Copying .env...")
if not os.path.exists(".env"):
    import shutil
    shutil.copy(".env.example", ".env")
    print(".env copied.")
else:
    print(".env already exists.")

print("Running seed_corpus...")
run_cmd(f"{venv_python} scripts/seed_corpus.py")

print("Running gen_agent_queries...")
run_cmd(f"{venv_python} scripts/gen_agent_queries.py")

print("Running gen_spend...")
run_cmd(f"{venv_python} scripts/gen_spend.py")

print("Running verify_lite...")
run_cmd(f"{venv_python} scripts/verify_lite.py")

print("Setup completed.")
