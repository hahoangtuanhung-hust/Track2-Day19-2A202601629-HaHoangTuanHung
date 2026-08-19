import subprocess
import os

def run_cmd(cmd):
    print(f"Running: {cmd}")
    env = os.environ.copy()
    venv_scripts = os.path.abspath(os.path.join(".venv", "Scripts"))
    env["PATH"] = venv_scripts + os.pathsep + env["PATH"]
    
    result = subprocess.run(cmd, shell=True, env=env)
    if result.returncode != 0:
        print(f"FAILED: {cmd}")
        return False
    return True

notebooks = ["notebooks/03_search_api_benchmark.ipynb", "notebooks/04_feast_feature_store.ipynb"]

for nb in notebooks:
    print(f"\nExecuting {nb}...")
    cmd = f"jupyter nbconvert --to notebook --execute --inplace \"{nb}\" --ExecutePreprocessor.timeout=900"
    run_cmd(cmd)

print("\nDone executing notebooks 3 and 4.")
