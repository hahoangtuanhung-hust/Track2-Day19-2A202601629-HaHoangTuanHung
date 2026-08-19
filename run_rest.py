import subprocess
import os
import glob

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

# Get notebooks 05 to 08
notebooks = sorted(glob.glob("notebooks/0[5-8]*.ipynb"))

for nb in notebooks:
    print(f"\nExecuting {nb}...")
    cmd = f"jupyter nbconvert --to notebook --execute --inplace \"{nb}\" --ExecutePreprocessor.timeout=900"
    if not run_cmd(cmd):
        print("Stopping due to error.")
        break

print("\nDone executing remaining notebooks.")
