import glob
import subprocess
import os
import sys

def run_cmd(cmd):
    print(f"Running: {cmd}")
    # Update environment to include the venv Scripts directory in PATH
    env = os.environ.copy()
    venv_scripts = os.path.abspath(os.path.join(".venv", "Scripts"))
    env["PATH"] = venv_scripts + os.pathsep + env["PATH"]
    
    result = subprocess.run(cmd, shell=True, env=env)
    if result.returncode != 0:
        print(f"FAILED: {cmd}")
        return False
    return True

# Convert py to ipynb
print("Converting notebooks...")
run_cmd("jupytext --to notebook --update notebooks/0*.py")

# Execute ipynb
notebooks = glob.glob("notebooks/0*.ipynb")
notebooks.sort()

success = True
for nb in notebooks:
    print(f"\nExecuting {nb}...")
    cmd = f"jupyter nbconvert --to notebook --execute --inplace \"{nb}\" --ExecutePreprocessor.timeout=900"
    if not run_cmd(cmd):
        success = False

if success:
    print("\nALL NOTEBOOKS EXECUTED SUCCESSFULLY")
else:
    print("\nSOME NOTEBOOKS FAILED")
    sys.exit(1)
