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

# Sync
run_cmd("jupytext --to notebook --update notebooks/08_feature_engineering.py")

# Execute
nb = "notebooks/08_feature_engineering.ipynb"
print(f"\nExecuting {nb}...")
cmd = f"jupyter nbconvert --to notebook --execute --inplace \"{nb}\" --ExecutePreprocessor.timeout=900"
run_cmd(cmd)

print("\nDone executing notebook 08.")
