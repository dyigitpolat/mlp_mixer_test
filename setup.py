import subprocess

# Get the CUDA version string
cuda_version_string = subprocess.run(["nvcc", "--version"], stdout=subprocess.PIPE).stdout.decode("utf-8").split("\n")[3].split(",")[1].split(" ")[2]

# Translate version string to int, i.e., 10.2 -> 102
cuda_version = int(cuda_version_string.split(".")[0]) * 10 + int(cuda_version_string.split(".")[1])

# Create a virtual environment and install torch and torchvision
subprocess.run(["python3", "-m", "venv", "env"]) 
subprocess.run(["env/bin/pip", "install", "torch", "torchvision",  "--index-url" , f"https://download.pytorch.org/whl/cu{cuda_version}"])
subprocess.run(["env/bin/pip", "install", "wandb"])