import getpass
from huggingface_hub import HfApi

print("Starting Hugging Face deployment...")

token = getpass.getpass("Enter HF Write Token: ")
api = HfApi()

try:
    api.upload_folder(
        folder_path=".", 
        repo_id="saransh619/Tomato-Disease-AI",
        repo_type="space",
        token=token,
        ignore_patterns=["venv/*", ".git/*", "__pycache__/*", "deploy.py"]
    )
    print("Deployment successful. Space is now building.")
except Exception as e:
    print(f"Deployment failed: {e}")
