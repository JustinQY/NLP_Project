# setup_env.py
import subprocess

def install(package):
    print(f"Installing {package}...")
    subprocess.run(
        ["pip", "install", package],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"✅ {package} installed.")

def main():
    install("datasets")
    install("bitsandbytes -U")

if __name__ == "__main__":
    main()