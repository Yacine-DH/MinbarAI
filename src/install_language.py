"""
Verify that all required dependencies for MinbarAI are installed correctly.
"""
import sys

def check_dependencies():
    """Check if all required packages are installed."""
    required_packages = {
        'faster_whisper': 'faster-whisper',
        'translate': 'translate',
        'numpy': 'numpy',
        'requests': 'requests',
    }
    
    missing_packages = []
    
    print("Checking MinbarAI dependencies...")
    print("-" * 40)
    
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✓ {package_name}")
        except ImportError:
            print(f"✗ {package_name} (NOT INSTALLED)")
            missing_packages.append(package_name)
    
    print("-" * 40)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print(f"\nInstall them with:")
        print(f"  pip install {' '.join(missing_packages)}")
        return False
    else:
        print("\n✓ All dependencies installed!")
        print("\nYou can now run:")
        print("  python src/transcribe.py")
        print("  python src/main.py")
        return True

if __name__ == "__main__":
    success = check_dependencies()
    sys.exit(0 if success else 1)