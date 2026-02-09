"""
ComfyUI-AceMusic installation script.
Automatically installs required dependencies when loaded by ComfyUI Manager.
"""

import subprocess
import sys
import importlib


def install_package(package_name, pip_name=None):
    """Install a package if not already available."""
    if pip_name is None:
        pip_name = package_name
    try:
        importlib.import_module(package_name)
    except ImportError:
        print(f"[ComfyUI-AceMusic] Installing {pip_name}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pip_name, "--quiet"]
        )


# Core dependencies
install_package("soundfile", "soundfile")
install_package("scipy", "scipy")

# Check for ACE-Step (optional but required for actual generation)
try:
    import acestep  # noqa: F401
    print("[ComfyUI-AceMusic] ACE-Step is installed.")
except ImportError:
    print(
        "[ComfyUI-AceMusic] WARNING: ACE-Step is not installed.\n"
        "  Music generation will not work until ACE-Step is installed.\n"
        "  Install with: pip install git+https://github.com/ace-step/ACE-Step.git\n"
        "  If that fails, see: https://github.com/hiroki-abe-58/ComfyUI-AceMusic#troubleshooting"
    )
