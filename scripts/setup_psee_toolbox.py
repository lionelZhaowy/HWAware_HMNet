"""Install HMNet's pinned upstream GEN1 tools into the ignored dependency folder."""

from pathlib import Path
import shutil
import subprocess
import tempfile

URL = "https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox.git"
COMMIT = "c09d34a4fb8dbfd2db7081bf5e26078c2aa94fc7"
DESTINATION = Path(__file__).resolve().parents[1] / "hmnet/utils/psee_toolbox"


def main():
    marker = DESTINATION / "UPSTREAM_COMMIT"
    if DESTINATION.exists():
        if marker.exists() and marker.read_text().strip() == COMMIT:
            print(f"Already installed: {DESTINATION} ({COMMIT})")
            return
        raise SystemExit(f"Existing unrecognized dependency; left untouched: {DESTINATION}")
    with tempfile.TemporaryDirectory(prefix="hmnet-psee-") as temporary:
        checkout = Path(temporary) / "checkout"
        subprocess.run(["git", "clone", "--no-checkout", URL, str(checkout)], check=True)
        subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", COMMIT], check=True)
        shutil.copytree(checkout / "src", DESTINATION)
        shutil.copy2(checkout / "LICENSE", DESTINATION / "LICENSE")
        marker.write_text(COMMIT + "\n")
    print(f"Installed: {DESTINATION} ({COMMIT})")


if __name__ == "__main__":
    main()
