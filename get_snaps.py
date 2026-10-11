import subprocess
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

API_KEY = "3406e87c3f92f5c6dd80eff1c501a0a5"
BASE_DIR = Path(r"D:\Abhilash_Thesis_Data\Simulation_Data")

SIMULATIONS = ["TNG100-1"]
SNAPS = [92, 93, 94, 95]

MAX_WORKERS = 8

BASE_URL = "http://www.tng-project.org/api"

def download_snapshot(sim, snap):
    try:
        print(f"Starting {sim} | Snapshot {snap}")

        outdir = BASE_DIR / sim / f"groups_{snap:03d}"
        outdir.mkdir(parents=True, exist_ok=True)

        url = f"{BASE_URL}/{sim}/files/groupcat-{snap}/?format=api"

        wget_cmd = [
            "wget",
            "-nd",
            "-nc",
            "-nv",
            "-e", "robots=off",
            "-l", "1",
            "-r",
            "-A", "hdf5",
            "--content-disposition",
            f"--header=API-Key: {API_KEY}",
            url
        ]

        subprocess.run(wget_cmd, cwd=outdir, check=True)

        # Download checksums
        checksum_url = f"{BASE_URL}/{sim}/checksums/groupcat-{snap}/"
        checksum_cmd = [
            "wget",
            "-O", "checksums.txt",
            "-nv",
            f"--header=API-Key: {API_KEY}",
            checksum_url
        ]

        subprocess.run(checksum_cmd, cwd=outdir, check=True)

        # Verify if sha256sum exists
        try:
            subprocess.run(
                ["sha256sum", "-c", "checksums.txt"],
                cwd=outdir,
                check=True
            )
            print(f"✓ Verified {sim} | Snapshot {snap}")
        except FileNotFoundError:
            print(f"(sha256sum not found — skipped verification for {snap})")

        return f"Done: {sim} {snap}"

    except Exception as e:
        return f"FAILED: {sim} {snap} | {e}"


def main():
    jobs = []

    for sim in SIMULATIONS:
        for snap in SNAPS:
            jobs.append((sim, snap))

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(download_snapshot, sim, snap)
            for sim, snap in jobs
        ]

        for future in as_completed(futures):
            print(future.result())


if __name__ == "__main__":
    main()