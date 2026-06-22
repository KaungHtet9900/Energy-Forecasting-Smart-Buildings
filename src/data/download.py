"""Download the BDG2 and ASHRAE GEPIII datasets.

BDG2 files are stored with Git LFS, so they must be pulled from the LFS *media*
endpoint (``media.githubusercontent.com/media/...``) rather than ``raw...`` which
only returns ~130-byte pointer stubs.

GEPIII is a Kaggle competition; the account must have *joined* the competition
(accepted the rules) before the API can download it.

Usage::

    python -m src.data.download            # both (BDG2 always; GEPIII if possible)
    python -m src.data.download --bdg2
    python -m src.data.download --gepiii
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import requests

from src.utils import get_logger, load_config

log = get_logger("download", logfile="logs/download.log")
CHUNK = 1 << 20  # 1 MiB


# --------------------------------------------------------------------------
# BDG2
# --------------------------------------------------------------------------
def _download_stream(url: str, dest: Path, min_bytes: int = 1000) -> bool:
    """Stream a URL to disk with a progress log. Skip if already present & valid."""
    if dest.exists() and dest.stat().st_size > min_bytes:
        log.info("skip (exists, %.1f MB): %s", dest.stat().st_size / 1e6, dest.name)
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            last_pct = -10
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(CHUNK):
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = int(done * 100 / total)
                        if pct >= last_pct + 10:
                            log.info("  %s: %d%% (%.1f/%.1f MB)", dest.name, pct,
                                     done / 1e6, total / 1e6)
                            last_pct = pct
        if tmp.stat().st_size < min_bytes:
            raise IOError(f"downloaded file too small ({tmp.stat().st_size} B) "
                          f"- likely an LFS pointer or error page")
        tmp.replace(dest)
        log.info("done (%.1f MB): %s", dest.stat().st_size / 1e6, dest.name)
        return True
    except Exception as e:  # noqa: BLE001
        log.error("FAILED %s -> %s : %s", url, dest.name, e)
        if tmp.exists():
            tmp.unlink()
        return False


def download_bdg2(cfg) -> bool:
    base = cfg["download"]["bdg2_base"].rstrip("/")
    out = cfg["paths"]["raw_bdg2"]
    ok = True
    for name, rel in cfg["download"]["bdg2_files"].items():
        url = f"{base}/{rel}"
        dest = out / f"{name}.csv"
        ok &= _download_stream(url, dest)
    return ok


# --------------------------------------------------------------------------
# GEPIII (Kaggle)
# --------------------------------------------------------------------------
def download_gepiii(cfg) -> bool:
    """Download the labeled GEPIII files needed for external validation.

    Primary source is a public Kaggle *dataset* mirror (no competition-join needed);
    only the labeled files are fetched (building_metadata, train=2016, weather_train).
    Falls back to the official competition API if a mirror is not configured.
    """
    out = cfg["paths"]["raw_gepiii"]
    out.mkdir(parents=True, exist_ok=True)
    mirror = cfg.get("external", {}).get("gepiii_mirror")
    needed = ["building_metadata.csv", "train.csv", "weather_train.csv"]

    if all((out / f).exists() for f in needed):
        log.info("skip GEPIII (already present in %s)", out)
        return True

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:  # noqa: BLE001
        log.error("kaggle package unavailable: %s", e)
        return False
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:  # noqa: BLE001
        log.error("Kaggle auth failed (check ~/.kaggle credentials): %s", e)
        return False

    if mirror:
        log.info("downloading GEPIII labeled files from mirror '%s'", mirror)
        for f in needed:
            if (out / f).exists():
                continue
            try:
                api.dataset_download_file(mirror, f, path=str(out), quiet=False)
            except Exception as e:  # noqa: BLE001
                log.error("failed to fetch %s from mirror: %s", f, e)
                return False
    else:
        comp = cfg["download"]["gepiii_competition"]
        log.info("downloading GEPIII competition '%s' (requires join)", comp)
        try:
            api.competition_download_files(comp, path=str(out), quiet=False)
        except Exception as e:  # noqa: BLE001
            log.error("GEPIII competition download failed (join required?): %s", e)
            return False

    # extract any *.zip / *.csv.zip singles
    for zf in list(out.glob("*.zip")):
        with zipfile.ZipFile(zf) as z:
            z.extractall(out)
        zf.unlink()
    log.info("GEPIII ready: %s", sorted(p.name for p in out.glob("*.csv")))
    return True


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Download BDG2 + GEPIII datasets")
    ap.add_argument("--bdg2", action="store_true", help="download BDG2 only")
    ap.add_argument("--gepiii", action="store_true", help="download GEPIII only")
    args = ap.parse_args()

    cfg = load_config()
    do_bdg2 = args.bdg2 or not (args.bdg2 or args.gepiii)
    do_gepiii = args.gepiii or not (args.bdg2 or args.gepiii)

    ok = True
    if do_bdg2:
        log.info("=== BDG2 ===")
        ok &= download_bdg2(cfg)
    if do_gepiii:
        log.info("=== GEPIII ===")
        g = download_gepiii(cfg)
        if not g:
            log.warning("GEPIII not downloaded (see message above). "
                        "Pipeline can proceed on BDG2; run --gepiii later.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
