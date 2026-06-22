"""Wait for training to finish, then run the full downstream pipeline.

Polls until no `src.train` process remains, then executes (each guarded):
    evaluate -> interpret -> external_validation (GEPIII) -> paper

Intended to run in the background:  python -m src.finalize
"""
from __future__ import annotations

import subprocess
import time

from src.utils import load_config, get_logger

log = get_logger("finalize", logfile="logs/finalize.log")


def train_running() -> bool:
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object {$_.CommandLine -like '*src.train*'} | Measure-Object).Count"],
            capture_output=True, text=True, timeout=60).stdout.strip()
        return int(out) > 0
    except Exception:  # noqa: BLE001
        return False


def main() -> int:
    log.info("waiting for src.train to finish...")
    waited = 0
    while train_running():
        time.sleep(20)
        waited += 20
        if waited % 300 == 0:
            log.info("  ...still training (%d min elapsed)", waited // 60)
    log.info("training finished; beginning finalization")

    cfg = load_config()
    from src import evaluate, interpret, paper
    import src.external_validation as ext

    for label, fn in [("evaluate", lambda: evaluate.evaluate(cfg)),
                      ("interpret", lambda: interpret.interpret(cfg))]:
        try:
            fn()
            log.info("%s: OK", label)
        except Exception as e:  # noqa: BLE001
            log.exception("%s FAILED: %s", label, e)

    # external validation on GEPIII (guarded)
    try:
        if ext._check_data(cfg):
            df, _ = ext.build_external(cfg, max_buildings=60)
            df = ext.featurize_external(cfg, df)
            arr = ext.window_external(cfg, df)
            if len(arr["Y"]) > 0:
                ext.evaluate_external(cfg, arr)
                log.info("external_validation: OK")
            else:
                log.warning("external_validation: no windows built")
        else:
            log.warning("external_validation: GEPIII data missing")
    except Exception as e:  # noqa: BLE001
        log.exception("external_validation FAILED: %s", e)

    try:
        paper.build(cfg)
        log.info("paper: OK")
    except Exception as e:  # noqa: BLE001
        log.exception("paper FAILED: %s", e)

    log.info("FINALIZATION COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
