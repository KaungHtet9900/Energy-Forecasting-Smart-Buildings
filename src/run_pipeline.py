"""Run the full pipeline end-to-end in one process (for the scaled-up re-run).

Stages: preprocess -> features -> windowing -> eda -> train(all) -> evaluate ->
interpret -> external_validation -> paper -> export.

Core data/train stages are fatal (abort on failure); analysis/output stages are
guarded so one failure does not block the rest. Run in the background:

    python -m src.run_pipeline
"""
from __future__ import annotations

import sys
import time

from src.utils import get_logger

log = get_logger("pipeline", logfile="logs/pipeline.log")


def run(name, mainfn, fatal=True):
    log.info("======================== STAGE: %s ========================", name)
    t0 = time.time()
    saved = sys.argv
    sys.argv = [name]  # clean argv so each module's argparse sees defaults
    try:
        rc = mainfn()
        log.info("%s: OK (%.0fs, rc=%s)", name, time.time() - t0, rc)
        return True
    except SystemExit as e:
        if e.code not in (0, None):
            log.error("%s exited with %s", name, e.code)
            if fatal:
                raise
        return e.code in (0, None)
    except Exception as e:  # noqa: BLE001
        log.exception("%s FAILED: %s", name, e)
        if fatal:
            raise
        return False
    finally:
        sys.argv = saved


def main() -> int:
    from src.data import preprocess, features, windowing
    from src import eda, train, evaluate, interpret, paper, export
    import src.external_validation as ext

    t0 = time.time()
    run("preprocess", preprocess.main, fatal=True)
    run("features", features.main, fatal=True)
    run("windowing", windowing.main, fatal=True)
    run("eda", eda.main, fatal=False)
    run("train", train.main, fatal=True)
    run("evaluate", evaluate.main, fatal=False)
    run("interpret", interpret.main, fatal=False)
    run("external_validation", ext.main, fatal=False)
    run("paper", paper.main, fatal=False)
    run("export", export.main, fatal=False)
    log.info("PIPELINE COMPLETE in %.1f min", (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
