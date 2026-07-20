"""Watch-folder (live) reduction: polling, settling, stack growth, retries,
analysis triggering, and that the live file feeds the normal analysis Step 1.

Integration is injected (no pyFAI needed) — the seam the watcher exposes for
exactly this purpose.
"""
import sys
import tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py
from seriesxrd.reduce.watch import WatchSession, run_watch

NPT = 100
X = np.linspace(1.0, 8.0, NPT)


def _fake_integrate(task):
    i, src, want_cake = task
    if "bad" in str(src):
        return {"index": i, "file": str(src), "ok": False, "error": "boom"}
    y = 5.0 + 50.0 * np.exp(-((X - 3.0) ** 2) / 0.01)
    return {"index": i, "file": str(src), "ok": True, "error": "",
            "seconds": 0.01, "radial": X, "intensity": y,
            "intensity_robust": y.copy(), "intensity_sigmaclip": y.copy()}


def _cfg(td):
    return {"session_name": "live", "dataset_dir": str(td),
            "file_patterns": "*.tif;*.h5", "processed_root": str(td / "proc"),
            "logs_root": str(td / "logs"), "npt_1d": "100",
            "handoff_file": ""}


def _settled_touch(ws, path, payload=b"x"):
    """Create a file and run the settle cycle (first poll arms, second is
    allowed to process)."""
    Path(path).write_bytes(payload)
    n = ws.cycle()
    assert n == 0, "a brand-new file must not be processed on its first poll"


def test_settle_append_and_analysis_trigger():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        analyzed = []
        ws = WatchSession(_cfg(td), integrate=_fake_integrate,
                          analyze=analyzed.append, steps="12")
        assert ws.cycle() == 0                     # empty folder

        _settled_touch(ws, td / "a.tif")
        assert ws.cycle() == 1                     # settled -> appended
        assert analyzed == [str(ws.h5_path)]

        _settled_touch(ws, td / "b.tif")
        assert ws.cycle() == 1
        assert len(analyzed) == 2

        with h5py.File(str(ws.h5_path), "r") as h:
            assert bool(h.attrs["live_mode"])
            assert h["patterns/intensity"].shape == (2, NPT)
            assert h["patterns/intensity_robust"].shape == (2, NPT)
            names = [x.decode() for x in h["frames/filename"][:]]
            assert names == ["a.tif", "b.tif"]
            assert np.allclose(h["patterns/radial"][:], X)
            assert h["frames/ok"][:].all()

        # An empty poll must not re-trigger analysis.
        assert ws.cycle() == 0 and len(analyzed) == 2

        # The live file feeds the normal analysis Step 1 unchanged.
        from seriesxrd.analysis.background import run_background_separation
        out = td / "an.h5"
        man = run_background_separation(ws.h5_path, out)
        assert man["n_frames"] == 2 and out.is_file()


def test_failed_frame_retries_then_gives_up():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        ws = WatchSession(_cfg(td), integrate=_fake_integrate, steps="")
        _settled_touch(ws, td / "ok.tif")
        assert ws.cycle() == 1
        _settled_touch(ws, td / "bad.tif")
        for _ in range(3):                        # retried each poll
            assert ws.cycle() == 0
        assert "bad.tif" in next(iter(ws.failed))
        assert ws.cycle() == 0                    # failed: not retried again
        with h5py.File(str(ws.h5_path), "r") as h:
            assert h["patterns/intensity"].shape[0] == 1   # never appended


def test_growing_stack_holds_back_newest_frame():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        stack = td / "run.h5"
        with h5py.File(str(stack), "w") as h:
            h.create_dataset("entry/data/data",
                             data=np.zeros((3, 32, 24), "u2"),
                             maxshape=(None, 32, 24), chunks=(1, 32, 24))
            h.create_dataset("entry/data/timestamp",
                             data=1e9 + 2.0 * np.arange(3),
                             maxshape=(None,), chunks=(1,))
        ws = WatchSession(_cfg(td), integrate=_fake_integrate, steps="")
        # First poll: the stack just grew 0 -> 3, so the newest index (#2)
        # is held back (a half-written chunk must never be read).
        assert ws.cycle() == 2
        # Second poll: count stable -> the held-back frame lands.
        assert ws.cycle() == 1
        # Detector appends two more frames mid-run.
        with h5py.File(str(stack), "r+") as h:
            h["entry/data/data"].resize(5, axis=0)
            ts = h["entry/data/timestamp"]
            ts.resize(5, axis=0)
            ts[3:] = 1e9 + 2.0 * np.array([3, 4])
        assert ws.cycle() == 1                    # 3 appended, #4 held back
        assert ws.cycle() == 1                    # #4 lands
        with h5py.File(str(ws.h5_path), "r") as h:
            names = [x.decode() for x in h["frames/filename"][:]]
            assert [n.split("#")[-1] for n in names] == [
                "000000", "000001", "000002", "000003", "000004"]
            # harvested per-frame timestamps landed in arrival order
            stamps = [x.decode() for x in h["frames/timestamp"][:]]
            assert all(stamps), stamps
            from datetime import datetime
            secs = [datetime.fromisoformat(s).timestamp() for s in stamps]
            assert np.allclose(np.diff(secs), 2.0)


def test_analyze_every_and_finish_flush():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        analyzed = []
        ws = WatchSession(_cfg(td), integrate=_fake_integrate,
                          analyze=analyzed.append, steps="12", analyze_every=2)
        _settled_touch(ws, td / "a.tif")
        assert ws.cycle() == 1 and analyzed == []       # 1 of 2 batches
        _settled_touch(ws, td / "b.tif")
        assert ws.cycle() == 1 and len(analyzed) == 1   # 2nd batch triggers
        _settled_touch(ws, td / "c.tif")
        assert ws.cycle() == 1 and len(analyzed) == 1   # 1 pending again
        man = ws.finish()                               # flushes the leftover
        assert len(analyzed) == 2
        assert man["n_frames"] == 3 and man["n_failed"] == 0


def test_run_watch_loop_and_own_output_excluded():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cfg = _cfg(td)
        # the live output lives INSIDE the watched folder and matches *.h5 —
        # it must never be re-ingested as a frame source
        out = td / "reduced_live.h5"
        (td / "a.tif").write_bytes(b"x")
        man = run_watch(cfg, poll=0.01, steps="", max_cycles=4,
                        integrate=_fake_integrate, out_path=out)
        assert man["n_frames"] == 1
        with h5py.File(str(out), "r") as h:
            names = [x.decode() for x in h["frames/filename"][:]]
            assert names == ["a.tif"]


def test_resume_skips_stored_frames_and_appends_new():
    """An interrupted watch resumes into its live file: stored frames are
    skipped (matched by stored name), new frames append, and shapes/channels
    come from the file, not the config."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cfg = _cfg(td)
        out = td / "proc" / "live.h5"
        out.parent.mkdir(parents=True)
        ws1 = WatchSession(cfg, integrate=_fake_integrate, steps="",
                           out_path=out)
        _settled_touch(ws1, td / "a.tif")
        assert ws1.cycle() == 1
        _settled_touch(ws1, td / "b.tif")
        assert ws1.cycle() == 1
        ws1.finish()

        # a fresh session on the same output must refuse without resume
        try:
            WatchSession(cfg, integrate=_fake_integrate, steps="",
                         out_path=out)
            assert False, "expected ValueError on existing --out"
        except ValueError as e:
            assert "resume" in str(e)

        # resume: a.tif/b.tif are skipped, only the new frame lands
        (td / "c.tif").write_bytes(b"x")
        ws2 = WatchSession(cfg, integrate=_fake_integrate, steps="",
                           out_path=out, resume=True)
        assert ws2.writer.n == 2 and "a.tif" in ws2.seen and "b.tif" in ws2.seen
        assert ws2.cycle() == 0                    # settle arm for c.tif
        assert ws2.cycle() == 1
        assert ws2.cycle() == 0                    # nothing re-processed
        man = ws2.finish()
        assert man["n_frames"] == 1                # THIS session appended one
        with h5py.File(str(out), "r") as h:
            names = [x.decode() for x in h["frames/filename"][:]]
            assert names == ["a.tif", "b.tif", "c.tif"]
            assert h["patterns/intensity"].shape == (3, NPT)
            assert np.allclose(h["patterns/radial"][:], X)   # not rewritten
            assert list(h["frames/frame_index"][:]) == [0, 1, 2]

        # resume of a missing file errors cleanly
        try:
            WatchSession(cfg, integrate=_fake_integrate, steps="",
                         out_path=td / "nope.h5", resume=True)
            assert False, "expected FileNotFoundError"
        except FileNotFoundError:
            pass
        # resume refuses a non-live reduced file
        plain = td / "proc" / "plain.h5"
        with h5py.File(str(plain), "w") as h:
            h.create_group("patterns").create_dataset(
                "intensity", data=np.zeros((1, 4), "f4"))
        try:
            WatchSession(cfg, integrate=_fake_integrate, steps="",
                         out_path=plain, resume=True)
            assert False, "expected ValueError for a non-live file"
        except ValueError as e:
            assert "live-mode" in str(e)


def test_resume_channel_flags_follow_file():
    """A live file written without the sigmaclip channel keeps its shape on
    resume even if the config would now enable that channel."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        cfg = _cfg(td)
        cfg["sigmaclip_1d"] = False
        out = td / "live.h5"
        ws1 = WatchSession(cfg, integrate=_fake_integrate, steps="",
                           out_path=out)
        _settled_touch(ws1, td / "a.tif")
        assert ws1.cycle() == 1
        with h5py.File(str(out), "r") as h:
            assert "intensity_sigmaclip" not in h["patterns"]

        cfg2 = dict(cfg)
        cfg2["sigmaclip_1d"] = True                # config changed mid-run
        (td / "b.tif").write_bytes(b"x")
        ws2 = WatchSession(cfg2, integrate=_fake_integrate, steps="",
                           out_path=out, resume=True)
        ws2.cycle(); ws2.cycle()
        with h5py.File(str(out), "r") as h:
            assert "intensity_sigmaclip" not in h["patterns"]   # file wins
            assert h["patterns/intensity"].shape[0] == 2


def main() -> None:
    test_settle_append_and_analysis_trigger()
    test_failed_frame_retries_then_gives_up()
    test_growing_stack_holds_back_newest_frame()
    test_analyze_every_and_finish_flush()
    test_run_watch_loop_and_own_output_excluded()
    test_resume_skips_stored_frames_and_appends_new()
    test_resume_channel_flags_follow_file()
    print("WATCH TEST OK")


if __name__ == "__main__":
    main()
