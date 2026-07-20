"""HDF5/NeXus stack ingestion (core/io): Eiger-style master files expand into
per-frame sources and read back slice-by-slice through the same
read_detector_image call the reduce workers use."""
import sys
import tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py
from seriesxrd.core.io import (
    h5_stack_info, expand_frame_sources, read_detector_image,
    frame_display_name, is_h5_frame_spec, parse_h5_frame_spec,
)


def _stack(path, n=4, shape=(32, 24), dset="entry/data/data", attrs=None):
    with h5py.File(str(path), "w") as h:
        data = np.arange(n * shape[0] * shape[1], dtype="u4").reshape(n, *shape)
        h.create_dataset(dset, data=data)
        for k, v in (attrs or {}).items():
            h.attrs[k] = v
    return path


def test_stack_info_autodetect():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # NeXus-convention path found without hints.
        p = _stack(td / "run.h5")
        info = h5_stack_info(p)
        assert info["ok"], info["error"]
        assert info["data_path"] == "entry/data/data"
        assert info["n_frames"] == 4 and info["frame_shape"] == (32, 24)

        # Unusual layout: found by the walk (largest 3D image dataset wins).
        q = td / "odd.h5"
        with h5py.File(str(q), "w") as h:
            h.create_dataset("small/stack", data=np.zeros((2, 20, 20), "u2"))
            h.create_dataset("big/stack", data=np.zeros((6, 20, 20), "u2"))
            h.create_dataset("not_image", data=np.zeros((100, 3)))  # too thin
        info = h5_stack_info(q)
        assert info["ok"] and info["data_path"] == "big/stack" and info["n_frames"] == 6

        # A single 2D image container = one frame.
        s = td / "single.h5"
        with h5py.File(str(s), "w") as h:
            h.create_dataset("entry/data/data", data=np.zeros((40, 40), "u2"))
        info = h5_stack_info(s)
        assert info["ok"] and info["n_frames"] == 1 and info["ndim"] == 2

        # seriesxrd's own outputs are refused (a results file in the data folder
        # must not be re-ingested as frames)...
        r = _stack(td / "reduced_x.h5", attrs={"tool": "seriesxrd.reduce"})
        info = h5_stack_info(r)
        assert not info["ok"] and "seriesxrd output" in info["error"]
        # ...unless an explicit data path overrides the refusal.
        info = h5_stack_info(r, "entry/data/data")
        assert info["ok"] and info["n_frames"] == 4

        # Explicit path that doesn't exist errors cleanly.
        info = h5_stack_info(p, "nope/data")
        assert not info["ok"] and "no dataset" in info["error"]


def test_expand_and_read_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        _stack(td / "b_run.h5", n=3)
        # plain single-image files pass through untouched
        tif_a, tif_c = td / "a.tif", td / "c.tif"
        tif_a.write_bytes(b"")
        tif_c.write_bytes(b"")
        # a seriesxrd output in the same folder is skipped, not fatal
        _stack(td / "reduced_old.h5", attrs={"schema_version": "1"})

        files = sorted(td.iterdir())
        sources, n_stacks = expand_frame_sources(files)
        assert n_stacks == 1
        names = [frame_display_name(s, td) for s in sources]
        assert names == ["a.tif",
                         "b_run.h5::entry/data/data#000000",
                         "b_run.h5::entry/data/data#000001",
                         "b_run.h5::entry/data/data#000002",
                         "c.tif"], names

        spec = sources[2]                      # frame 1 of the stack
        assert is_h5_frame_spec(spec) and not is_h5_frame_spec(str(tif_a))
        f, d, i = parse_h5_frame_spec(spec)
        assert f.name == "b_run.h5" and d == "entry/data/data" and i == 1

        img = read_detector_image(spec)
        assert img.shape == (32, 24) and img.dtype == np.float32
        # slice content matches what was stored for frame 1
        expect = np.arange(3 * 32 * 24, dtype="u4").reshape(3, 32, 24)[1]
        assert np.allclose(img, expect)
        # flip works on the stack path too
        assert np.allclose(read_detector_image(spec, flip_up_down=True),
                           np.flipud(expect))

        # out-of-range frame errors cleanly
        bad = spec.replace("#000001", "#000009")
        try:
            read_detector_image(bad)
            assert False, "expected IndexError"
        except IndexError:
            pass


def test_explicit_data_path_expansion():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        p = td / "x.h5"
        with h5py.File(str(p), "w") as h:
            h.create_dataset("custom/frames", data=np.zeros((2, 30, 30), "u2"))
        # auto-detect finds it via the walk...
        srcs, n = expand_frame_sources([p])
        assert n == 1 and len(srcs) == 2
        # ...and the explicit path pins it (wrong path -> skipped, not fatal)
        srcs, n = expand_frame_sources([p], "custom/frames")
        assert n == 1 and srcs[0].endswith("custom/frames#000000")
        srcs, n = expand_frame_sources([p], "wrong/path")
        assert n == 0 and srcs == []


def test_stack_metadata_harvest():
    """NeXus per-frame metadata (timestamps, positions, temperature) is
    harvested from stack containers and aligned onto the expanded sources."""
    from seriesxrd.core.io import h5_stack_metadata, harvest_stack_metadata
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        p = td / "run.h5"
        n = 4
        with h5py.File(str(p), "w") as h:
            h.create_dataset("entry/data/data",
                             data=np.zeros((n, 32, 24), "u2"))
            # numeric per-frame timestamps (epoch seconds; only differences
            # matter downstream)
            h.create_dataset("entry/data/timestamp",
                             data=1.7e9 + 2.0 * np.arange(n))
            # ESRF-style positioners; y is a SCALAR (a still row) → broadcast
            h.create_dataset("entry/instrument/positioners/samx",
                             data=0.05 * np.arange(n))
            h.create_dataset("entry/instrument/positioners/samy", data=1.5)
            h.create_dataset("entry/sample/temperature",
                             data=300.0 + np.arange(n))

        meta = h5_stack_metadata(p, n)
        assert meta["pos_x"] is not None and np.allclose(
            meta["pos_x"], [0.0, 0.05, 0.10, 0.15])
        assert meta["pos_y"] is not None and np.allclose(meta["pos_y"], 1.5)
        assert meta["temperature"][2] == 302.0
        assert meta["timestamp"] is not None and len(meta["timestamp"]) == n
        # ISO strings whose differences reproduce the 2 s cadence
        from datetime import datetime
        t0 = datetime.fromisoformat(meta["timestamp"][0])
        t1 = datetime.fromisoformat(meta["timestamp"][1])
        assert abs((t1 - t0).total_seconds() - 2.0) < 1e-6

        # aligned onto a mixed expanded source list (plain file rows stay empty)
        tif = td / "a.tif"
        tif.write_bytes(b"")
        sources, _ = expand_frame_sources(sorted(td.iterdir()))
        hv = harvest_stack_metadata(sources)
        assert len(hv["timestamp"]) == n + 1
        k_plain = [i for i, s in enumerate(sources) if s.endswith("a.tif")][0]
        assert hv["timestamp"][k_plain] == "" and np.isnan(hv["pos_x"][k_plain])
        k2 = [i for i, s in enumerate(sources) if s.endswith("#000002")][0]
        assert hv["pos_x"][k2] == 0.10 and hv["temperature"][k2] == 302.0
        assert hv["n_frames_with_meta"] == n

        # explicit-path override beats the probe
        with h5py.File(str(p), "r+") as h:
            h.create_dataset("odd/place/tt", data=np.full(n, 77.0))
        meta = h5_stack_metadata(p, n, temperature_path="odd/place/tt")
        assert np.allclose(meta["temperature"], 77.0)

        # a container with images but no metadata → all None, no error
        q = td / "bare.h5"
        with h5py.File(str(q), "w") as h:
            h.create_dataset("entry/data/data", data=np.zeros((2, 32, 24), "u2"))
        meta = h5_stack_metadata(q, 2)
        assert all(meta[k] is None for k in
                   ("timestamp", "pos_x", "pos_y", "temperature"))


def test_harvested_positions_flow_to_analysis():
    """Reduced-file pos_x/pos_y (as the NeXus harvest writes them) are carried
    into the analysis file by Step 1, feeding the coordinate grid map."""
    from seriesxrd.analysis.background import run_background_separation
    from seriesxrd.analysis.frame_metadata import read_frame_metadata
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        red = td / "red.h5"
        n, nb = 4, 24
        rng = np.random.default_rng(0)
        mean = rng.normal(10, 1, (n, nb)).astype("f4")
        with h5py.File(str(red), "w") as h:
            h.attrs["unit"] = "q_A^-1"
            gp = h.create_group("patterns")
            gp.create_dataset("intensity", data=mean)
            gp.create_dataset("intensity_robust", data=mean.copy())
            gp.create_dataset("radial", data=np.linspace(1, 8, nb))
            gf = h.create_group("frames")
            gf.create_dataset("filename",
                              data=np.array([f"f{i}.h5::entry/data/data#{i:06d}"
                                             for i in range(n)], dtype=object),
                              dtype=h5py.string_dtype(encoding="utf-8"))
            gf.create_dataset("excluded", data=np.zeros(n, "?"))
            gf.create_dataset("pressure", data=np.full(n, np.nan))
            gf.create_dataset("pos_x", data=np.array([0.0, 0.1, 0.0, 0.1]))
            gf.create_dataset("pos_y", data=np.array([0.0, 0.0, 0.1, 0.1]))
        out = td / "an.h5"
        run_background_separation(red, out)
        md = read_frame_metadata(out)
        assert np.allclose(md["pos_x"], [0.0, 0.1, 0.0, 0.1])
        assert np.allclose(md["pos_y"], [0.0, 0.0, 0.1, 0.1])


def main() -> None:
    test_stack_info_autodetect()
    test_expand_and_read_roundtrip()
    test_explicit_data_path_expansion()
    test_stack_metadata_harvest()
    test_harvested_positions_flow_to_analysis()
    print("H5 INGEST TEST OK")


if __name__ == "__main__":
    main()
