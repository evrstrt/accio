from accio.core.extract import pano_frames


def test_pano_frames_ordering_and_timestamps(tmp_path):
    for i in (2, 0, 1, 10):
        (tmp_path / f"pano_{i:05d}.jpg").touch()
    frames = pano_frames(tmp_path, fps=2.0)
    assert [f.index for f in frames] == [0, 1, 2, 10]
    assert [f.t_sec for f in frames] == [0.0, 0.5, 1.0, 5.0]


def test_pano_frames_empty_dir(tmp_path):
    assert pano_frames(tmp_path, fps=2.0) == []
