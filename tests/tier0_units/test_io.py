"""Tier 0 — spikefi.utils.io: trial-numbering and filepath-construction
guarantees behind make_filepath, calculate_trial and rename_if_multiple.
"""


import os
from pathlib import Path

import spikefi.utils.io as sfio


def test_calculate_trial_is_zero_for_a_missing_or_empty_directory(
        tmp_path: Path
) -> None:
    """A non-existent or empty directory has no collisions, so the trial
    is 0 and rename_if_multiple leaves the name untouched."""
    missing_dir = str(tmp_path / 'does_not_exist')
    empty_dir = str(tmp_path)

    assert sfio.calculate_trial('a.pkl', missing_dir) == 0
    assert sfio.calculate_trial('a.pkl', empty_dir) == 0
    assert sfio.rename_if_multiple('a.pkl', empty_dir) == 'a.pkl'


def test_rename_if_multiple_suffixes_the_next_free_trial_number(
        tmp_path: Path
) -> None:
    """Each existing trial pushes the suggested name to the next free
    number, one step at a time."""
    d = str(tmp_path)
    (tmp_path / 'a.pkl').touch()
    assert sfio.rename_if_multiple('a.pkl', d) == 'a (1).pkl'

    (tmp_path / 'a (1).pkl').touch()
    assert sfio.rename_if_multiple('a.pkl', d) == 'a (2).pkl'


def test_calculate_trial_continues_past_the_highest_existing_trial(
        tmp_path: Path
) -> None:
    """The next trial is derived from the highest existing trial marker,
    not merely from how many colliding files exist."""
    d = str(tmp_path)
    (tmp_path / 'a.pkl').touch()
    (tmp_path / 'a (5).pkl').touch()

    assert sfio.calculate_trial('a.pkl', d) == 6


def test_calculate_trial_ignores_a_same_stem_file_with_another_extension(
        tmp_path: Path
) -> None:
    """A file sharing the stem but not the extension is not a collision.
    The fixture is extensionless: removesuffix(extension) is a no-op on it,
    so its bare name would otherwise still match the trial regex."""
    d = str(tmp_path)
    (tmp_path / 'a').touch()

    assert sfio.calculate_trial('a.pkl', d) == 0


def test_calculate_trial_ignores_a_file_that_merely_contains_the_name(
        tmp_path: Path
) -> None:
    """A file whose name merely contains fname as a substring, rather than
    matching it exactly (optionally with a trial marker), is not a
    collision."""
    d = str(tmp_path)
    (tmp_path / 'xxa (7).pkl').touch()

    assert sfio.calculate_trial('a.pkl', d) == 0
    assert sfio.rename_if_multiple('a.pkl', d) == 'a.pkl'


def test_make_filepath_creates_the_parent_directory_and_joins_the_name(
        tmp_path: Path
) -> None:
    """make_filepath creates parentdir if missing and returns the plain
    join of parentdir and fname when rename is not requested."""
    parentdir = str(tmp_path / 'fresh_subdir')

    fpath = sfio.make_filepath('a.pkl', parentdir)

    assert os.path.isdir(parentdir)
    assert fpath == os.path.join(parentdir, 'a.pkl')


def test_make_filepath_rename_flag_selects_between_overwrite_and_a_new_name(
        tmp_path: Path
) -> None:
    """With rename=False the same path is returned every time (overwrite
    semantics); with rename=True an existing file forces a new path."""
    parentdir = str(tmp_path)
    fpath = sfio.make_filepath('a.pkl', parentdir)
    open(fpath, 'w').close()

    same_path = sfio.make_filepath('a.pkl', parentdir, rename=False)
    new_path = sfio.make_filepath('a.pkl', parentdir, rename=True)

    assert same_path == fpath
    assert new_path != fpath


def test_each_make_filepath_wrapper_targets_its_own_directory() -> None:
    """Each convenience wrapper joins its own directory global, so a
    rebinding of that global (as tests_out_dir does) is honored."""
    assert os.path.dirname(sfio.make_res_filepath('a.pkl')) == sfio.RES_DIR
    assert os.path.dirname(sfio.make_fig_filepath('a.pkl')) == sfio.FIG_DIR
    assert os.path.dirname(sfio.make_net_filepath('a.pkl')) == sfio.NET_DIR
    assert os.path.dirname(sfio.make_out_filepath('a.pkl')) == sfio.OUT_DIR
