import numpy as np
import pytest
from examples.libero.live_vision_intervention import box_mask, background_mask, matched_object_mask


def test_object_box_and_background_area():
    ids = np.full((20, 30), -1)
    ids[5:9, 8:10] = 105
    ids[6, 12] = 106
    box = box_mask(ids, [105, 106])
    assert box.sum() == 20
    forbidden = ids != -1
    forbidden[0:8, 20:30] = True
    bg = background_mask(box, forbidden)
    assert bg.sum() == box.sum()
    assert not (bg & forbidden).any()
    np.testing.assert_array_equal(bg, background_mask(box, forbidden))


def test_invisible_object_does_not_create_occluder():
    empty = box_mask(np.zeros((20, 30)), [5])
    assert not empty.any()
    assert not background_mask(empty, np.ones_like(empty)).any()


def test_background_fails_if_it_would_cover_an_object():
    target = np.zeros((20, 30), bool)
    target[1:9, 1:9] = True
    with pytest.raises(ValueError, match='Not enough visible background'):
        background_mask(target, np.ones_like(target))


def test_fragmented_background_uses_equal_area_without_object_overlap():
    target = np.zeros((20, 30), bool)
    target[1:9, 1:9] = True
    forbidden = np.zeros_like(target)
    forbidden[::3] = True
    mask = background_mask(target, forbidden)
    assert mask.sum() == target.sum()
    assert not (mask & forbidden).any()


def test_matched_area_when_background_is_smaller_than_object_box():
    ids=np.ones((20,30),int)
    ids[2:18,2:28]=9
    forbidden=ids==9
    target=matched_object_mask(ids,[9],forbidden)
    assert target.sum()==(~forbidden).sum()
    assert (ids[target]==9).all()
    background=background_mask(target,forbidden)
    assert background.sum()==target.sum()
    assert not (background & forbidden).any()
