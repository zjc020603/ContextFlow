"""Pixel-only current-camera controls; simulator physics and robot state remain intact."""
import numpy as np
from PIL import Image

CAMERAS = (('agentview', 'observation/image'), ('robot0_eye_in_hand', 'observation/wrist_image'))


def segmentation(env, camera):
    # Match image_for_policy: raw render -> rotate 180 deg -> 224 square.
    raw = env.sim.render(256, 256, camera_name=camera, segmentation=True)
    ids = np.where(raw[..., 0] == 5, raw[..., 1], -1).astype(np.int32)[::-1, ::-1]
    return np.asarray(Image.fromarray(ids, mode='I').resize((224, 224), Image.NEAREST))


def object_geom_ids(env, name):
    obj = env.env.objects_dict[name]
    names = list(obj.contact_geoms) + list(obj.visual_geoms)
    return [env.sim.model.geom_name2id(n) for n in names]


def box_mask(ids, geoms):
    ys, xs = np.where(np.isin(ids, geoms))
    mask = np.zeros(ids.shape, bool)
    if len(xs):
        mask[ys.min():ys.max()+1, xs.min():xs.max()+1] = True
    return mask


def background_mask(target, forbidden):
    """Equal area on visible background; prefer same rectangle, otherwise compact pixel region."""
    ys, xs = np.where(target)
    if not len(xs):
        return np.zeros_like(target)
    h, w = ys.max()-ys.min()+1, xs.max()-xs.min()+1
    integral = np.pad(forbidden.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    counts = integral[h:, w:] - integral[:-h, w:] - integral[h:, :-w] + integral[:-h, :-w]
    yy, xx = np.where(counts == 0)
    if not len(xx) or h*w != int(target.sum()):
        # The wrist view may have no sufficiently large empty rectangle.
        # Use exactly the same number of background pixels, never spill onto objects.
        yy, xx = np.where(~forbidden)
        count = int(target.sum())
        if len(xx) < count:
            raise ValueError(f'Not enough visible background pixels: {len(xx)} < {count}')
        center_y, center_x = ys.mean(), xs.mean()
        seed = np.argmax((yy-center_y)**2 + (xx-center_x)**2)
        order = np.argsort((yy-yy[seed])**2 + (xx-xx[seed])**2, kind='stable')[:count]
        mask = np.zeros_like(target)
        mask[yy[order], xx[order]] = True
        assert mask.sum() == target.sum() and not np.any(mask & forbidden)
        return mask
    # Same search rule every frame. Top-left tie breaking is deterministic.
    i = np.argmax((yy - ys.min())**2 + (xx - xs.min())**2)
    mask = np.zeros_like(target)
    mask[yy[i]:yy[i]+h, xx[i]:xx[i]+w] = True
    assert mask.sum() == target.sum() and not np.any(mask & forbidden)
    return mask



def matched_object_mask(ids, geoms, forbidden):
    """Cap both object/background controls at the available background area.

    Prefer covering actual object pixels, then nearby box pixels. Full box when
    feasible; exact shared area budget otherwise, explicitly reported as partial.
    """
    full = box_mask(ids, geoms)
    budget = int((~forbidden).sum())
    if int(full.sum()) <= budget:
        return full
    ys, xs = np.where(full)
    out = np.zeros_like(full)
    if budget:
        visible = np.isin(ids[ys, xs], geoms)
        distance = (ys - ys.mean())**2 + (xs - xs.mean())**2
        order = np.lexsort((distance, ~visible))[:budget]
        out[ys[order], xs[order]] = True
    return out

def intervene(env, request, target, other, mode):
    result = dict(request)
    masks, counts = {}, {}
    before = env.get_sim_state().copy()
    for camera, key in CAMERAS:
        ids = segmentation(env, camera)
        forbidden_ids = []
        for i in range(env.sim.model.ngeom):
            name = env.sim.model.geom_id2name(i) or ''
            if name.startswith(('robot', 'gripper')) or any(name.startswith(obj) for obj in env.env.objects_dict):
                forbidden_ids.append(i)
        forbidden = np.isin(ids, forbidden_ids)
        target_geoms = object_geom_ids(env, target)
        full_target = box_mask(ids, target_geoms)
        target_mask = matched_object_mask(ids, target_geoms, forbidden)
        if mode == 'mask_target':
            mask = target_mask
        elif mode == 'mask_other':
            mask = matched_object_mask(ids, object_geom_ids(env, other), forbidden)
        elif mode == 'mask_background':
            mask = background_mask(target_mask, forbidden)
        else:
            raise ValueError(mode)
        image = request[key].copy()
        image[mask] = 127
        assert np.array_equal(image[~mask], request[key][~mask])
        result[key] = image
        masks[key] = mask
        counts[key] = {'masked_pixels': int(mask.sum()), 'target_box_pixels': int(target_mask.sum()),
                       'full_target_box_pixels': int(full_target.sum()),
                       'area_cap_active': bool(target_mask.sum() < full_target.sum()),
                       'target_visible_pixels_occluded': int((mask & np.isin(ids, target_geoms)).sum()),
                       'target_visible_pixels': int(np.isin(ids, object_geom_ids(env, target)).sum()),
                       'mask_is_rectangle': bool(mask.sum() == ((np.ptp(np.where(mask)[0])+1)*(np.ptp(np.where(mask)[1])+1))) if mask.any() else True,
                       'robot_pixels_occluded': int((mask & np.isin(ids, [i for i in range(env.sim.model.ngeom)
                           if (env.sim.model.geom_id2name(i) or '').startswith(('robot', 'gripper'))])).sum())}
    np.testing.assert_array_equal(before, env.get_sim_state())
    return result, masks, counts
