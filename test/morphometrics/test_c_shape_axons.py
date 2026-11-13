import numpy as np
import pytest
from scipy import ndimage as ndi
from AxonDeepSeg.morphometrics.compute_morphometrics import get_axon_morphometrics, get_watershed_segmentation


def _c_shaped_axon(shape=(240, 240), center=(120, 120), r_in=45, r_out=65, opening_deg=70):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    r = np.hypot(yy - center[0], xx - center[1])
    theta = np.degrees(np.arctan2(yy - center[0], xx - center[1]))
    return (r >= r_in) & (r <= r_out) & (np.abs(theta) > opening_deg / 2)


def _add_myelin(im_axon, thickness):
    return (ndi.distance_transform_edt(~im_axon) <= thickness) & ~im_axon


@pytest.mark.unit
def test_c_shaped_axon_keeps_all_its_pixels():
    im_axon = _c_shaped_axon()
    im_myelin = _add_myelin(im_axon, 8)
    # the centroid of a C shaped axon is in its opening, outside of the axon
    cy, cx = ndi.center_of_mass(im_axon)
    assert not im_axon[int(cy), int(cx)]

    stats, _, labels = get_axon_morphometrics(im_axon, im_myelin=im_myelin, pixel_size=0.1,
                                              return_index_image=True, return_im_axonmyelin_label=True)
    assert len(stats) == 1
    assert np.all(labels[im_axon] == 1)
    assert np.all(labels[im_myelin] == 1)
    assert not np.isnan(stats['gratio'][0])


@pytest.mark.unit
def test_c_shaped_axon_is_not_merged_with_the_axon_inside_its_opening():
    im_c = _c_shaped_axon()
    yy, xx = np.mgrid[:240, :240]
    im_round = np.hypot(yy - 120, xx - 120) <= 20
    im_axon = im_c | im_round
    im_myelin = _add_myelin(im_axon, 8)

    stats, _, labels = get_axon_morphometrics(im_axon, im_myelin=im_myelin, pixel_size=0.1,
                                              return_index_image=True, return_im_axonmyelin_label=True)
    assert len(stats) == 2
    lab_c = labels[im_c][0]
    lab_round = labels[im_round][0]
    # each axon stays entirely in its own instance
    assert lab_c != lab_round
    assert np.all(labels[im_c] == lab_c)
    assert np.all(labels[im_round] == lab_round)
    # the stats of each axon come from its own fiber
    row_c = lab_c - 1
    fiber_c = labels == lab_c
    assert stats['axonmyelin_area'][row_c] == pytest.approx(fiber_c.sum() * 0.1 ** 2)


@pytest.mark.unit
def test_watershed_keeps_axon_parts_only_connected_diagonally():
    # two blocks touching by a corner: one axon object (connectivity 2), but two parts for the watershed (connectivity 1)
    im_axon = np.zeros((120, 120), bool)
    im_axon[20:60, 20:60] = True
    im_axon[60:100, 60:100] = True
    # a neighbouring axon on the left of the first block
    im_neighbour = np.zeros_like(im_axon)
    im_neighbour[20:60, 2:10] = True
    im_myelin = _add_myelin(im_axon | im_neighbour, 6)
    labels = (lambda: get_watershed_segmentation(im_axon | im_neighbour, im_myelin))()
    lab = labels[60, 60]
    assert np.all(labels[im_axon] == lab)
    assert np.all(labels[im_neighbour] != lab)
