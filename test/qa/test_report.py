# coding: utf-8

import base64
import re
from io import BytesIO

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from AxonDeepSeg.morphometrics.compute_morphometrics import (
    get_axon_morphometrics,
    save_axon_morphometrics,
)
from AxonDeepSeg.qa.report import build_axon_data, encode_label_map, generate_qa_report


def _decode(data_uri):
    header, payload = data_uri.split(",", 1)
    assert header == "data:image/png;base64"
    return np.asarray(Image.open(BytesIO(base64.b64decode(payload))).convert("RGB"))


def _synthetic_masks(n_side=3):
    """Grid of round axons with a myelin ring, enough to get a few morphometrics rows."""
    yy, xx = np.mgrid[0:200, 0:200]
    axon = np.zeros((200, 200), bool)
    myelin = np.zeros((200, 200), bool)
    for i in range(n_side):
        for j in range(n_side):
            cy, cx = 35 + 65 * i, 35 + 65 * j
            r = np.hypot(yy - cy, xx - cx)
            axon |= r < 15
            myelin |= (r >= 15) & (r < 22)
    image = np.where(axon, 200, np.where(myelin, 40, 120)).astype(np.uint8)
    return image, axon, myelin


class TestCore(object):
    @pytest.mark.unit
    def test_encode_label_map_round_trip_keeps_labels_above_255(self):
        labels = np.zeros((4, 4), np.int64)
        labels[0, 0] = 1
        labels[1, 1] = 300
        labels[2, 2] = 65535
        axon = labels > 0
        myelin = np.zeros_like(axon)
        myelin[3, 3] = True

        rgb = _decode(encode_label_map(labels, axon, myelin))

        decoded = rgb[..., 0].astype(int) + 256 * rgb[..., 1].astype(int)
        assert decoded.tolist() == labels.tolist()
        assert rgb[1, 1, 2] == 1 and rgb[3, 3, 2] == 2 and rgb[0, 1, 2] == 0

    @pytest.mark.unit
    def test_encode_label_map_rejects_too_many_axons(self):
        labels = np.array([[65536]])
        with pytest.raises(ValueError):
            encode_label_map(labels, labels > 0, labels < 0)

    @pytest.mark.unit
    def test_build_axon_data_gives_bbox_per_morphometrics_row(self):
        labels = np.zeros((10, 10), int)
        labels[1:3, 2:5] = 1   # row 0
        labels[6:9, 6:8] = 3   # row 2, row 1 has no pixels
        df = pd.DataFrame({
            "axon_diam (um)": [1.0, 0.5, 2.0],
            "myelin_thickness (um)": [0.2, 0.1, 0.3],
            "gratio": [0.7, np.nan, 0.8],
            "image_border_touching": [False, False, True],
        })

        data = build_axon_data(df, labels)

        assert [a["id"] for a in data] == [0, 1, 2]
        assert data[0]["bbox"] == [2, 1, 5, 3]
        assert data[1]["bbox"] is None
        assert data[2]["bbox"] == [6, 6, 8, 9]
        assert data[2]["border"] is True

    @pytest.mark.integration
    def test_generate_qa_report_writes_one_self_contained_file(self, tmp_path):
        image, axon, myelin = _synthetic_masks()
        stats, _, _, labels = get_axon_morphometrics(
            im_axon=axon, im_myelin=myelin, pixel_size=0.1,
            return_index_image=True, return_border_info=True,
            return_instance_seg=True, return_im_axonmyelin_label=True,
        )
        morpho = tmp_path / "Morphometrics.csv"
        save_axon_morphometrics(morpho, stats)

        report = generate_qa_report(
            image, axon, myelin, labels, morpho, tmp_path / "QA",
            image_name="sample", default_filters={"min_diameter": 0.3},
        )

        html = report.read_text(encoding="utf-8")
        assert report.name == "AxonDeepSeg_QA_Report.html"
        # every image is embedded: no src pointing to a local file
        local_srcs = [s for s in re.findall(r'src="([^"]+)"', html)
                      if not s.startswith(("data:", "http", "${"))]
        assert local_srcs == []
        assert html.count("data:image/jpeg;base64,") >= 1
        assert '"min_diameter": 0.3' in html
        assert f'"id": {len(stats) - 1}' in html
        # no close-up PNGs written anymore
        assert not list((tmp_path / "QA").glob("axon_*_labeled.png"))
