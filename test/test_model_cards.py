import pytest
import requests

import ads_base.model_cards as cards

HTTP_FOUND = 302

class TestCore(object):
    def setup_method(self):
        self.model_dict = cards.MODELS

    # --------------model_cards tests-------------- #
    @pytest.mark.unit
    def test_all_fields_are_present(self):
        for _, model_info in self.model_dict.items():
            assert list(model_info.keys()).sort() == [
                "name", 
                "task", 
                "n_classes", 
                "model-info", 
                "training-data", 
                "weights"
            ].sort()
    
    @pytest.mark.unit
    def test_all_URLs_are_valid(self):
        for _, model_info in self.model_dict.items():
            for _, url in model_info["weights"].items():
                if url is not None:
                    assert requests.head(url).status_code == HTTP_FOUND