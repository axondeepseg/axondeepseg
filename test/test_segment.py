# coding: utf-8

import pytest
import os
from AxonDeepSeg.segment import *

class TestCore(object):
    def setup(self):

        self.testPath = os.path.dirname(os.path.abspath(__file__))

        self.projectPath = os.path.split(self.testPath)[0]

        self.modelPath = os.path.join(self.projectPath, 'AxonDeepSeg/models/default_SEM_model_v1')
        self.imageFolderPath = os.path.join(self.testPath, '__test_segment_files__')
        self.imagePath = os.path.join(self.imageFolderPath, 'image.png')
        self.statsFilename = 'model_statistics_validation.json'

    @classmethod
    def teardown_class(cls):
        testPath = os.path.dirname(os.path.abspath(__file__))
        projectPath = os.path.split(testPath)[0]
        imageFolderPath = os.path.join(testPath, '__test_segment_files__')

        outputFiles = ['image_seg-axon.png', 'image_seg-myelin.png', 'image_seg-axonmyelin.png']
        for fileName in outputFiles:
            if os.path.exists(os.path.join(imageFolderPath, fileName)):
                os.remove(os.path.join(imageFolderPath, fileName))

    #--------------generate_config_dict tests--------------#
    @pytest.mark.unittest
    def test_generate_config_dict_outputs_dict(self):

        config = generate_config_dict(os.path.join(self.modelPath,'config_network.json'))

        assert type(config) is dict

    @pytest.mark.unittest
    def test_generate_config_dict_throws_exception_for_nonexisting_file(self):

        with pytest.raises(ValueError):
            config = generate_config_dict(os.path.join(self.modelPath,'n0n_3xist1ng_f1l3.json'))

    #--------------generate_resolution tests--------------#
    @pytest.mark.unittest
    def test_generate_resolution_returns_expected_known_project_cases(self):

        assert generate_resolution('SEM', 512) == 0.1
        assert generate_resolution('SEM', 256) == 0.2
        assert generate_resolution('TEM', 512) == 0.01


    #--------------segment_folders tests--------------#
    @pytest.mark.integritytest
    def test_foo1(self):

        path_model, config = generate_default_parameters('SEM', self.modelPath)

        overlap_value = 25
        resolution_model = generate_resolution('SEM', 512)
        segment_folders(path_testing_images_folder = self.imageFolderPath, path_model =  path_model, 
                        overlap_value = overlap_value, config = config, resolution_model = resolution_model,
                        acquired_resolution = 0.37, verbosity_level=2)
        
    #--------------segment_image tests--------------#
    @pytest.mark.integritytest
    def test_foo2(self):

        path_model, config = generate_default_parameters('SEM', self.modelPath)

        overlap_value = 25
        resolution_model = generate_resolution('SEM', 512)
        segment_image(path_testing_image = self.imagePath, path_model =  path_model, 
                      overlap_value = overlap_value, config = config, resolution_model = resolution_model,
                       acquired_resolution = 0.37, verbosity_level=2)
