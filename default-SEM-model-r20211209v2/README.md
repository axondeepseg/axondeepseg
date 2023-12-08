# default-SEM-model
AxonDeepSeg default SEM model and testing image. This model works at a resolution of 0.1 micrometer per pixel and was trained on rat spinal cord data collected via a Scanning Electron Microscope (SEM).


# Steps to train this model
1. Get `ivadomed` version: [[55fc2067]](https://github.com/ivadomed/ivadomed/commit/55fc2067cbb9c97a711e32cf8b5a377fb6d517be)
2. Get the data: `data_axondeepseg_sem` (Dataset Annex version: 1cddcc6bef21782b22ff17f34502c6e90e22c504)
3. Copy the "model_seg_rat_axon-myelin_sem.json" and "split_dataset.joblib" files, and update the following fields: `fname_split`, `path_output`, `path_data` and `gpu_ids`.
4. Run ivadomed: `ivadomed -c path/to/the/config/file`
5. The trained model file will be saved under the `path_output` directory.
