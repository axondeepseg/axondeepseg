'''
Model descriptions and download links for all models supported in AxonDeepSeg.
'''

MODELS = {
    "generalist": {
        "name": "model_seg_generalist",
        "task": "myelinated-axon-segmentation",
        "n_classes": 2,
        "model-info": "Multi-domain axon and myelin segmentation model trained on TEM, SEM, BF and CARS data.",
        "training-data": "Training data consists of an aggregation of 6 datasets covering multiple microscopy modalities, species and pathologies. For a detailed description of the 6 datasets used for training, see https://arxiv.org/abs/2409.11552",
        "weights": {
            "ensemble": "https://github.com/axondeepseg/model_seg_generalist/releases/download/r20240224/model_seg_axonmyelin_generalist.zip",
            "light": "https://github.com/axondeepseg/model_seg_generalist/releases/download/r20240416/model_seg_generalist_light.zip"
        }
    },
    "dedicated-BF": {
        "name": "model_seg_generalist_BF",
        "task": "myelinated-axon-segmentation",
        "n_classes": 2,
        "model-info": "Axon and myelin segmentation model trained on Bright-Field data.",
        "training-data": "3 different BF datasets were aggregated. For a detailed description of the dataset used for training, see https://arxiv.org/abs/2409.11552.",
        "weights": {
            "ensemble": None,
            "light": "https://github.com/axondeepseg/model_seg_generalist/releases/download/r20240416/model_seg_generalist_bf_light.zip"
        }
    },
    "dedicated-SEM": {
        "name": "model_seg_rat_axon-myelin_SEM",
        "task": "myelinated-axon-segmentation",
        "n_classes": 2,
        "model-info": "Axon and myelin segmentation model trained on Scanning Electron Microscopy data.",
        "training-data": "14.8 Mpx of SEM rat spinal cord images with a resolution of 0.13 um/px. The dataset used for this model is publicly available here: https://github.com/axondeepseg/data_axondeepseg_sem.",
        "weights": {
            "ensemble": "https://github.com/axondeepseg/default-SEM-model/releases/download/r20240403/model_seg_rat_axon-myelin_sem_ensemble.zip",
            "light": "https://github.com/axondeepseg/default-SEM-model/releases/download/r20240403/model_seg_rat_axon-myelin_sem_light.zip"
        }
    },
    "dedicated-CARS": {
        "name": "model_seg_rat_axon-myelin_CARS",
        "task": "myelinated-axon-segmentation",
        "n_classes": 2,
        "model-info": "Axon and myelin segmentation model trained on Coherent Anti-Stokes Raman Scattering data.",
        "training-data": "2.6 Mpx of CARS rat spinal cord images, resolution of 0.225 um/px. The training set is relatively small so the model is not as robust as the generalist model.",
        "weights": {
            "ensemble": "https://github.com/axondeepseg/default-CARS-model/releases/download/r20240403/model_seg_rat_axon-myelin_cars_ensemble.zip",
            "light": "https://github.com/axondeepseg/default-CARS-model/releases/download/r20240403/model_seg_rat_axon-myelin_cars_light.zip"
        }
    },
    "unmyelinated-TEM": {
        "name": "model_seg_unmyelinated_sickkids",
        "task": "unmyelinated-axon-segmentation",
        "n_classes": 1,
        "model-info": "Unmyelinated axon segmentation model trained on TEM data.",
        "training-data": "5 large TEM mouse images, resolution of 0.0048 um/px. Unmyelinated axons were manually annotated. The data comes from Sick Kids Hospital, Toronto.",
        "weights": {
            "ensemble": "https://github.com/axondeepseg/model_seg_unmyelinated_tem/releases/download/v1.1.0/model_seg_unmyelinated_sickkids_tem_best.zip",
            "light": None
        }
    }
}

def get_supported_models():
    supported_models = []
    for m in MODELS.keys():
        if MODELS[m]["weights"]["ensemble"] is not None:
            supported_models.append(MODELS[m]["name"] + "_ensemble")
        if MODELS[m]["weights"]["light"] is not None:
            supported_models.append(MODELS[m]["name"] + "_light")
    return supported_models