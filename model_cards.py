models = {
    "generalist": {
        "name": "model_seg_generalist",
        "task": "myelinated-axon-segmentation",
        "n_classes": 2,
        "model-info": "Multi-domain axon and myelin segmentation model trained on TEM, SEM, BF and CARS data.",
        "training-data": "Training data consists of an aggregation of 6 datasets covering multiple microscopy modalities, species and pathologies. For a detailed description of the 6 datasets used for training, see <insert paper>",
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
        "training-data": "3 different BF datasets were aggregated. For a detailed description of the dataset used for training, see <insert paper>.",
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
        "training-data": "14.8 Mpx of SEM rat images with a resolution of 0.13 um/px. The dataset used for this model is publicly available here: https://github.com/axondeepseg/data_axondeepseg_sem.",
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
        "training-data": "<ADD_DESCRIPTION>",
        "weights": {
            "ensemble": "https://github.com/axondeepseg/default-CARS-model/releases/download/r20240403/model_seg_rat_axon-myelin_cars_ensemble.zip",
            "light": "https://github.com/axondeepseg/default-CARS-model/releases/download/r20240403/model_seg_rat_axon-myelin_cars_light.zip"
        }
    },
    "unmyelinated-sickkids": {}
}

def pretty_print_model(model):
    print(f"Model name:\n\t{model['name']}")
    print(f"Number of classes:\n\t{model['n_classes']}")
    print(f"Overview:\n\t{model['model-info']}")
    print(f"Training data:\n\t{model['training-data']}")