from pathlib import Path


# segmentation files suffix names
axonmyelin_suffix = Path('_seg-axonmyelin.png')         # axon + myelin segmentation suffix file name
axon_suffix = Path('_seg-axon.png')                     # axon segmentation suffix file name
myelin_suffix = Path('_seg-myelin.png')                 # myelin segmentation suffix file name
index_suffix = Path('_index.png')                       # image with the index of the axons
axonmyelin_index_suffix = Path('_axonmyelin_index.png') # Colored axonmyelin segmentation + the index image
unmyelinated_suffix = Path('_seg-uaxon.png')            # unmyelinated axon segmentation suffix file name
unmyelinated_index_suffix = Path('_uaxon_index.png')    # Colored unmyelinated axon segmentation + the index image
nnunet_suffix=Path('_seg-nnunet.png')                   # nnunet raw segmentation suffix

side_effect_suffixes = tuple(
    [
        str(s) for s in [
            axonmyelin_suffix, axon_suffix, myelin_suffix, index_suffix, 
            axonmyelin_index_suffix, unmyelinated_suffix, unmyelinated_index_suffix,
            nnunet_suffix
        ]
    ]
)

# morphometrics file suffix name
morph_suffix = Path('axon_morphometrics.xlsx')
unmyelinated_morph_suffix = Path('uaxon_morphometrics.xlsx')
instance_suffix = Path('_instance-map.png')             # Colored instance map of the segmentation

# List of valid image extensions
valid_extensions = [
    ".ome.tif",
    ".ome.tiff",
    ".ome.tf2",
    ".ome.tf8",
    ".ome.btf",
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg"
    ]

