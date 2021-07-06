from pathlib import Path


# segmentation files suffix names
axonmyelin_suffix = Path('_seg-axonmyelin.png') # axon + myelin segmentation suffix file name
axon_suffix = Path('_seg-axon.png')             # axon segmentation suffix file name
myelin_suffix = Path('_seg-myelin.png')         # myelin segmentation suffix file name
index_suffix = Path('_index.png')         # image with the index of the axons
axonmyelin_index_suffix = Path('_axonmyelin_index.png') # Colored axonmyelin segmentation + the index image

# morphometrics file suffix name
morph_suffix = Path('axon_morphometrics.xlsx')
