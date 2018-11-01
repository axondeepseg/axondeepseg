
import os
import argparse
from argparse import RawTextHelpFormatter
from AxonDeepSeg.visualization.get_masks import *

def main(argv=None):
    ap = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    requiredName = ap.add_argument_group('required arguments')

    requiredName.add_argument('-d', '--dir', required=True, help='Directory containing masks files to perform quality control.')

    args = vars(ap.parse_args(argv))
    masks_dir = str(args['dir'])

    file_list = [file for file in os.listdir(masks_dir) if (file.endswith(('.png','.jpg','.jpeg','.tif','.tiff'))) and (file.startswith('mask'))]
    
    bad_files = dict()

    for file in file_list:
        image_properties = get_image_unique_vals_properties(os.path.join(masks_dir, file))
        if image_properties['num_uniques'] > 3:
            bad_files[file] = image_properties['num_uniques']

    for k, v in sorted(bad_files.items()):
        print(k, v)



# Calling the script
if __name__ == '__main__':
	main()
