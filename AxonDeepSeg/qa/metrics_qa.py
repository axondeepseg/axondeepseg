# coding: utf-8

# Scientific modules imports
import numpy as np
import scipy

# Graphs and plots imports
import matplotlib.pyplot as plt
import pandas as pd

import pathlib
from pathlib import Path

import matplotlib.pyplot as plt

import AxonDeepSeg.ads_utils as ads
from AxonDeepSeg.morphometrics.compute_morphometrics import get_axon_morphometrics

mpl_config = Path(pathlib.Path(__file__).parent.resolve() / 'custom_matplotlibrc')
plt.style.use(mpl_config)
plt.rcParams["figure.figsize"] = (9,6)

class MetricsQA:
    def __init__(self, morphometrics_file):
        """
        :param path_pixelsize_file: path of the txt file indicating the pixel size of the sample
        :return: the pixel size value.
        """

        self.file_name = Path(morphometrics_file)

        self.df = pd.read_csv(morphometrics_file)

    def list_metrics(self):
        print('\n'.join(list(self.df.columns.values[3:])))

    def plot(self, metric_name, save_folder = None, quiet = False):
        x = self.df[metric_name].to_numpy()

        fig, (ax1, ax2) = plt.subplots(1, 2, width_ratios=[3, 1])

        # bins='fd' uses the Freedman Diaconis Estimator to find the optimal number of bins
        count, bins, ignored = ax1.hist(x,bins='fd', histtype='bar', ec='black')
        ax1.set(xlabel=metric_name, ylabel='Count') 

        ax2.axis('off')
        ax2.set(title='Stats')

        ax2.annotate('mean: ', xy=(0, 0.9))
        ax2.annotate('median: ', xy=(0, 0.8))
        ax2.annotate('std: ', xy=(0, 0.7))
        ax2.annotate('iqr: ', xy=(0, 0.6))
        ax2.annotate('min: ', xy=(0, 0.5))
        ax2.annotate('max: ', xy=(0, 0.4))
        ax2.annotate('NaNs #:', xy=(0, 0.3))

        ax2.annotate(
            np.format_float_positional(np.nanmean(x), precision=2, trim='0'),
            xy=(1, 0.9)
            )
        ax2.annotate(
            np.format_float_positional(np.nanmedian(x), precision=2, trim='0'),
            xy=(1, 0.8)
            )
        ax2.annotate(
            np.format_float_positional(np.nanstd(x), precision=2, trim='0'),
            xy=(1, 0.7)
            )
        ax2.annotate(
            np.format_float_positional(scipy.stats.iqr(x[~np.isnan(x)]), precision=2, trim='0'),
            xy=(1, 0.6))
        
        ax2.annotate(
            np.format_float_positional(np.nanmin(x), precision=2, trim='0'),
            xy=(1, 0.5)
            )
        ax2.annotate(
            np.format_float_positional(np.nanmax(x), precision=2, trim='0'),
            xy=(1, 0.4)
            )
        ax2.annotate(np.sum(np.isnan(x)), xy=(1, 0.3))
        if quiet == False:
            fig.show()

        if save_folder is not None:
            plt.savefig(Path(Path(save_folder) / metric_name))
        
        mean = np.format_float_positional(np.nanmean(x), precision=2, trim='0')
        std = np.format_float_positional(np.nanstd(x), precision=2, trim='0')

        return (mean, std)
    
    def plot_all(self, save_folder=None, quiet=False):
        metric_list = list(self.df.columns.values[3:])

        for metric in metric_list:
            if self.df[metric].to_numpy().dtype==np.float64:
                self.plot(metric, save_folder, quiet)


    def get_flagged_objects(self, im_axonmyelin_label, save_folder):

        axonmyelin_img = im_axonmyelin_label

        df = self.df

        
        flagged_objects = np.array([])
        flagged_objects = np.append(flagged_objects, df.loc[df['gratio'] >=0.99].index.to_numpy())
        flagged_objects = np.append(flagged_objects, df.loc[df['axon_area (um^2)'] <= min(df['axon_area (um^2)'])*5].index.to_numpy())
        flagged_objects = np.append(flagged_objects, df.loc[df['myelin_area (um^2)'] <= min(df['myelin_area (um^2)'])*5].index.to_numpy())

        flagged_objects = np.unique(flagged_objects)

        mask = np.zeros_like(axonmyelin_img)

        # go through each element in arr
        for id in flagged_objects:
            locations = np.where(np.isclose(axonmyelin_img, id+1))
            mask[locations] = 1
    
        ads.imwrite(Path(save_folder) / 'flagged_objects.png', mask*255)
        return (flagged_objects, mask)
    
    def generate_axon_closeups(self, qa_folder, image, axon_label, myelin_label, im_axonmyelin_label, buffer_pixels=20):
        """Generate closeup images of each axon with overlay using real image data - ONLY current axon"""
        axon_data = []
        
        for axon_id in range(len(self.df)):
            # Get axon properties
            row = self.df.iloc[axon_id]
            axon_diameter = row['axon_diam (um)']
            myelin_thickness = row['myelin_thickness (um)']
            gratio = row['gratio']

            # Total number of axons
            n_axons = len(self.df)

            # Percentile rank (0-100, where 100 = largest)
            diameter_pct = self.df['axon_diam (um)'].fillna(-1).rank(pct=True).iloc[axon_id] * 100
            thickness_pct = self.df['myelin_thickness (um)'].fillna(-1).rank(pct=True).iloc[axon_id] * 100
            gratio_pct = self.df['gratio'].fillna(-1).rank(pct=True).iloc[axon_id] * 100

            # Absolute rank (1 = smallest, n = largest)
            diameter_rank = int(self.df['axon_diam (um)'].fillna(-1).rank(method="min").iloc[axon_id])
            thickness_rank = int(self.df['myelin_thickness (um)'].fillna(-1).rank(method="min").iloc[axon_id])
            gratio_rank = int(self.df['gratio'].rank(method="min").fillna(-1).iloc[axon_id])


            # Find the axon pixels in the label image (axon_id + 1 because labels start at 1)
            current_axon_id = axon_id + 1
            axon_myelin_mask = (im_axonmyelin_label == current_axon_id)
            
            if not np.any(axon_myelin_mask):
                print(f"Warning: No pixels found for axon {axon_id} (ID: {current_axon_id})")
                continue
                
            # Get bounding box coordinates
            y_coords, x_coords = np.where(axon_myelin_mask)
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            
            # Add buffer with boundary checks
            y_min_buf = max(0, y_min - buffer_pixels)
            y_max_buf = min(image.shape[0], y_max + buffer_pixels + 1)
            x_min_buf = max(0, x_min - buffer_pixels)
            x_max_buf = min(image.shape[1], x_max + buffer_pixels + 1)
            
            # Crop all images to the same region
            image_crop = image[y_min_buf:y_max_buf, x_min_buf:x_max_buf]
            axon_crop = axon_label[y_min_buf:y_max_buf, x_min_buf:x_max_buf]  # Boolean: 1=axon, 0=not axon
            myelin_crop = myelin_label[y_min_buf:y_max_buf, x_min_buf:x_max_buf]  # Boolean: 1=myelin, 0=not myelin
            label_crop = im_axonmyelin_label[y_min_buf:y_max_buf, x_min_buf:x_max_buf]  # Integer IDs
            
            # Create masks for ONLY the current axon
            current_region_mask = (label_crop == current_axon_id)
            axon_current_mask = axon_crop.astype(bool) & current_region_mask
            myelin_current_mask = myelin_crop.astype(bool) & current_region_mask
            
            # Save original axon image (without labels)
            original_path = qa_folder / f'axon_{axon_id}_original.png'
            plt.figure(figsize=(8, 8))
            if len(image_crop.shape) == 2:
                plt.imshow(image_crop, cmap='gray')
            else:
                plt.imshow(image_crop)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(original_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            # Create the labeled closeup image with overlay
            labeled_path = qa_folder / f'axon_{axon_id}_labeled.png'
            
            # Create figure with original image and overlay
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Display original image (convert to RGB if grayscale)
            if len(image_crop.shape) == 2:
                ax.imshow(image_crop, cmap='gray', alpha=1.0)
            else:
                ax.imshow(image_crop, alpha=1.0)
            
            # Create overlay with semi-transparent colors - ONLY for current axon
            overlay = np.zeros((image_crop.shape[0], image_crop.shape[1], 4))
            
            # Blue for axon (RGBA: 0,0,1,0.5) - ONLY current axon
            overlay[axon_current_mask] = [0, 0, 1, 0.5]  # Blue with 50% opacity
            
            # Red for myelin (RGBA: 1,0,0,0.5) - ONLY current axon's myelin
            overlay[myelin_current_mask] = [1, 0, 0, 0.5]  # Red with 50% opacity
            
            ax.imshow(overlay)
            
            # Remove axes and add title
            ax.axis('off')
            
            plt.tight_layout()
            plt.savefig(labeled_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            axon_data.append({
                'id': axon_id,
                'diameter': float(axon_diameter),
                'thickness': float(myelin_thickness),
                'gratio': float(gratio),
                'diameterPercentile': f"{diameter_pct:.1f}",
                'thicknessPercentile': f"{thickness_pct:.1f}",
                'gratioPercentile': f"{gratio_pct:.1f}",
                'diameterRank': f"{diameter_rank} of {n_axons}",
                'thicknessRank': f"{thickness_rank} of {n_axons}",
                'gratioRank': f"{gratio_rank} of {n_axons}",
                'imagePath': str(original_path.name),
                'labeledImagePath': str(labeled_path.name)
            })
        
        return axon_data

    def save_seg_overlay(self, image, axon_label, myelin_label, qa_folder):
        """Save overlay of axons and myelin."""
        overlay = np.zeros((image.shape[0], image.shape[1], 3))
        overlay[axon_label == 1] = [255, 0, 0]
        overlay[myelin_label == 1] = [0, 255, 0]
        overlay = overlay.astype(np.uint8)

        plt.figure(figsize=(8, 8))
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(qa_folder / 'base_image.png', dpi=150, bbox_inches='tight')

        plt.imshow(overlay, alpha=0.5)

        plt.savefig(qa_folder / 'segmentation_overlay.png', dpi=150, bbox_inches='tight')
        plt.close()
