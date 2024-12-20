#######################################################################################################################
### This document serves to define visualizations for the evaluation of the VISDRONE dataset
#######################################################################################################################

# Import section
from ultralytics import YOLO, settings
import os
import cv2

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# %matplotlib inline

from pathlib import Path
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from umap import UMAP
import plotly.express as px

                    
################################################################################################################                    
def plot_image_sz_distribution(df):
    """Plot height and width distributions for each dataset split"""
    save_dir = Path("plots")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Prepare distribution data
    distribution_df = df[['test_train_val_split', 'img_ht', 'img_wd']]
    distribution_df = distribution_df.groupby(
        ['test_train_val_split', 'img_ht', 'img_wd']
    ).size().reset_index(name='count')

    # Sort values
    distribution_df = distribution_df.sort_values(
        by=['test_train_val_split', 'img_ht', 'img_wd', 'count'], 
        ascending=[True, True, True, True]
    )

    splits = distribution_df['test_train_val_split'].unique()

    # Create figure with 2 rows (height, width) and splits columns
    fig, axes = plt.subplots(2, len(splits), figsize=(5 * len(splits), 10), sharey='row')

    # Plot each split
    for i, split in enumerate(splits):
        split_df = distribution_df[distribution_df['test_train_val_split'] == split]
        
        # Height distribution (top row)
        height_df = split_df.groupby('img_ht')['count'].sum().reset_index()
        height_df.plot(
            kind='bar', 
            x='img_ht', 
            y='count', 
            ax=axes[0, i], 
            title=f'{split} Height Distribution'
        )
        axes[0, i].set_xlabel('Image Height')
        if i == 0:
            axes[0, i].set_ylabel('Count')
            
        # Width distribution (bottom row)
        width_df = split_df.groupby('img_wd')['count'].sum().reset_index()
        width_df.plot(
            kind='bar', 
            x='img_wd', 
            y='count', 
            ax=axes[1, i], 
            title=f'{split} Width Distribution'
        )
        axes[1, i].set_xlabel('Image Width')
        if i == 0:
            axes[1, i].set_ylabel('Count')

    # Save and show plot
    save_path = save_dir / "image_sz_distribution_by_split.png"
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    
    print(f"Plot saved to {save_path}")

#######################################################################################################
    
def plot_class_distribution_by_set(df):
    '''Plots the class distribution for each dataset split in a bar graph'''
    # Define the order of classes
    class_order = sorted(df['cls'].unique())  # Sort classes in ascending order

    # Get unique splits in the correct order (assumes 'train', 'val', 'test' order)
    train_val_list = list(df['test_train_val_split'].unique())

    # Get normalized class distributions and reindex to ensure correct class order
    df_test = df[df['test_train_val_split'] == train_val_list[0]]['cls'].value_counts(normalize=True).reindex(class_order, fill_value=0)
    df_val = df[df['test_train_val_split'] == train_val_list[1]]['cls'].value_counts(normalize=True).reindex(class_order, fill_value=0)
    df_train = df[df['test_train_val_split'] == train_val_list[2]]['cls'].value_counts(normalize=True).reindex(class_order, fill_value=0)

    # Define the x-axis labels and the width for each bar
    x_labels = [str(item) for item in class_order]
    bar_width = 0.25
    x = np.arange(len(x_labels))

    # Sample counts for each split
    test_counts = list(df_test.values)
    train_counts = list(df_train.values)
    val_counts = list(df_val.values)

    plt.figure(figsize=(20, 6))

    # Plot each bar with an offset in the x-axis position
    bars_test = plt.bar(x - bar_width, test_counts, width=bar_width, label='Test')
    bars_train = plt.bar(x, train_counts, width=bar_width, label='Train')
    bars_val = plt.bar(x + bar_width, val_counts, width=bar_width, label='Val')

    # Add count labels on top of each bar
    for bars in [bars_test, bars_train, bars_val]:
        for bar in bars:
            yval = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                yval,
                f'{int(float(yval) * 100)}%',
                ha='center',
                va='bottom'
            )

    # Customize the chart
    plt.xticks(x, x_labels, rotation=45)
    plt.title('Class Distribution for labels across Datasets')
    plt.xlabel('Class')
    plt.ylabel('Percentage')
    plt.legend()

    # Save the plot
    save_path = Path("plots") / "class_distribution_by_set.png"
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()

    print(f"Plot saved to {save_path}")

##########################################################################################################
    
def plot_cls_representations(data_dir: str, df: pd.DataFrame):
    '''Create a plot of the largest representation of the class as well
    as the context of the bounding box in the image'''
    # Create directory to save plots if it doesn’t exist
    save_dir = Path("images")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Filter for the largest bounding boxes within each target class
    df['area'] = df['width'] * df['height']  # Calculate area of bounding box
    filtered_df = df.loc[df.groupby('cls')['area'].idxmax()]  # Select rows with the max area for each class

    # Iterate over the filtered dataset and save each full image with bounding box and the cropped image
    for index, row in filtered_df.iterrows():
        # Construct the image path
        img_path = Path(data_dir/ row['test_train_val_split'] / 'images' / f"{row['img_id']}.jpg")
        img = cv2.imread(str(img_path))  # Read the image file
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for display

        # Image dimensions
        img_height, img_width = img.shape[:2]

        # Extract YOLO format bounding box information
        x_center = row['x_center'] * img_width
        y_center = row['y_center'] * img_height
        box_width = row['width'] * img_width
        box_height = row['height'] * img_height

        # Calculate the top-left and bottom-right coordinates of the bounding box
        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        # Draw the bounding box on the image
        img_with_box = img.copy()
        img_with_box = cv2.rectangle(img_with_box, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)

        # Crop the image using the bounding box coordinates
        cropped_img = img[y1:y2, x1:x2]

        # Create the plot with full image and cropped image side by side
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Show the full image with bounding box on the left
        axes[0].imshow(img_with_box)
        axes[0].set_title(f"Full Image with Bounding Box\nClass: {row['cls']}")
        axes[0].axis('off')

        # Show the cropped image on the right
        axes[1].imshow(cropped_img)
        axes[1].set_title("Cropped Image")
        axes[1].axis('off')

        plt.tight_layout()  # Adjust layout for clarity
        
        # Save the plot to the 'images' directory with a unique filename
        save_path = save_dir / f"{row['img_id']}_class_{row['cls']}.jpg"
        plt.savefig(save_path, bbox_inches='tight')
        
        # Close the plot to free memory
        plt.close(fig)
        
        print(f"Plot saved to {save_path}")
        
####################################################################################################################

def plot_class_density_by_split(data_fm):
    # Create 'plots' directory if it doesn’t exist
    save_dir = Path("plots")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Get unique classes and splits
    classes = sorted(data_fm['cls'].unique())
    splits = data_fm['test_train_val_split'].unique()
    
    # Set up the number of columns and calculate the number of rows needed
    num_columns = 3
    num_classes = len(classes)
    num_rows = (num_classes + num_columns - 1) // num_columns  # Calculate rows needed

    # Loop through each split to generate a separate plot for each
    for split in splits:
        # Filter the DataFrame for the current split
        split_df = data_fm[data_fm['test_train_val_split'] == split]
        
        # Create a figure with subplots: three columns, multiple rows
        fig, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns * 6, num_rows * 6), squeeze=False)

        # Flatten the axes array for easy indexing
        axes = axes.flatten()

        # Loop through each class and create a subplot for each within the split
        for idx, cls in enumerate(classes):
            # Filter the DataFrame for the current class within the split
            class_df = split_df[split_df['cls'] == cls]
            
            # Group by (x_center, y_center) and count occurrences for density
            density_df = class_df.groupby(['x_center', 'y_center']).size().reset_index(name='count')

            # Create scatter plot for the current class on its subplot
            scatter = axes[idx].scatter(
                density_df['x_center'], 
                density_df['y_center'], 
                s=density_df['count'] * 10,  # Adjust size for better visibility
                c=density_df['count'], 
                cmap='viridis', 
                alpha=0.7
            )

            # Add color bar for density on each plot
            fig.colorbar(scatter, ax=axes[idx], label='Number of Points')
            
            # Set plot details
            axes[idx].set_title(f"Class {cls} Density")
            axes[idx].set_xlabel("X Center (Normalized)")
            axes[idx].set_ylabel("Y Center (Normalized)")
            axes[idx].set_xlim(0, 1)
            axes[idx].set_ylim(0, 1)

        # Hide any extra subplots if the number of classes is not a perfect multiple of columns
        for j in range(idx + 1, len(axes)):
            fig.delaxes(axes[j])

        # Add the main title with the split name
        fig.suptitle(f"Class Density by Position for {split} Dataset", fontsize=18)

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make space for suptitle

        # Save the figure to the 'plots' directory with the split name in the filename
        save_path = save_dir / f"class_density_by_position_{split}.png"
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        plt.close()  # Close the figure to free up memory

        print(f"Multi-plot saved for {split} to {save_path}")     
        

####################################################################################################################
def plot_class_density_by_split_size(data_fm):
    """Plot class size distribution by pixel dimensions for each split"""
    save_dir = Path("plots")
    save_dir.mkdir(parents=True, exist_ok=True)

    classes = sorted(data_fm['cls'].unique())
    splits = data_fm['test_train_val_split'].unique()
    
    num_columns = 3
    num_classes = len(classes)
    num_rows = (num_classes + num_columns - 1) // num_columns

    for split in splits:
        split_df = data_fm[data_fm['test_train_val_split'] == split].copy()  # Create copy
        fig, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns * 6, num_rows * 6), squeeze=False)
        axes = axes.flatten()

        for idx, cls in enumerate(classes):
            # Create explicit copy for class data
            class_df = split_df[split_df['cls'] == cls].copy()
            
            # Calculate dimensions using loc
            class_df.loc[:, 'pixel_height'] = (class_df['height'] * class_df['img_ht']).round()
            class_df.loc[:, 'pixel_width'] = (class_df['width'] * class_df['img_wd']).round()
            
            # Group and count
            density_df = (
                class_df.groupby(['pixel_height', 'pixel_width'])
                .size()
                .reset_index(name='count')
                )

            scatter = axes[idx].scatter(
                density_df['pixel_width'],
                density_df['pixel_height'],
                s=density_df['count'] * 10,
                c=density_df['count'],
                cmap='viridis',
                alpha=0.7
            )

            fig.colorbar(scatter, ax=axes[idx], label='Object Count')
            
            axes[idx].set_title(f"{cls} Size Distribution (pixels)")
            axes[idx].set_xlabel("Width (pixels)")
            axes[idx].set_ylabel("Height (pixels)")
            
            # Set reasonable limits based on data
            max_width = density_df['pixel_width'].max()
            max_height = density_df['pixel_height'].max()
            axes[idx].set_xlim(0, max_width)
            axes[idx].set_ylim(0, max_height)

        # Hide extra subplots
        for j in range(idx + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(f"Class Size Distribution in Pixels for {split} Dataset", fontsize=18)
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        save_path = save_dir / f"class_pixel_size_distribution_{split}.png"
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()
        plt.close()

        print(f"Pixel size distribution plot saved for {split} to {save_path}")
##############################################################################################
##############################################################################################

def visualize_embeddings(
    features_matrix: np.ndarray, 
    cluster_labels: np.ndarray, 
    image_names: list, 
    method: str = 'umap',
    save_path: str = None,
    save_format: str = 'html'
) -> None:
    
    # Dimensionality reduction
    if method == 'umap':
        reducer = UMAP(n_components=2, random_state=42)
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
    else:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        
    reduced_embeddings = reducer.fit_transform(features_matrix)
    
    # Create DataFrame with all info
    df = pd.DataFrame({
            'x': reduced_embeddings[:, 0],
            'y': reduced_embeddings[:, 1],
            'cluster': cluster_labels,
            'image': image_names
        })
        
    fig = px.scatter(
        df,
        x='x', y='y',
        color='cluster',
        hover_data=['image'],
        labels={'color': 'Cluster'},
        title=f'2D Visualization using {method.upper()}',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(width=800, height=600, template='plotly_white')
    
    # Save plot if path provided
    if save_path:
        if save_format == 'html':
            fig.write_html(f"./plots/{save_path}_{method}.html")
        elif save_format == 'png':
            fig.write_image(f"./plots/{save_path}_{method}.png")
            
    fig.show()