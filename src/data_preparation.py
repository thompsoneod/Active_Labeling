import shutil
import getpass

import pandas as pd
import numpy as np
import src.preprocess as pp
import tools.fiftyone as fo_tools

from pathlib import Path
from typing import List, Tuple


#####################################################################################################################
#################### Create experiment folder and copy items with simlinks  ############################
def create_folders(target_dir: str, splits:list=['train', 'val']) -> None:
    for dir in splits:
        for sub_dir in ['images', 'labels']:
            directory = Path(f'{target_dir}/{dir}/{sub_dir}')
            directory.mkdir(parents=True, exist_ok=True)
            # print(f'Created {target_dir / dir / sub_dir}')
            print(f'Created {directory}')
            
def create_simlinks(src_file, dest_dir) -> None:
    if not Path(dest_dir).exists():
        print(f'{dest_dir} does not exists')
    
    img_src = Path(src_file)
    lbl_src = Path(str(img_src.with_suffix('.txt')).replace('images', 'labels'))
    
    if img_src.exists() and lbl_src.exists():
        dest_img = Path(dest_dir) / 'images' / img_src.name
        dest_lbl = Path(dest_dir) / 'labels' / lbl_src.name
        dest_img.symlink_to(img_src)
        dest_lbl.symlink_to(lbl_src)
        
        
#####################################################################################################################
#################### Created to test the function with min and max confidence threshold ############################

def select_low_confidence_images(
    df: pd.DataFrame,
    confidence_threshold_min: float = 0.3,
    confidence_threshold_max: float = 0.6,
    uniqueness_threshold: float = 0.6
    ) -> List[str]:
    """
    Select images with confidence scores within range from each cluster.
    
    Args:
        df: DataFrame with confidence_mean, uniqueness, and tags columns
        confidence_threshold_min: Minimum confidence score threshold
        confidence_threshold_max: Maximum confidence score threshold
        uniqueness_threshold: Uniqueness score threshold
    
    Returns:
        List of selected image names
    """
    # Filter based on thresholds
    filtered_df = df[
        (df['confidence_mean'].between(confidence_threshold_min, confidence_threshold_max)) & 
        (df['uniqueness'] > uniqueness_threshold)
    ]
    
    selected_images = []
    cluster_list = sorted(filtered_df['cluster'].unique())
    
    for clust in cluster_list:
        # Get cluster subset
        clust_df = filtered_df[filtered_df['cluster'] == clust]
        # most_likely_images = (clust_df.groupby('image','filepath')['confidence_mean']
        #                     .mean()
        #                     .sort_values(ascending=True))
        most_likely_images = (clust_df.groupby('filepath')['confidence_mean']
                            .mean()
                            .sort_values(ascending=True))
        
        # Select images based on cluster size
        if len(most_likely_images) <= 15:
            selected = most_likely_images.index.tolist()
        else:
            n_select = int(len(most_likely_images) * 0.15)
            selected = most_likely_images.index[:n_select].tolist()
            
        selected_images.extend(selected)
    
    print(f"Selected {len(selected_images)} images from {len(cluster_list)} clusters")
    return selected_images

####################################################################################
##################### Created to test the function with classes ####################


def select_low_confidence_images_by_cls(
    df: pd.DataFrame,
    classes: List[str],
    confidence_threshold_min: float = 0.3,
    confidence_threshold_max: float = 0.6,
    uniqueness_threshold: float = 0.5
) -> List[str]:
    """
    Select images with confidence scores within range for specified classes.
    
    Args:
        df: DataFrame with confidence_mean, uniqueness, and tags columns
        classes: List of classes to consider for confidence filtering
        confidence_threshold_min: Minimum confidence score threshold
        confidence_threshold_max: Maximum confidence score threshold
        uniqueness_threshold: Uniqueness score threshold
    
    Returns:
        List of selected image names
    """
    # Filter for specified classes
    class_df = df[df['label'].isin(classes)]
    
    # Filter based on thresholds
    filtered_df = class_df[
        (class_df['confidence_mean'].between(confidence_threshold_min, confidence_threshold_max)) & 
        (class_df['uniqueness'] > uniqueness_threshold)
    ]
    
    selected_images = []
    cluster_list = sorted(filtered_df['tags'].unique())
    
    for clust in cluster_list:
        clust_df = filtered_df[filtered_df['tags'] == clust]
        most_likely_images = (clust_df.groupby('image')['confidence_mean']
                            .mean()
                            .sort_values(ascending=True))
        
        if len(most_likely_images) < 10:
            selected = most_likely_images.index.tolist()
        else:
            n_select = int(len(most_likely_images) * 0.1)
            selected = most_likely_images.index[:n_select].tolist()
            
        selected_images.extend(selected)
    
    print(f"Selected {len(selected_images)} images from {len(cluster_list)} clusters")
    print(f"Classes considered: {classes}")
    return selected_images

####################################################################################
##################### Create Yaml for training ####################


def create_yaml(
    img_dir: str,
    # output_path: str,
    # model: dict[str],
) -> None:
    """Create dataset YAML file with comments
    img_dir: Directory containing images
    output_path: Path to save YAML file
    cls_names: List of class names
    """
    # Create yaml content with comments as a string
    # cls_names = model.names
    df = pp.create_df_from_yolo(img_dir)
    
    trn_df = df[df['test_train_val_split']=='train']['cls'].value_counts()
    val_df = df[df['test_train_val_split']=='val']['cls'].value_counts()
    yaml_content = f"""# YOLOv8 Dataset Configuration
created on: {pd.Timestamp.now()}
created for: {getpass.getuser()}

# Base path to dataset
path: {img_dir}

# Image directories
train: train/images  # Training image folder    {df[df['test_train_val_split']=='train']['img_id'].unique().shape[0]} images
val: val/images      # Validation image folder  {df[df['test_train_val_split']=='val']['img_id'].unique().shape[0]} images

# Class names and indices (train/val split)
names:
  0: pedestrian        # Individual walking         {trn_df['0']} / {val_df['0']} instances
  1: people            # Groups of people           {trn_df['1']} / {val_df['1']} instances
  2: bicycle           # Two-wheeled pedal vehicle  {trn_df['2']} / {val_df['2']} instances
  3: car               # Personal automobile        {trn_df['3']} / {val_df['3']} instances
  4: van               # Delivery/cargo van         {trn_df['4']} / {val_df['4']} instances
  5: truck             # Large cargo vehicle        {trn_df['5']} / {val_df['5']} instances
  6: tricycle          # Three-wheeled vehicle      {trn_df['6']} / {val_df['6']} instances
  7: awning-tricycle   # Covered three-wheeler      {trn_df['7']} / {val_df['7']} instances
  8: bus               # Public transport vehicle   {trn_df['8']} / {val_df['8']} instances
  9: motor             # Motorcycle/scooter         {trn_df['9']} / {val_df['9']} instances
"""
    
    # Write content directly to file
    output_path = img_dir
    with open(f'{output_path}/dataset.yaml', 'w') as f:
        f.write(yaml_content)
        
    print(f"Saved YAML file to {output_path}")
    
    
####################################################################################
##################### Create test_directory for training ####################
import numpy as np
import json

def setup_test_dir(
    fo_json: str,
    method: str='confidence',
    data_dir: str='./dataset/img_proc',
    train_split: str='test-dev',
    val_split: str='VisDrone2019-DET-val',
    test_split: str=None
    ):
    '''
    This function will create a test directory with the following structure:
    train_dir
        - images
        - labels
    val_dir
        - images
        - labels
    test_dir
        - images
        - labels
    dataset.yaml
    
    This function takes a the fiftyone json file and creates the directory structure along with the yaml file 
    for the different methods of training.
    
    Args:
        fo_json: JSON - the fiftyone json file
        data_dir: str - the directory to save the test data to
        train_split: str - the split to use for the training data
        val_split: str - the split to use for the validation data
        test_split: str - the split to use for the test data
        
    '''
    
    # read in the fiftyone json file
    fo_data, avg_fo_data = fo_tools.read_fiftyone_json(fo_json)
    # create folders for the test data
    create_folders(data_dir)
    
    # create img list for the val and train data
    if train_split is not None and fo_data['filepath'].str.contains(train_split).values.any():
        train_imgs = fo_data['filepath'][fo_data['filepath'].str.contains(train_split)].unique()
        print(f'train_imgs in dataset: {len(train_imgs)}')
        if method == 'confidence':
            train_selected_imgs = select_low_confidence_images(avg_fo_data[avg_fo_data['filepath'].isin(train_imgs)])
        else:
            train_selected_imgs = np.random.choice(train_imgs, int(len(train_imgs) * 0.1), replace=False)
            print(f'Randomly selected {len(train_selected_imgs)} images')
        for img in train_selected_imgs:
            create_simlinks(img, data_dir + '/train/')  
    else:
        print(f'No train_imgs available')
        
    if val_split and fo_data['filepath'].str.contains(val_split).any():
        val_imgs = fo_data['filepath'][fo_data['filepath'].str.contains(val_split)].unique()
        print(f'val_imgs in dataset: {len(val_imgs)}')
        for img in val_imgs:
            create_simlinks(img, data_dir + '/val/')
    else:
        print(f'No val_imgs available')
        
    if test_split and fo_data['filepath'].str.contains(test_split).any():
        test_imgs = fo_data['filepath'][fo_data['filepath'].str.contains(test_split)].unique()
        print(f'test_imgs in dataset: {len(test_imgs)}')
        for img in test_imgs:
            create_simlinks(img, data_dir + '/test/')
    else:
        print(f'No test_imgs available')
            
    # create yaml file for the dataset
    create_yaml(data_dir)
    
#### Example of how to use the setup_test_dir function ####
# fo_json = '/Path/to/fiftyone/json/file.json'
# data_dir = '/Path/to/data'
# train_split = 'split_name_train' 
# val_split = 'split_name_val' 
# test_split = 'split_name_test' *Optional
# setup_test_dir(fo_json, data_dir, train_split, val_split, test_split)
