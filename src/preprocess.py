#######################################################################################################################
### This is to define the core functions for the evaluation of the VISDRONE dataset
#######################################################################################################################
import os
import cv2
import av
import cv2

import pandas as pd
import numpy as np

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, Dict, List
from sklearn.cluster import KMeans
from ultralytics import YOLO


#######################################################################################################################
### Function to convert VisDrone annotations to YOLO format
#######################################################################################################################

def visdrone2yolo(dir: str) -> None:
    '''
    dir : path to the directory containing the VisDrone dataset
    '''
    
    def convert_box(size, box):
        # Convert VisDrone box to YOLO xywh box
        dw = 1. / size[0]
        dh = 1. / size[1]
        return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh
    
    for d in 'VisDrone2019-DET-train', 'VisDrone2019-DET-val', 'VisDrone2019-DET-test-dev':
        dir_d = Path(dir) / d
        (dir_d / 'labels').mkdir(parents=True, exist_ok=True)  # make labels directory
        pbar = tqdm((dir_d / 'annotations').glob('*.txt'), desc=f'Converting {dir_d.name}')
        for f in pbar:
            img_size = Image.open((dir_d / 'images' / f.name).with_suffix('.jpg')).size
            lines = []
            with open(f, 'r') as file:  # read annotation.txt
                for row in [x.split(',') for x in file.read().strip().splitlines()]:
                    if row[4] == '0':  # VisDrone 'ignored regions' class 0
                        continue
                    cls = int(row[5]) - 1
                    box = convert_box(img_size, tuple(map(int, row[:4])))
                    lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")
                    with open(str(f).replace(f'{os.sep}annotations{os.sep}', f'{os.sep}labels{os.sep}'), 'w') as fl:
                        fl.writelines(lines)  # write label.txt
                        
#######################################################################################################################
# Function to create a dataframe from a directory with yolo format labels
#######################################################################################################################

def create_df_from_yolo(dataset_dir: str) -> pd.DataFrame:
    '''
    dataset_dir : path to the directory containing the dataset with yolo format labels
    cls_dict : dictionary containing the class names and their corresponding class numbers
        you can use model.names if you're using an ultralytics model
    '''
    dataset_dir = Path(dataset_dir)
    dataset_dict = {}
    lbl_count = 1
    for file in dataset_dir.rglob('*.jpg'):
        file_name = file.stem
        if file.exists():
            im = cv2.imread(str(file))
            im_ht, im_dw, channels = im.shape
            
            lbl_file = file.parents[1] / 'labels' / (file_name + '.txt')
            if lbl_file.exists():
                with open(lbl_file, 'r') as f:
                    lines = f.readlines()
                    
                    # print(Path(file.parts[-3], file.parts[-2], file.parts[-1]))
                    for line in lines:
                        line = line.strip().split(' ')
                        dataset_dict[lbl_count] = {
                            'img_id': file_name,
                            'img_ht': im_ht,
                            'img_wd': im_dw,
                            'channels': channels,
                            'img_type': file.suffix, 
                            'test_train_val_split': file.parents[1].name, 
                            'cls': line[0],  
                            'x_center': float(line[1]), 
                            'y_center': float(line[2]), 
                            'width': float(line[3]), 
                            'height': float(line[4]),
                            'bbox_sz_W': round(float(line[3])*im_dw,2),
                            'bbox_sz_H': round(float(line[4])*im_ht,2),
                            }
                        lbl_count += 1
    df = pd.DataFrame(dataset_dict).T
    df.to_csv('dataset.csv', index=False)
    return df

#######################################################################################################################
# Function to cut video into frames
#######################################################################################################################
def extract_key_frames(file_path: Path | str, save_path: Path | str) -> Path:
    """
    Extract key frames from video file.
    
    Args:
        file_path: Path to video file
        save_path: Base path to save extracted frames
        
    Returns:
        Path to folder containing extracted frames
    
    Raises:
        FileNotFoundError: If video file doesn't exist
    """
    # Convert to Path objects
    file_path = Path(file_path)
    save_path = Path(save_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Video file not found: {file_path}")
    
    # Create save directory
    name = file_path.stem
    save_fldr = save_path / name / 'key_frames'
    save_fldr.mkdir(parents=True, exist_ok=True)
    
    def count_key_frames(video_path: Path) -> int:
        count = 0
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            stream.codec_context.skip_frame = "NONKEY"
            for _ in container.decode(stream):
                count += 1
        return count
    
    key_frame_count = count_key_frames(file_path)
    
    with av.open(str(file_path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.skip_frame = "NONKEY"
        
        for i, frame in enumerate(tqdm(
            container.decode(stream), 
            total=key_frame_count, 
            desc="Extracting key frames"
        )):
            frame_path = save_fldr / f"{name}_{i}.jpg"
            frame.to_image().save(frame_path)
    
    print(f"Key frames saved to {save_fldr}")
    return save_fldr

#######################################################################################################################
# Function to extract embeddings and perform clustering on a dataset
#######################################################################################################################

def extract_embeddings(
    model: YOLO,
    image_dir: str,
    layer_idx: int = 8,
    img_size: int = 640,
    n_clusters: int = 10
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Extract embeddings from YOLO model and perform clustering.
    
    Args:
        model: YOLO model instance
        image_dir: Directory containing images
        layer_idx: Index of layer to extract features from
        img_size: Image size for model input
        n_clusters: Number of clusters for K-means
        
    Returns:
        Tuple containing:
        - DataFrame with image IDs and cluster assignments
        - Array of embeddings
    """
    # Validate inputs
    image_path = Path(image_dir)
    if not image_path.exists():
        raise ValueError(f"Image directory not found: {image_dir}")
        
    # Initialize containers
    embeddings_dict: Dict[str, np.ndarray] = {}
    all_embeddings: List[np.ndarray] = []
    image_paths: List[Path] = []
    
    def hook_fn(module, input, output):
        embeddings_dict['features'] = output.detach().cpu().numpy()
    
    # Use context manager for hook
    hook = model.model.model[layer_idx].register_forward_hook(hook_fn)
    try:
        # Process images
        for img_file in image_path.glob('*.[jp][pn][g]'):
            try:
                # Get embeddings
                _ = model(str(img_file), imgsz=(img_size, img_size))
                features = embeddings_dict.get('features')
                
                if features is not None:
                    flattened_features = features.reshape(-1)
                    all_embeddings.append(flattened_features)
                    image_paths.append(img_file)
                    
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
                continue
                
        if not all_embeddings:
            raise ValueError("No valid embeddings extracted")
            
        # Normalize embeddings to same size
        max_length = max(len(embedding) for embedding in all_embeddings)
        padded_embeddings = [
            np.pad(embedding, (0, max_length - len(embedding)), 'constant')
            for embedding in all_embeddings
        ]
        
        # Create feature matrix and normalize
        features_matrix = np.vstack(padded_embeddings)
        features_matrix = (features_matrix - features_matrix.mean(axis=0)) / features_matrix.std(axis=0)
        
        # Perform clustering
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init='auto'
        )
        cluster_labels = kmeans.fit_predict(features_matrix)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'img_id': [p.stem for p in image_paths],
            'cluster': cluster_labels
        })
        
        return results_df, features_matrix
        
    finally:
        hook.remove()
        
