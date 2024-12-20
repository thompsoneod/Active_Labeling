import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.utils.yolo as foy
import pandas as pd
import numpy as np
import json
import pandas as pd

from pathlib import Path
from typing import Tuple, Dict, Any

from pathlib import Path


# Load the dataset
def fo_load_data(data_dir: Path):
    dataset = fo.Dataset.from_dir(
        data_dir,
        dataset_type=fo.types.FiftyOneDataset)
    return dataset
# Load the labels

# compute similarity/uniqueness
def compute_similarity_uniqueness(
    dataset: fo.Dataset,
    method_vis: str=None,
    embeddings: np.array=None) -> None:
    '''Compute metadata, uniqueness, similarity, and visualization for a FiftyOne dataset'''
    dataset.compute_metadata(progress=True)
    fob.compute_uniqueness(dataset, progress=True)
    fob.compute_similarity(dataset, progress=True, brain_key="similarity")
    if embeddings is not None:
        clstr_df = None
        fob.compute_visualization(
            dataset,
            embeddings=embeddings,
            method="umap",
            output_dir="./results/umap_vis",
            labels=clstr_df["cluster"].values,
            label_field="cluster",
            overwrite=True,
            progress=True,
            brain_key="umap_vis",
        )
    else:
        for method_vis in ["umap", "tsne", "pca"]:
            fob.compute_visualization(
                dataset,
                embeddings=f'resnet101_embed_{method_vis}',
                method=method_vis,
                output_dir=f"./results/{method_vis}_vis",
                overwrite=True,
                progress=True,
                brain_key=f"{method_vis}_vis",
            )

# add cluster tags to dataset
def add_cluster_tags_to_dataset(
    dataset: fo.Dataset,
    cluster_df: pd.DataFrame
    ) -> fo.Dataset:
    """Add cluster tags to existing FiftyOne dataset"""
    
    # Create mapping of image IDs to cluster labels
    cluster_map = dict(zip(cluster_df['img_id'], cluster_df['cluster']))
    
    # Add cluster tags to samples
    with fo.ProgressBar() as pb:
        for sample in pb(dataset):
            img_id = Path(sample.filepath).stem
            if img_id in cluster_map:
                sample.tags = sample.tags or []
                sample.tags.append(f"cluster_{cluster_map[img_id]}")
                sample.save()
    
    print(f"Added cluster tags to {len(dataset)} samples")
    return dataset

# compute predictions and add them to the dataset
# def compute_preds(wts_file: Path, dataset: fo.Dataset.from_dir, pred_field: str = 'predictions', mdl_conf: float=0.25):
#     mdl = model_obj(wts_file)
#     results = dataset.apply_model(mdl, label_field=pred_field, confidence=mdl_conf)
#     results = results.evaluate_detections(
#         'predictions',
#         'ground_truth',
#         compute_mAP=True)
    

#######################################################################################################################
# Function to extract embeddings and perform clustering on a dataset
#######################################################################################################################

def add_cluster_tags_to_dataset(
    dataset: fo.Dataset,
    cluster_df: pd.DataFrame
    ) -> fo.Dataset:
    """Add cluster tags to existing FiftyOne dataset"""
    
    # Create mapping of image IDs to cluster labels
    cluster_map = dict(zip(cluster_df['img_id'], cluster_df['cluster']))
    
    # Add cluster tags to samples
    with fo.ProgressBar() as pb:
        for sample in pb(dataset):
            img_id = Path(sample.filepath).stem
            if img_id in cluster_map:
                sample.tags = sample.tags or []
                sample.tags.append(f"cluster_{cluster_map[img_id]}")
                sample.save()
    
    print(f"Added cluster tags to {len(dataset)} samples")
    return dataset

#######################################################################################################################
# Function to read json output of fiftyone and convert to pandas dataframe
#######################################################################################################################

def read_fiftyone_json(json_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read FiftyOne JSON export and convert to structured DataFrames.
    
    Args:
        json_file: Path to FiftyOne JSON export file
        
    Returns:
        Tuple containing:
        - DataFrame with all detections and their properties
        - DataFrame with averaged confidence scores per image/label
        
    Raises:
        FileNotFoundError: If JSON file doesn't exist
        KeyError: If required fields are missing from JSON
    """
    
    # Validate input file
    if not Path(json_file).exists():
        raise FileNotFoundError(f"JSON file not found: {json_file}")
        
    # Load JSON data
    with open(json_file, 'r') as f:
        samples = json.load(f)
    
    if 'samples' not in samples:
        raise KeyError("JSON file missing 'samples' field")

    # Process samples into dictionary
    dataset_dict: Dict[str, Any] = {}
    ind = 0
    for i, sample in enumerate(samples['samples']):
        image_id = Path(sample['filepath']).stem
        
        if not ('baseline_mdl_preds' in sample and 'detections' in sample['baseline_mdl_preds']):
            continue
            
        for j, detection in enumerate(sample['baseline_mdl_preds']['detections']):
            dataset_dict[f"{ind}"] = {
                'image': image_id,
                'im_width': sample.get('metadata')['width'],
                'im_height': sample.get('metadata')['height'],
                'label': detection.get('label'),
                'bbox': detection.get('bounding_box'),
                'confidence': detection.get('confidence'),
                'uniqueness': sample.get('uniqueness', 0.0),  # Default to 0.0 if missing
                'tags': sample.get('tags')[0],
                'cluster': sample.get('tags')[-1],
                'filepath': sample.get('filepath')
            }
            ind += 1

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(dataset_dict, orient='index')
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Process bounding boxes
    bbox_cols = ['x_center', 'y_center', 'width', 'height']
    bbox_df = pd.DataFrame(df['bbox'].tolist(), columns=bbox_cols, index=df.index)
    df = df.drop(columns=['bbox']).join(bbox_df)

    # Calculate averages
    avg_df = (
        df.drop(columns=bbox_cols)
        .groupby(['image', 'label', 'uniqueness','tags', 'cluster', 'filepath'], as_index=False)
        .agg(confidence_mean=('confidence', 'mean'))
        )

    # Add label counts
    label_counts = (
        df.groupby(['image', 'label'])
        .size()
        .reset_index(name='label_count')
        )
    
    
    avg_df = avg_df.merge(label_counts, on=['image', 'label'], how='left')
    
    return df, avg_df


