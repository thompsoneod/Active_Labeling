import pandas as pd

from pathlib import Path
from ultralytics import YOLO
from scipy.stats import stats

#############################################################################################################
########## LOAD MODEL
#############################################################################################################
def load_model(mdl_path: str = None):
    if mdl_path:
        return YOLO(mdl_path, seed=42)
    else:
        return YOLO('./mdl_wts/baseline_mdl.pt')

#############################################################################################################
########## TRAIN MODEL
#############################################################################################################    
    
def train_model(
    model, 
    data_dir: Path, 
    epochs: int = 2, 
    im_size: int = 640
    ):
    
    results = model.train(
        project='visdrone_al_test', 
        name=f'{data_dir.stem}', 
        data=f'{data_dir}/dataset.yaml', 
        seed=42, 
        imgsz=im_size, 
        epochs=epochs, 
        device=0, 
        batch=16)
    
    maps = results.maps
    
    return results

#############################################################################################################
########## EVALUATE INTERPRET RESULTS
#############################################################################################################

def interpret_results(stats: dict) -> str:
    """Interpret statistical test results"""
    significant = stats['p_value'] < 0.05
    improvement = ((stats['active_mean'] - stats['random_mean']) / stats['random_mean']) * 100
    
    return {
        'significant': significant,
        'improvement': improvement,
        't_stat': stats['t_statistic'],
        'p_value': stats['p_value']
    }

#############################################################################################################
########## EVALUATE MODEL RESULTS
#############################################################################################################

def evaluate_model_results(csv_path: str, column: str='metrics/mAP50(B)') -> dict:
    """
    Analyze results from CSV with two-sample t-test
    """
    try:
        # Load and validate data
        df = pd.read_csv(Path(csv_path))
        if column not in df.columns:
            raise ValueError(f"Required column {column} not found")

        # Split groups
        random_group = df[df['Unnamed: 0'].str.contains('rand', na=False)][column]
        active_group = df[~df['Unnamed: 0'].str.contains('rand', na=False)][column]

        # Perform t-test
        t_stat, p_val = stats.ttest_ind(active_group, random_group)

        return {
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'random_mean': float(random_group.mean()),
            'active_mean': float(active_group.mean()),
            'sample_size': len(df)
        }
        
    except Exception as e:
        print(f"Error analyzing results: {str(e)}")
        return None


