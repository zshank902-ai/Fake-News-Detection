import pandas as pd
from sklearn.model_selection import train_test_split
from src.train import train_sota_model
import os

def main():
    # --- 1. DATASET LOADING ---
    print("Loading Datasets...")
    # Using the local dataset we merged earlier
    if os.path.exists('data/merged_multilingual_dataset.csv'):
        df = pd.read_csv('data/merged_multilingual_dataset.csv')
    else:
        print("Dataset not found! Please ensure data/merged_multilingual_dataset.csv exists.")
        return

    # --- 2. TRAIN-TEST SPLIT ---
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    print(f"Data Ready: {len(train_df)} train, {len(val_df)} val")

    # --- 3. RUN SOTA TRAINING ---
    # Parameters optimized for T4/A100 GPUs or local RTX
    train_sota_model(
        train_df=train_df,
        val_df=val_df,
        model_name='xlm-roberta-base',
        batch_size=2,        # Optimized for 4GB VRAM (RTX 3050 Ti)
        grad_accum_steps=32, # Effective batch size = 64 (Better convergence)
        epochs=5,            
        lr=3e-5              
    )

if __name__ == "__main__":
    main()
