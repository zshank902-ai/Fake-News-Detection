import pandas as pd
from sklearn.model_selection import train_test_split
from src.train import train_sota_model
import os
import time

def main():
    print("Loading Quick Demo Datasets...")
    if os.path.exists('data/merged_multilingual_dataset.csv'):
        # Only take first 100 rows for a quick demo
        df = pd.read_csv('data/merged_multilingual_dataset.csv').head(100)
    else:
        print("Dataset not found!")
        return

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    print(f"Data Ready: {len(train_df)} train, {len(val_df)} val")

    print("Running quick demo training to generate graph...")
    train_sota_model(
        train_df=train_df,
        val_df=val_df,
        model_name='xlm-roberta-base',
        batch_size=2,
        grad_accum_steps=2, 
        epochs=3, # 3 epochs to show a line graph
        lr=3e-5              
    )
    print("Demo training complete. Graph should be in reports/training_graph.png")

if __name__ == "__main__":
    main()
