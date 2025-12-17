import pandas as pd
import os
import argparse
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV with reports")
    parser.add_argument("--output_dir", type=str, default="splits", help="Directory to save split CSVs")
    parser.add_argument("--val_size", type=float, default=0.1, help="Proportion of validation set")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    
    # Filter only rows with reports if needed, but user said "Text GT is only in train data"
    # Assuming the input CSV is the one with reports.
    # We should ensure we don't split empty reports into validation if we want to validate text generation?
    # But for now, let's just split the provided CSV.
    
    # Check if 'report' column exists
    if 'report' in df.columns:
        # Filter out rows with empty reports if we want to be strict, 
        # but maybe the user wants to use all labeled data.
        # Let's just split whatever is in there.
        pass
    
    train_df, val_df = train_test_split(df, test_size=args.val_size, random_state=args.seed)
    
    train_path = os.path.join(args.output_dir, "train_split.csv")
    val_path = os.path.join(args.output_dir, "val_split.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    print(f"Split completed.")
    print(f"Train: {len(train_df)} rows -> {train_path}")
    print(f"Val: {len(val_df)} rows -> {val_path}")

if __name__ == "__main__":
    main()
