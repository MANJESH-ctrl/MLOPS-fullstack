import pandas as pd
from sklearn.model_selection import train_test_split

def simple_sample_data(source_path, target_path, sample_frac):
    """
    Simple function to sample data and save it.
    
    source_path: where your original data.csv is
    target_path: where to save sampled data (e.g., 'locals3/sampled.csv')
    sample_frac: what percent to sample (0.1 = 10%)
    """
    # Load data
    df = pd.read_csv(source_path)
    
    # Sample it (stratified by target column if exists)
    if 'Response' in df.columns:
        X = df.drop('Response', axis=1)
        y = df['Response']
        X_sample, _, y_sample, _ = train_test_split(
            X, y, 
            train_size=sample_frac, 
            random_state=42, 
            stratify=y
        )
        df_sample = pd.concat([X_sample, y_sample], axis=1)
    else:
        df_sample = df.sample(frac=sample_frac, random_state=42)
    
    # Save it
    df_sample.to_csv(target_path, index=False)
    
    print(f"Original: {len(df):,} rows")
    print(f"Sampled: {len(df_sample):,} rows ({sample_frac*100}%)")
    print(f"Saved to: {target_path}")
    
    return df_sample



df_sample = simple_sample_data('C:\\projects\\MLOPS\\references\\data.csv', 'C:\\projects\\MLOPS\\locals3\\sampled.csv', 0.01)