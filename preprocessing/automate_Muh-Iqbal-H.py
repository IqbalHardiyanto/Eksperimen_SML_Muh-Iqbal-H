import pandas as pd
import numpy as np
import argparse
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def preprocess_and_split_data(input_path, output_dir=None, test_size=0.3, val_size=0.5, random_state=42):
    """
    Melakukan preprocessing data dan splitting secara otomatis dari file CSV
    
    Parameters:
    input_path (str): Path ke file CSV
    output_dir (str): Direktori untuk menyimpan data hasil preprocessing
    test_size (float): Proporsi data uji (default: 0.3)
    val_size (float): Proporsi data validasi dari data uji (default: 0.5)
    random_state (int): Seed untuk reproduktibilitas (default: 42)
    
    Returns:
    tuple: (df_preprocessed, split_info) jika output_dir=None
    """
    
    df = pd.read_csv(input_path)
    
    df_clean = df.dropna()
    
    df_clean = df_clean.drop_duplicates()
    
    def remove_outliers_iqr(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    numerical_features = df_clean.select_dtypes(include=np.number).columns
    for col in numerical_features:
        df_clean = remove_outliers_iqr(df_clean, col)
    
    label_encoder = LabelEncoder()
    df_clean['label_encoded'] = label_encoder.fit_transform(df_clean['label'])
    
    X = df_clean.drop(['label', 'label_encoded'], axis=1)
    y = df_clean['label_encoded']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state, stratify=y_temp
    )
    
    # Gabungkan semua data dan tambahkan kolom 'dataset_type'
    train_df = pd.DataFrame(X_train, columns=X.columns).assign(
        label=y_train.values,
        label_encoded=y_train.values,
        dataset_type='train'
    )
    
    val_df = pd.DataFrame(X_val, columns=X.columns).assign(
        label=label_encoder.inverse_transform(y_val),
        label_encoded=y_val.values,
        dataset_type='val'
    )
    
    test_df = pd.DataFrame(X_test, columns=X.columns).assign(
        label=label_encoder.inverse_transform(y_test),
        label_encoded=y_test.values,
        dataset_type='test'
    )
    
    df_preprocessed = pd.concat([train_df, val_df, test_df], axis=0)
    
    # output_dir, simpan hasil preprocessing
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'df_soil_preprocessed.csv')
        df_preprocessed.to_csv(output_path, index=False)
        print(f"Data hasil preprocessing disimpan di: {output_path}")
        
        # Simpan juga metadata
        split_info = {
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
            'total_size': len(df_preprocessed),
            'random_state': random_state
        }
        pd.DataFrame.from_dict(split_info, orient='index').to_csv(
            os.path.join(output_dir, 'split_info.csv'), header=False)
    
    return df_preprocessed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Preprocessing and Splitting')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output_dir', type=str, help='Output directory for processed data')
    parser.add_argument('--test_size', type=float, default=0.3, help='Test set proportion')
    parser.add_argument('--val_size', type=float, default=0.5, help='Validation set proportion from test set')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    preprocess_and_split_data(
        args.input,
        args.output_dir,
        args.test_size,
        args.val_size,
        args.random_state
    )
    print("Preprocessing dan Data Splitting selesai!")