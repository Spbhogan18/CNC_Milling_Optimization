"""Preprocess CSV into numpy arrays and save a compressed .npz for training.
Also performs encoding and scaling.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/milling_sample.csv')
    parser.add_argument('--out', default='data/processed.npz')
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    df = df.dropna()
    tool_dummies = pd.get_dummies(df['tool_type'], prefix='tool')
    X = pd.concat([df[['spindle_speed','feed_rate','depth_of_cut','coolant']], tool_dummies], axis=1)
    y_ra = df['surface_roughness'].values
    y_en = df['energy_consumption'].values
    X_train, X_test, y_ra_train, y_ra_test, y_en_train, y_en_test = train_test_split(
        X, y_ra, y_en, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    os.makedirs('results', exist_ok=True)
    np.savez_compressed(args.out,
                        X_train=X_train_s, X_test=X_test_s,
                        y_ra_train=y_ra_train, y_ra_test=y_ra_test,
                        y_en_train=y_en_train, y_en_test=y_en_test,
                        feature_names=list(X.columns))
    joblib.dump(scaler, 'results/scaler.joblib')
    print('Saved processed data to', args.out)
