"""Train RF models for Ra and Energy and save them.
"""
import argparse
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/processed.npz')
    parser.add_argument('--out_dir', default='results')
    args = parser.parse_args()
    data = np.load(args.input, allow_pickle=True)
    X_train = data['X_train']
    X_test = data['X_test']
    y_ra_train = data['y_ra_train']
    y_ra_test = data['y_ra_test']
    y_en_train = data['y_en_train']
    y_en_test = data['y_en_test']

    rf_ra = RandomForestRegressor(n_estimators=200, random_state=0)
    rf_ra.fit(X_train, y_ra_train)
    y_ra_pred = rf_ra.predict(X_test)

    rf_en = RandomForestRegressor(n_estimators=200, random_state=1)
    rf_en.fit(X_train, y_en_train)
    y_en_pred = rf_en.predict(X_test)

    def print_metrics(y_true, y_pred, name):
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        print(f"--- {name} ---")
        print('R2:', r2_score(y_true, y_pred))
        print('RMSE:', mean_squared_error(y_true, y_pred))
        print('MAE:', mean_absolute_error(y_true, y_pred))

    print_metrics(y_ra_test, y_ra_pred, 'Surface Roughness (Ra)')
    print_metrics(y_en_test, y_en_pred, 'Energy')

    os.makedirs(args.out_dir, exist_ok=True)
    joblib.dump(rf_ra, f'{args.out_dir}/rf_ra.joblib')
    joblib.dump(rf_en, f'{args.out_dir}/rf_en.joblib')
    print('Saved models to', args.out_dir)
