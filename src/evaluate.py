import argparse
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pareto', default='results/pareto_solutions.csv')
    parser.add_argument('--models', nargs=2, default=['results/rf_ra.joblib','results/rf_en.joblib'])
    args = parser.parse_args()
    pareto = pd.read_csv(args.pareto)
    print('Pareto head:')
    print(pareto.head())
    os.makedirs('results/figures', exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.scatter(pareto['energy_consumption'], pareto['surface_roughness'])
    plt.xlabel('Energy')
    plt.ylabel('Surface Roughness')
    plt.title('Pareto front (predicted)')
    plt.tight_layout()
    plt.savefig('results/figures/pareto.png')
    print('Saved results/figures/pareto.png')
