# CNC Milling ML Optimization Project

This repository contains everything needed to reproduce a machine-learning-driven
multi-objective optimization pipeline for CNC milling (predict surface roughness and energy,
then optimize parameters via NSGA-II).

## Quick start
1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
2. Generate synthetic data (optional) or place your CSV in data/:
   ```bash
   python src/generate_synthetic_data.py --out data/milling_sample.csv --n 2000
   ```
3. Preprocess:
   ```bash
   python src/preprocessing.py --input data/milling_sample.csv --out data/processed.npz
   ```
4. Train models:
   ```bash
   mkdir -p results
   python src/train_models.py --input data/processed.npz --out_dir results
   ```
5. Optimize (NSGA-II):
   ```bash
   python src/optimize_nsga2.py --models results/rf_ra.joblib results/rf_en.joblib --out results/pareto_solutions.csv
   ```
6. Evaluate / plot:
   ```bash
   python src/evaluate.py --pareto results/pareto_solutions.csv --models results/rf_ra.joblib results/rf_en.joblib
   ```

See the `src/` directory for all scripts. Replace the synthetic data with your real dataset by ensuring column names match:
['spindle_speed','feed_rate','depth_of_cut','coolant','tool_type','surface_roughness','energy_consumption'].
