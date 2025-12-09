"""Generate a synthetic milling dataset for quick testing.
Outputs a CSV with columns: spindle_speed, feed_rate, depth_of_cut, coolant (0/1), tool_type (A,B),
surface_roughness, energy_consumption
"""
import numpy as np
import pandas as pd
import argparse

rng = np.random.default_rng(seed=42)

def synth_row(n):
    spindle = rng.uniform(500, 5000, n)
    feed = rng.uniform(0.05, 0.5, n)
    depth = rng.uniform(0.1, 5.0, n)
    coolant = rng.integers(0,2,n)
    tool_type = rng.choice(['A','B'], size=n)
    # synthetic ground truth rules (demo)
    ra = (0.8) + 0.5*feed + 0.25*depth + 0.1*(coolant==0).astype(float)
    ra += rng.normal(scale=0.05, size=n)
    energy = 0.02*spindle + 5.0*depth + 10.0*feed + 3.0*(tool_type=='B').astype(float)
    energy += rng.normal(scale=50.0, size=n)
    df = pd.DataFrame({
        'spindle_speed': spindle,
        'feed_rate': feed,
        'depth_of_cut': depth,
        'coolant': coolant,
        'tool_type': tool_type,
        'surface_roughness': ra,
        'energy_consumption': energy
    })
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='data/milling_sample.csv')
    parser.add_argument('--n', type=int, default=1000)
    args = parser.parse_args()
    df = synth_row(args.n)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")
