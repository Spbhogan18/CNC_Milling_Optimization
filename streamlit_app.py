# Streamlit UI for CNC Milling Optimization
# Save this file as: streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from deap import base, creator, tools, algorithms
import random
import matplotlib.pyplot as plt

# ------------------------------
# LOAD MODELS & SCALER
# ------------------------------
@st.cache_resource
def load_models():
    rf_ra = joblib.load('results/rf_ra.joblib')
    rf_en = joblib.load('results/rf_en.joblib')
    scaler = joblib.load('results/scaler.joblib')
    return rf_ra, rf_en, scaler

rf_ra, rf_en, scaler = load_models()

# Parameter bounds
BOUNDS = {
    'spindle_speed': (500, 5000),
    'feed_rate': (0.05, 0.5),
    'depth_of_cut': (0.1, 5.0),
}

# ------------------------------
# STREAMLIT UI CONFIG
# ------------------------------
st.set_page_config(page_title="CNC Milling Optimization", layout="wide")
st.title("🛠️ CNC Milling Optimization Dashboard")
st.write("Use machine learning and NSGA-II optimization to explore optimal machining parameters.")

# ------------------------------
# SIDEBAR — USER INPUTS
# ------------------------------
st.sidebar.header("Manual Prediction Inputs")
spindle = st.sidebar.slider("Spindle Speed (RPM)", 500, 5000, 2000)
feed = st.sidebar.slider("Feed Rate (mm/rev)", 0.05, 0.5, 0.20)
depth = st.sidebar.slider("Depth of Cut (mm)", 0.1, 5.0, 2.0)
coolant = st.sidebar.selectbox("Coolant", [0, 1])
tool_type = st.sidebar.selectbox("Tool Type", ["A", "B"])  

# ------------------------------
# MAKE SINGLE PREDICTION
# ------------------------------
if st.sidebar.button("Predict Surface Roughness & Energy"):
    toolA = 1 if tool_type == "A" else 0
    toolB = 1 - toolA
    X = np.array([[spindle, feed, depth, coolant, toolA, toolB]])
    Xs = scaler.transform(X)

    pred_ra = rf_ra.predict(Xs)[0]
    pred_en = rf_en.predict(Xs)[0]

    st.subheader("📌 Prediction Results")
    c1, c2 = st.columns(2)
    c1.metric("Predicted Surface Roughness (Ra)", f"{pred_ra:.4f}")
    c2.metric("Predicted Energy Consumption", f"{pred_en:.2f}")

# ------------------------------
# OPTIMIZATION SECTION
# ------------------------------
st.header("⚙️ Run NSGA-II Optimization")
st.write("This will generate a Pareto-optimal set of machining parameters that balance energy & surface roughness.")
pop_size = st.slider("Population Size", 20, 200, 80)
generations = st.slider("Number of Generations", 10, 200, 60)
run_opt = st.button("🚀 Run Optimization")

RNG = random.Random(42)

# NSGA helper
def decode_individual(ind):
    return {
        'spindle_speed': ind[0],
        'feed_rate': ind[1],
        'depth_of_cut': ind[2],
        'coolant': int(round(ind[3])),
        'tool_B': int(round(ind[4]))
    }

# ------------------------------
# EXECUTE NSGA-II
# ------------------------------
if run_opt:
    st.write("Running optimization… this may take ~1–3 minutes.")

    creator.create('FitnessMin', base.Fitness, weights=(-1.0, -1.0))
    creator.create('Individual', list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()

    toolbox.register('spindle', RNG.uniform, *BOUNDS['spindle_speed'])
    toolbox.register('feed', RNG.uniform, *BOUNDS['feed_rate'])
    toolbox.register('depth', RNG.uniform, *BOUNDS['depth_of_cut'])
    toolbox.register('coolant', RNG.randint, 0, 1)
    toolbox.register('toolB', RNG.randint, 0, 1)

    toolbox.register('individual', tools.initCycle, creator.Individual,
                     (toolbox.spindle, toolbox.feed, toolbox.depth, toolbox.coolant, toolbox.toolB), n=1)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    def eval_ind(ind):
        dec = decode_individual(ind)
        toolA = 1 - dec['tool_B']
        X = np.array([[dec['spindle_speed'], dec['feed_rate'], dec['depth_of_cut'], dec['coolant'], toolA, dec['tool_B']]])
        Xs = scaler.transform(X)
        return float(rf_ra.predict(Xs)[0]), float(rf_en.predict(Xs)[0])

    toolbox.register('evaluate', eval_ind)
    toolbox.register('mate', tools.cxBlend, alpha=0.5)
    toolbox.register('mutate', tools.mutPolynomialBounded, eta=20.0,
                     low=[BOUNDS['spindle_speed'][0], BOUNDS['feed_rate'][0], BOUNDS['depth_of_cut'][0], 0, 0],
                     up=[BOUNDS['spindle_speed'][1], BOUNDS['feed_rate'][1], BOUNDS['depth_of_cut'][1], 1, 1],
                     indpb=0.2)
    toolbox.register('select', tools.selNSGA2)

    pop = toolbox.population(n=pop_size)
    hof = tools.ParetoFront()

    algorithms.eaMuPlusLambda(pop, toolbox, mu=pop_size, lambda_=pop_size, cxpb=0.7, mutpb=0.2, ngen=generations,
                              halloffame=hof, verbose=False)

    results = []
    for ind in hof:
        dec = decode_individual(ind)
        ra, en = ind.fitness.values
        dec.update({'surface_roughness': ra, 'energy_consumption': en})
        results.append(dec)

    df = pd.DataFrame(results)

    st.subheader("📊 Optimization Results — Pareto Front")
    st.dataframe(df)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df['energy_consumption'], df['surface_roughness'])
    ax.set_xlabel('Energy Consumption')
    ax.set_ylabel('Surface Roughness (Ra)')
    ax.set_title('Pareto Front')
    st.pyplot(fig)

    # Export
    df.to_csv('results/pareto_solutions_streamlit.csv', index=False)
    st.success("Optimization complete! File saved to results/pareto_solutions_streamlit.csv")

# ------------------------------
st.info("Use the left panel for manual ML predictions or scroll down to run full optimization.")
