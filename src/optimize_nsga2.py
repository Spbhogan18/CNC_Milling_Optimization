"""Run NSGA-II (via DEAP) using the trained surrogate models to find Pareto-optimal machining parameters.
"""
import argparse
import numpy as np
import joblib
from deap import base, creator, tools, algorithms
import random
import pandas as pd

RNG = random.Random(42)

BOUNDS = {
    'spindle_speed': (500, 5000),
    'feed_rate': (0.05, 0.5),
    'depth_of_cut': (0.1, 5.0),
}

def decode_individual(ind):
    return {
        'spindle_speed': ind[0],
        'feed_rate': ind[1],
        'depth_of_cut': ind[2],
        'coolant': int(round(ind[3])),
        'tool_B': int(round(ind[4]))
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs=2, required=True, help='rf_ra.joblib rf_en.joblib')
    parser.add_argument('--pop', type=int, default=80)
    parser.add_argument('--gens', type=int, default=120)
    parser.add_argument('--out', default='results/pareto_solutions.csv')
    args = parser.parse_args()
    rf_ra = joblib.load(args.models[0])
    rf_en = joblib.load(args.models[1])
    scaler = joblib.load('results/scaler.joblib')

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
        pred_ra = float(rf_ra.predict(Xs)[0])
        pred_en = float(rf_en.predict(Xs)[0])
        return pred_ra, pred_en

    toolbox.register('evaluate', eval_ind)
    toolbox.register('mate', tools.cxBlend, alpha=0.5)
    toolbox.register('mutate', tools.mutPolynomialBounded, eta=20.0, low=[BOUNDS['spindle_speed'][0], BOUNDS['feed_rate'][0], BOUNDS['depth_of_cut'][0], 0, 0],
                     up=[BOUNDS['spindle_speed'][1], BOUNDS['feed_rate'][1], BOUNDS['depth_of_cut'][1], 1, 1], indpb=0.2)
    toolbox.register('select', tools.selNSGA2)

    pop = toolbox.population(n=args.pop)
    hof = tools.ParetoFront()
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('avg', np.mean, axis=0)
    stats.register('min', np.min, axis=0)

    algorithms.eaMuPlusLambda(pop, toolbox, mu=args.pop, lambda_=args.pop, cxpb=0.7, mutpb=0.2, ngen=args.gens,
                              stats=stats, halloffame=hof, verbose=True)

    pareto_list = []
    for ind in hof:
        dec = decode_individual(ind)
        ra, en = ind.fitness.values
        dec.update({'surface_roughness': ra, 'energy_consumption': en})
        pareto_list.append(dec)
    pd.DataFrame(pareto_list).to_csv(args.out, index=False)
    print('Saved pareto solutions to', args.out)
