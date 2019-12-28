import os, sys
from constraints import Constraint
from random import random, sample
from scipy.spatial import distance
import numpy as np

def generate_random_samples(cons, delta_adaptability = 1.1, n_results = 10, try_count_thresh = 10**9, min_delta = 10**-12):
    """
    generate n_results random samples satisfying all specified constraints using a variant pf Markov Chain Monte Carlo sampling.
    delta_adaptability: float number > 1.0. Larger values result in faster change of delta (i.e. offset between a known good point and a newly generated point).
                        delta is increased each time a new valid sample is found and decreased when a new proposed sample fails to satisfy all constraints.
    """
    output, n_dim = [cons.get_example()], cons.get_ndim()
    delta = 1 / delta_adaptability

    for i in range(n_results - 1):
        delta = max(1, delta*delta_adaptability)
        try_count, vector_accepted = 0, False

        while not vector_accepted:
            try_count += 1
            center = sample(output,1)[0]
            vector = np.mod(center + np.random.uniform(-delta, delta, n_dim), 1)
            vector_accepted = cons.apply(vector)
            
            if vector_accepted:
                output.append(vector)
            else:
                if delta > min_delta:
                    delta /= delta_adaptability
                else:
                    delta = random()
            
            if try_count >= try_count_thresh:
                raise RuntimeError("Failed to find a valid point within the specified number of attempts")
                return None

    return output

def spread_out_samples(cons, samples, resample_attempts = 1000):
    """
    Spreads out sample points by finding the closest pair of points and attempting to replace one of the points in the pair.
    Sample is replaced if and only if the distance between the new closest pair of points is larger than the original one.
    Resample attempts are done one at a time. Point-to-point distances are recalculated after each attempt.
    """
    if len(samples) < 2:
        print('Sample spreading algorithm requires at least two points to generate a non-trivial solution')
        return samples, []
    
    min_dist_history = []

    for _ in range(resample_attempts):
        dists = distance.cdist(samples, samples, metric='euclidean')
        for i in range(len(dists)):
            dists[i,i] = cons.n_dim #set diagonal elements (which were  = 0) to something large to make finding actual minimal distances easy

        min_dist_inds = np.unravel_index(np.argmin(dists, axis = None), dists.shape)
        vector_ind_to_replace = min_dist_inds[0]
        min_dist_history.append(dists[min_dist_inds])

        resampled = generate_random_samples(cons, n_results = 2)[-1]

        dists_to_resampled = distance.cdist([resampled], samples[:vector_ind_to_replace] + samples[vector_ind_to_replace + 1:], metric='euclidean')
        new_min_dist_inds = np.unravel_index(np.argmin(dists_to_resampled, axis = None), dists_to_resampled.shape)
        
        if dists_to_resampled[new_min_dist_inds] > min_dist_history[-1]:
             samples[vector_ind_to_replace] = resampled

    return samples, min_dist_history

def parse_inputs(input_args):
    input_fname_path, output_filename, n_results = sys.argv[1:]
    try:
        n_results = int(n_results)
        if n_results < 2:
            raise ValueError("n_results must be an integer larger than 1")
    except:
        raise ValueError("n_results must be an integer larger than 1")

    if not os.path.isfile(input_fname_path):
        raise OSError("Input file not found")
    
    return input_fname_path, output_filename, n_results

if __name__ == "__main__":
    input_fname, output_fname, n_results = parse_inputs(sys.argv[1:])

    cons = Constraint(input_fname)
    samples = generate_random_samples(cons, n_results = n_results)
    samples, _ = spread_out_samples(cons, samples, resample_attempts = max(1000, n_results)) 
    np.savetxt(output_fname, samples, fmt='%f', delimiter=' ')
