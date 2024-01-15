import argparse
import numpy as np
import os
import json
import matplotlib.pyplot as plt

from pytsp.tspga import TSPGA
from pytsp.cities import get_distance_matrix, plot_cities, generate_cities

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-c","--cities", type=str, default=None, help="Path to numpy matrix indicating the cities positions on x-y plane.")
    parser.add_argument("--population-size", type=int, default=100, help="Size of the population.")
    parser.add_argument("--num-generations", type=int, default=1000, help="Number of generations to run.")
    parser.add_argument("--tournament-size", type=int, default=10, help="Tournament size.")
    parser.add_argument("--crossover", type=str, default="OX", choices=["OX", "CX"], help="Crossover method to use.")
    parser.add_argument("--mutation-rate", type=float, default=0.01, help="Mutation rate.")
    parser.add_argument("--num-swaps", type=int, default=1, help="Number of swaps for mutation.")
    parser.add_argument("--max-no-convergence", type=int, default=200, help="Maximum number of generations with no convergence.")
    parser.add_argument("--n-starts", type=int, default=10, help="Number of starts for the GA.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of jobs for running the starts in parallel.")
    parser.add_argument("--save-dir", type=str, default="./", help="Directory to save outputs from the run.")

    args = parser.parse_args()

    # Set output directory
    out_dir = os.path.join(args.save_dir,"tspga_multirun")
    os.makedirs(out_dir, exist_ok=True)

    # Load or create cities distribution
    if args.cities is None: cities = generate_cities(n=50)
    else: cities = np.load(args.cities)
    
    # Get distance matrix
    distance_matrix = get_distance_matrix(cities)
    
    # Run Genetic Algorithm
    genetic = TSPGA(distance_matrix)
    best_individual, fitnesses = genetic.run_multi(population_size=args.population_size, 
                                             num_generations=args.num_generations, 
                                             tournament_size=args.tournament_size, 
                                             crossover_method=args.crossover, 
                                             mutation_rate=args.mutation_rate, 
                                             num_swaps=args.num_swaps, 
                                             max_no_convergence=args.max_no_convergence,
                                             n_starts=args.n_starts,
                                             n_jobs=args.n_jobs)
    
    # Make & Save output plots
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14,7))
    genetic.plot_fitness_curves(ax=ax[0])
    plot_cities(cities=cities, tour=best_individual.tour, ax=ax[1], fitness=best_individual.fitness)
    plt.savefig(os.path.join(out_dir, "tspga_multirun.png"))

    # Save best trajectory
    with open(os.path.join(out_dir, "best_tour.json"), "w") as f:
        json.dump(best_individual.tour, f)

if __name__ == "__main__":
    main()