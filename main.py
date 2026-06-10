import os
import argparse
import cProfile
import pstats

from compflowlab.utils import input_read_func
from compflowlab.driver.run import run


# Set up argument parsing
parser = argparse.ArgumentParser(description='Process an input file and specify a working directory.')
parser.add_argument('working_directory', type=str, help='Path to the working directory.')
parser.add_argument('input_file', type=str, help='Path to the input file.')

args = parser.parse_args()

# Check if the working directory exists
if not os.path.isdir(args.working_directory):
    print(f"Error: The working directory '{args.working_directory}' does not exist.")
    exit(1)

# Check if the input file exists
if not os.path.isfile(args.input_file):
    print(f"Error: The input file '{args.input_file}' does not exist.")
    exit(1)

# read the input file
input_param  = input_read_func.read_input_file(args)

# collect all of variables from input file and prepare them for simulation
solver_param = input_read_func.init_solver_param(args,input_param)

# start running the simulation
if solver_param['profiling'] == True:

    with cProfile.Profile() as pr:

        state = run(solver_param)
    
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename=args.working_directory +'/'+ 'profiling_results.prof')

else: 

    state = run(solver_param)




