#!/bin/sh
#SBATCH --job-name=tjpcov-clusters-run
#SBATCH --mem=256000
#SBATCH --nodes=8
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --account=m1727
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --licenses=cfs,SCRATCH
#SBATCH --output=/pscratch/sd/m/mkwiecie/tjpcov/tjpcov-clusters-run.log-%j.txt
#SBATCH --mail-user=mkwiecie@ucsc.edu
#SBATCH --mail-type=ALL
#SBATCH --module=mpich
 
input=your_config.yaml
output=your_output.sacc

# Source the desc-python python stack to get sacc, pyccl, etc.
source /global/common/software/lsst/common/miniconda/setup_current_python.sh

# Run tjpcov
python3 run_tjpcov.py $input -o $output
