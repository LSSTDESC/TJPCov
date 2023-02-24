#! /bin/bash
<<comment
This is a simple helper script to run TJPCov on NERSC with srun.  You can use 
the commands below:

This will allocate 16 nodes at NERSC (cori haswell) on the DESC account

    salloc -N 16 -C haswell -A m1727 -t 01:00:00 --qos interactive 

This will run this script (which runs tjpcov)

    srun -u -n 16 -c 64 run_tjpcov_mpi.sh your_config.yaml your_output.sacc

comment

input=$1
output=$2

# Source the desc-python python stack to get sacc, pyccl, etc.
source /global/common/software/lsst/common/miniconda/setup_current_python.sh

# Run tjpcov
time python3 run_tjpcov.py $input -o $output
