#! /bin/bash
<<comment
salloc -N 16 -C haswell -A m1727 -t 03:00:00 --qos interactive 
srun -n 16 -c 64 clusters-mpi.sh conf_covariance_clusters.yaml sacc_with_cov.sacc
comment

input=$1
output=$2

source /global/common/software/lsst/common/miniconda/setup_current_python.sh

time python3 /global/homes/m/mkwiecie/desc/repos/TJPCov/run_tjpcov.py /global/homes/m/mkwiecie/desc/repos/TJPCov/examples/clusters/$input -o /global/homes/m/mkwiecie/desc/repos/TJPCov/examples/clusters/$output
