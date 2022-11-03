#!/usr/bin/python

import os, sys

sys.path.append("/global/u1/m/mkwiecie/desc/repos/TJPCov")
sys.path.append("/global/u1/m/mkwiecie/.local/lib/python3.8/site-packages")

import pickle
from tjpcov_new.covariance_cluster_counts import CovarianceClusterCounts

if __name__ == "__main__":

    input_yml = "/global/u1/m/mkwiecie/desc/repos/TJPCov/examples/clusters/tjpcov_conf_minimal_clusters.yaml"
    cov = CovarianceClusterCounts(input_yml)
    cov_nxn = cov.get_covariance()

    fname = "/global/u1/m/mkwiecie/desc/repos/TJPCov/examples/clusters/cov_nxn_mpi_large.pkl"
    with open(fname, "wb") as ff:
        pickle.dump(cov_nxn, ff)
