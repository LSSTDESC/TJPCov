# Examples

Here we have some examples of calculating covariance matrices. The Python interface allows both in-notebook and mpi-capable computation of terms using the specified parameters in `.yaml` files.
In the main directory, we see a file called `tjpcov.py`. We may use this interface to compute a covariance matrix given an input specification file and an output name for the produced `sacc` file. 
We can generally run a computation as 
```
python3 tjpcov.py ./input/yaml/file.yaml ./output/sacc/file.sacc
```

An example submission script to an high-performance computing cluster with SLURM is also provided in the main directory in the `run_tjpcov_mpi.slurm` file. 

## Computing a full 3x2pt Covariance Matrix in Fourier Space

We provide two examples of computing a small 3x2pt covariance matrix with DESY1-like data in harmonic space: one with a survey mask (`full_3x2pt_cov_namaster_example.yml`) and one that uses the $f_\mathrm{sky}$ approximation (`full_3x2pt_cov_example.yml`). 
Including sky masks as input will call `NaMaster` to compute the relevant mode-mixing matrices. 
Both of these files require an input `sacc` datavector, from which  `TJPCov` will read in the data binning scheme, n(z)'s, and data combinations. 
In these examples, we make use of the data used to test TJPCov. However, the format is general enough to be able to swap different data.
The cosmology is also specified as a separate `.yaml` file, which helps to ensure a consistent cosmology between covariance computations. The cosmology is defined in the `3x2pt_cov_cosmology.yaml` of this directory. 

## Computing Additional Covariance Matrices

[Active Development]
