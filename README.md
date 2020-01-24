# TJPCov
TJPCov is a general covariance calculator interface to be used within LSST DESC


## Planning
The initial discussion about the API is happening in [google doc](https://docs.google.com/document/d/1uA_82Ld7k0PPJaljMyelN_dJ4ZRin44FbJja5nZJlVc/edit?usp=sharing]).

and in the #desc-tjpcov channel on the LSST DESC.

See also [terms of reference](doc/Terms_of_Reference.md).


We have a working example now. Clone the repo and do:

python run_gaussian_cov.py des_y1_3x2pt.yaml

There should be two temp files with real and fourier space covariances.
