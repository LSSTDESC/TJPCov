
# test/test_*.py:
sanity checks and comparison between API and example/Gaussian_cov-DES.ipynb

## Run basic tests
```
pytest -rs --log-cli-level=INFO --capture=no   
```

## Run all the covariances and compare with example
```
pytest -rs --log-cli-level=INFO --capture=no --runslow 
```
