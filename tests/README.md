
# test_2.py:
sanity checks and comparison between API and example/Gaussian_cov-DES.ipynb

## Run basic tests
```
python3 -m pytest -rs --log-cli-level=INFO --capture=no  tests/test_2.py 
```

## Run all the covariances and compare with example
```
python3 -m pytest -rs --log-cli-level=INFO --capture=no --runslow tests/test_2.py 
```
