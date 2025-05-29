# Faster-pWasserstein

Implementation of "Scalable Approximation Algorithms for p-Wasserstein Distance and Its Variants"

To install the requirements and cythonize the code:
```
pip install -r requirements.txt
python setup.py build_ext --inplace
```

To compute the p-Wasserstein distance between two sets of samples from a uniform distribution inside the unit hypercube:
```
python main.py --sample_size 5000 --p 2 --dim 5 --distribution Uniform
```

To generate two discrete distributions over the samples from a normal distribution (computing OT instead of matching):
```
python main.py --sample_size 5000 --p 2 --dim 5 --distribution Normal_same --ot
```
