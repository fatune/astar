# astar
cython realisation of a-star path finding

## building

python3 setup.py build_ext --inplace

## testing:
pytest -s tests/test_test.py


So far pure python realisation is quite fast on itself and cython optimization doesn't seem to be even necessary.
