# Calib 

a tool for calibrating control-bounded converters.
For additional tools related to the inputs and outputs of this tool see [calib_python](../calib_python/README.md)


## Validate

To validate run

```bash
calib validate -i test_data/s_test.npy -f test_data/filter.npy -o test.npy
```
where
- -i (--input) is the path to the test control signals
- -f (--filter) is the path to the estimation filter
- -o (--output) is the output path of the generated data


## Calibrate

```bash
calib calibrate -i test_data/s_train.npy -f test_data/filter.npy -o train.npy --iterations 100000000 --batch-size 200 --step-size 1e-5
```
where 
- -i (--input) is the path to the training control signals
- -f (--filter) is the path to the estimator filter (note that this command overwrites the coefficients)
- -o (--output) is the output path of the generated data
- --iterations total number of iterations
- --batch-size the number of gradient steps that should be grouped per update
- --step-size the fixed step size


## Install
To install run
```bash
RUSTFLAGS="-C target-cpu=native" cargo install --path . 
```