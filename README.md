# iGEMMgen

Code generator for implicit gemm algorithm

# Generate kernel
Please install `numpy` before use this tool.
```
# generate code based on tunable configuration
python3 igemm_codegen.py config/igemm_v4r1_dynamic.config
```
The output file will result in `out` directory. result in a assembly file `*.s`, a codeobject `*.hsaco` and a host driver executable `conv_driver.exe`. This executable accept same cmdline argument as [MIOpenDriver](https://rocmsoftwareplatform.github.io/MIOpen/doc/html/driver.html). For a quick start, can use `script/v4r1_origin_conv.sh` int the top directory to launch the driver with several tensor descriptors.

# Iterate all possible combinations
```
# check all possible tiling based on configuation
python3 igemm_codegen.py config/igemm_v4r1_dynamic_seq.config > comb.txt
```

*more description to be added*
