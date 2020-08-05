*under rapid development*

# iGEMMgen

Code generator for implicit gemm algorithm (generic tensor contraction)

# Generate kernel
since we use f-string of python, we require python >= 3.6 to run.
```
# generate code based on tunable configuration
python3 igemm_codegen.py config/igemm_bwd_gtc.config
```
The output file will result in `out` directory. result in a assembly file `*.s`, a codeobject `*.hsaco` and a host driver executable `conv_driver.exe`. This executable accept same cmdline argument as [MIOpenDriver](https://rocmsoftwareplatform.github.io/MIOpen/doc/html/driver.html). e.g.
```
./conv_driver.exe  conv -n 128 -c 1024 -H 17 -W 17 -k 1024  -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F 2
```



*more description to be added*
