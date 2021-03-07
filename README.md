*under rapid development*

# iGEMMgen

Code generator for implicit gemm algorithm (generic tensor contraction)

# Generate kernel
since f-string is utilized in python, require python >= 3.6 to run.
```
# generate code based on tunable configuration, use one of following command to generate each direction
python3 igemm_codegen.py config/igemm_fwd_gtc_gfx908.config
python3 igemm_codegen.py config/igemm_bwd_gtc_gfx908.config
python3 igemm_codegen.py config/igemm_wrw_gtc_gfx908.config

# or auto generate code for all possible combinations, use one of following command to generate each direction
python3 igemm_codegen.py config/igemm_fwd_gtc_gfx908_seq.config
python3 igemm_codegen.py config/igemm_bwd_gtc_gfx908_seq.config
python3 igemm_codegen.py config/igemm_wrw_gtc_gfx908_seq.config
```

The output file will result in `out` directory. result in a assembly file `*.s` and several `*.inc` for different tile size, a codeobject `*.hsaco` and a host driver executable `conv_driver.exe`. This executable accept same cmdline argument as [MIOpenDriver](https://rocmsoftwareplatform.github.io/MIOpen/doc/html/driver.html). e.g.
```
./conv_driver.exe  conv -n 128 -c 1024 -H 17 -W 17 -k 1024  -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F 2 -V 1
```
currently this executable will run all the kernel configs one by one, the same as you used for kernel generation stage.

some environment variables may affect the behavior and printout of `conv_driver.exe`
* `IGEMM_HSACO` : indicate the path of code object to use. default use the generated one in currentl directory.
* `IGEMM_SCLK_MHZ` : current GPU sclk MHZ. used to calculate efficiency.
* `IGEMM_LOG_FASTEST_CONFIG` : set to `1` to print the fastest config from current convolution. default is `0`

*more description to be added*

# Third party code for fp16 data type
* `half.hpp` : When fp16 kernel is generated, `half.hpp` need to be installed, e.g.:
``` shell
wget https://github.com/pfultz2/half/archive/1.12.0.tar.gz
tar -zvxf 1.12.0.tar.gz
cp half-1.12.0/include/half.hpp /usr/local/include/
```
