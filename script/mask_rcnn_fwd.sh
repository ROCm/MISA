
#!/bin/sh
export IGEMM_HSACO=out/igemm_fwd_gtc_gfx908.hsaco
##export IGEMM_HSACO=out/tmp_all_reduced_reordered.hsaco
export IGEMM_SCLK_MHZ=1283
export IGEMM_LOG_FASTEST_CONFIG=0

# Flag enables fwd, bwd, wrw convolutions
FORW=1
Verify=1

set -x 

#mask-rcnn
./out/conv_driver2.exe conv -n 2 -c 256 -H 12 -W 18 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F $FORW -V $Verify
./out/conv_driver2.exe conv -n 2 -c 1024 -H 34 -W 84 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW -V $Verify
./out/conv_driver2.exe conv -n 2 -c 1024 -H 40 -W 52 -k 512 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -F $FORW -V $Verify
./out/conv_driver2.exe conv -n 2 -c 256 -H 100 -W 104 -k 12 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW -V $Verify
./out/conv_driver2.exe conv -n 4 -c 256 -H 14 -W 14 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F $FORW -V $Verify
./out/conv_driver2.exe conv -n 4 -c 256 -H 28 -W 28 -k 256 -y 2 -x 2 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -F $FORW -V $Verify
./out/conv_driver2.exe conv -n 3 -c 256 -H 28 -W 28 -k 80 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW -V $Verify

./out/conv_driver2.exe conv -n 1 -c 256 -H 100 -W 112 -k 12 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW -V $Verify 
./out/conv_driver2.exe conv -n 1 -c 256 -H 28 -W 28 -k 80 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW -V $Verify 
./out/conv_driver2.exe conv -n 1 -c 1024 -H 28 -W 84 -k 256 -y 1 -x 1 -p 0 -q 0 -u 1 -v 1 -l 1 -j 1 -F $FORW -V $Verify
./out/conv_driver2.exe conv -n 1 -c 1024 -H 48 -W 84 -k 512 -y 1 -x 1 -p 0 -q 0 -u 2 -v 2 -l 1 -j 1 -F $FORW -V $Verify 
./out/conv_driver2.exe conv -n 1 -c 256 -H 28 -W 84 -k 256 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -F $FORW -V $Verify

set +x 
