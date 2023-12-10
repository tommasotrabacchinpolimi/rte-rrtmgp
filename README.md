# Computation of radiative fluxes challenge
This is the implementation of the second challenge of the Parallel Computing course at Politecnico di Milano. \
The first task was to rewrite in C serial code the `gas_optical_depths_minor` kernel. which can be found at /content/rte-rrtmgp-cpp/src_kernels_cuda/gas_optics_kernels.cu. The kernel is launched twice in the `compute_tau_absorption` function in /content/rte-rrtmgp-cpp/src_kernels_cuda/gas_optics_kernel_launchers.cu \
The second task was to find a way to improve the performance of the application with respect to the serial baseline.

## Compilation instructions
To compile the following instruction must be followed:

    mkdir build 
    cd build
    cmake .. -DSYST=ubuntu -DUSECUDA=on -DSERIAL=<on|off> -DIMPROVED=<on|off> -DPROFILE=<on|off>

The application must be compiled specifying all the following options:
* ``` -DSERIAL ``` : must be set to 'on' to compile the serial version of the kernel
* ``` -DIMPROVED ``` : must be set to 'on' to compile the optimized version of the kernel. Note that this option has no effect if the ``` -DSERIAL ``` option is set to on.
* ``` -DPROFILE ``` : must be set to on to profile the execution time of the kernel and to print the results.

And finally:

    make

## Execution instructions
To run the application is necessary to do the following:

    cd ../allsky
    ./make_links.sh
    python allsky_init.py
    python allsky_run_cuda.py

To verify the accuracy of the results, the following python script can be used:

    python compare-to-reference.py

A correct implementation shall have a maximum percent difference with respect to reference data of the order of 10^-7.

## Relevant files
The file modified to do the tasks of this challenge are
* the kernel source [file](src_kernels_cuda/gas_optics_kernels.cu)
* the kernel launcher [file](src_kernels_cuda/gas_optics_kernel_launchers.cu)

## Original repository
The original repository can be found [here](https://github.com/microhh/rte-rrtmgp-cpp)