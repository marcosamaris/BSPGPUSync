Code Correctness
----------------


1. Dense  Matrix-Matrix multilication (mm)
        mm - Implements multi-GPU matmul using pthreads
        mm_2GPU- Implements the same with 1 host thread

Result after the execution displays "MISMATCH"
But the results are correct (this can be confirmed by checking
the result file /PARBOIL_PATH/benchmarks/mm/run/input/large/matrix3.txt and
 /PARBOIL_PATH/benchmarks/mm/run/output/large/matrix3.txt


2. Stencil code

        stencil_2GPU - 2 GPU impelemntation with GPUs attached under the same PCI-tree
        stencil_3GPU-  3 GPU impelemntation with GPUs attached under the same PCI-tree
        stencil_2GPU_nopeer -  2 GPU impelemntation
        stencil_3GPU_nopeer -  3 GPU impelemntation

All the code files display a MISMATCH in the result because the original
parboil stencil code is incorrect stencil_new implements the correct version of the stencil computation

After many iterations of the stencil computation the values start getting truncated and precision is lost. 
Floating point operations are not Associative and hence results obtatined from the different configurations
stated above are different. 
To check the correctness of the code reduce the iterations to 10 because at this level the numbers
can be completely presented using 32 bits
