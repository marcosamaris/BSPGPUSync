Multi-GPU  matrix multiplication 
----------------------------------

The Matrix multiplication kernel is Compute bound. Hence, it can get speedup by using multiple devices.
The code described here is multithreaded implementation. Pthreads are used for multithreading. 
Kernel implements a C=AXBT (matrices stored in column major format)

Each pthread controls 1 GPU and this implementation does not involve the threads to communicate with each other.
Number of pthreads = NUM of GPUS, this can be changed by modifying a #define NUM_GPUS in the code. 
Just changing this transparently handles the code to be ported to number of GPUs defined and no other 
change is required. 

Each pthread allocates and deallocates the memory for the GPU it is controlling. 
Each pthread makes a buffer in the host RAM which stores the part of A that it will be using to compute 
part of the result matrix. In short each holds a few columns of A (divided equally among them).
All the threads have to transfer whole B matrix.
Each GPU generates a few rows of the result matrix. The results computed by all GPUs are combined together to form 
the result matrix in the host RAM.

NOTE: The parboil test results show that there is MISMATCH in the results (for large input file) but they are CORRECT. This can be confirmed by examining the result files.



