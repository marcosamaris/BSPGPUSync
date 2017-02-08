Jacobi 7 point Stencil
-----------------------

Stencil kernel works along a plane and within the kernel moves along the z dimension
Iterations are done to refine the result by applying the stencil operation on the input many times.

NOTE: The parboil implementation does not do the swap of the buffers which holds the result and
      the i/p between  iterations. This is incorrect.

The multi-GPU mplementation does the swap.
The critical thing about this kernel is that the planes split between the multiple GPUs share state.
Hence, the shared state (plane) is first computed and transferred to the remote GPU while the remaining
part of the stencil operation on the other planes are performed.
Stencil kernel is bandwidth limited. Theoritical the multi-GPU implementation should get speedup.
This is practically possible if the transfer time is less than the time to compute the remaining
planes of the input. For the input specified in this parboil it almost scales linearly with the increase in GPUs.

The kernel specified in this file is specifically designed for 2 GPU system.
In this implementation the shared state or halo is computed by separate kernels.
Then the shared states are transferred between the GPUs while the remaining computations are 
done by another kernel on all the GPUs. The transfer is done by using asynchronous transfer
(cudaMemcpyAsync()) with the help of cuda streams to overlap transfer and computation.
The cost of transfer can be hidden behind computation depending upon the size of the problem.

This implementation is done apart from the other implementation which used 2GPUs where the GPUs 
are connected under the same PCI-tree
