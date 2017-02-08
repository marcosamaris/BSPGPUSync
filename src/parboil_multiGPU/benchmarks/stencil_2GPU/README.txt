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
In this implementation the shared state or halo is computed first by both the GPUs and transferred to the  
remote GPU from the kernel itself utilizing UVA (Unified Virtual Addressing) capability of the Fermi architecture.
The remote store does a PCI transfer while the  rest of the stencil computation is performed by the kernel.
cudamemcpypeer is performed after the computation to ensure that the transfer is completed. This type of memcpy 
is cheaper than performing a cudaDevicesync().

The remote and cudamemcpypeer work only if Peer access is enabled between the GPUs, for that it is necessary 
that they belong the same PCI-tree

This gives very linear speedup.


