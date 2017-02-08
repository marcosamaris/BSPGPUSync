Multi-GPU Cut-off Coulombic Potential (cutcp)
----------------------------------------------

This kernel is highly computationaly intensive and gives linear speedup with
increase in the number of GPUs

The multi-GPU implementation is done with the help of pthreads where each
thread controls a GPU and the work is evenly divided among all GPUs.


The input is regularized using small 3-D bins that hold the atoms whose effect
is computed at the various grid points. The work division is done in such a
way that each GPU computes a Cuboidal region of grid points and the combined 
work generates the potential across the entire grid.

An additional pthread is used apart from the threads controlling independent 
GPUs, for computing the potential at various grid points contributed by the
atoms which overflowed from the bins due to their restricted small size.

Thus, a full CPU-GPU utilization is achieved with hiding the CPU compute cost
behind the GPU compute.

