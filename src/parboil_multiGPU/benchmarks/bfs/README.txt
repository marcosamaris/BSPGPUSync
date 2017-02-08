MULTI-GPU bfs
-------------

This bfs implementation has 3 kernels
1. BFS_in_GPU_kernel : used when the number of nodes discovered in the new_frontier if < NUM_THREADS_PER_BLOCK
2. BFS_kernel_multi_blk_inGPU : Used when the   NUM_THREADS_PER_BLOCK > no. of nodes discovered <  NUM_THREADS_PER_BLOCK *NUM_SMs
3. BFS_kernel : This is a generalized kernel that works in all cases but is used when the no. of nodes > NUM_THREADS_PER_BLOCK *NUM_SMs

BFS implementation if highly bandwidth limited and also limited by the size of shared memory storage. 
There is scope of speedup for BFS when moved onto multiple devices

The multiple-GPUs can be used in the last situation for huge number of nodes discovered.
This code implements a 2-GPU version of BFS which is NOT getting speedup for reasons explained later.

Whenever the number of discovered in the new frontier > NUM_THREADS_PER_BLOCK * NUM_SMs
1. The current state of execution (gloabl variables, buffer containing the new_frontier) are transfered to the 2nd GPU  
2. The work is divided among the 2-GPUs based on the number of blocks. 
   1st GPU gets Num_of_nodes= NUM_BLOCKS/2 * NUM_THREADS_PER_BLOCK 
   2nd GPU gets Num_of_nodes= NUM_BLOCKS - ceil(NUM_BLOCKS/2) * NUM_THREADS_PER_BLOCK
3. Based on the situation (NUM_BLOCKS%2 value) determines what type of kernel to be launched
4. Each type of kernel does basically the same work but they also update the state of the whole frontier on the respective GPUs.
   They discover the nodes corresponding to their current frontier nodes only. 
5. The number of nodes discovered by the 2nd GPU along with the buffer holding the new frontier nodes is transferred to the 1st GPU

The next level is treated int the same way only if the sum of nodes discovered > NUM_THREADS_PER_BLOCK * NUM_SMs or else the other type of
kernels mentioned at the start are used based on the total number of nodes.

This implementation does not produce any speedup for the test graph used because
1. SF Graph used is very deep with long diameter. The average degree of each node is a max of 6. The above multi-GPU implementation might
   be useful when the graph has large degree.
2. The Inter-GPU transfers done to maintain the same state of execution on both GPUs might become a bottleneck if the work done by both GPUs
   is not high at each level of the tree. The transfers cannot be non-blocking and hence the price of transfer has to be paid.
