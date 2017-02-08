
/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
#include <parboil.h>
#include <stdio.h>
#include <stdlib.h>

#include "file.h"
#include "common.h"
#include "kernels.h"
#include "kernels.cu"
#define KEY 1
static int generate_data(float *A0, int nx,int ny,int nz) 
{	
	srand(54321);
	int s=0;
	for(int i=0;i<nz;i++)
	{
		for(int j=0;j<ny;j++)
		{
			for(int k=0;k<nx;k++)
			{
				A0[s] = (rand() / (float) RAND_MAX);
				s++;
			}
		}
	}
	return 0;
}

int main(int argc, char** argv) {
	struct pb_TimerSet timers;
	struct pb_Parameters *parameters;
	

	cudaSetDevice(0);	
	printf("CUDA accelerated 7 points stencil codes****\n");
	printf("Author: Li-Wen Chang <lchang20@illinois.edu>\n");
	parameters = pb_ReadParameters(&argc, argv);

	pb_InitializeTimerSet(&timers);
	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	

#if KEY>0
        int nx=64,ny=64,nz=8;
        int size;
    int tx=16,ty=4;
#else
        int nx=640,ny=640,nz=640;
        int size;
    int tx=32,ty=8;
#endif
	
	//declaration
    int iteration=100;
	float c0=1.0f;
	float c1=1.0f/6.0f;
/*
	if (argc<7) 
    {
      printf("Usage: probe nx ny nz tx ty t\n"
	     "nx: the grid size x\n"
	     "ny: the grid size y\n"
	     "nz: the grid size z\n"
         "tx: the block size x\n"
         "ty: the block size y\n"
		 "t: the iteration time");
      return -1;
    }

	nx = atoi(argv[1]);
	if (nx<1)
		return -1;
	ny = atoi(argv[2]);
	if (ny<1)
		return -1;
	nz = atoi(argv[3]);
	if (nz<1)
		return -1;
	tx = atoi(argv[4]);
	if(tx<1)
		return -1;
	ty = atoi(argv[5]);
	if(ty<1)
		return -1;
	iteration = atoi(argv[6]);
	if(iteration<1)
		return -1;
*/
	
	//host data
	float *h_A0;
	float *h_Anext;
	//device
	float *d_A0;
	float *d_Anext;
float *tmp;
	

	//load data from files
	pb_SwitchToTimer(&timers, pb_TimerID_IO);
	/*
	inputData(parameters->inpFiles[0], &nz, &ny, &nz);
		*/
	
	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	
	size=nx*ny*nz;
	
	h_A0=(float*)malloc(sizeof(float)*size);
	h_Anext=(float*)malloc(sizeof(float)*size);
	generate_data(h_A0, nx,ny,nz);
	
	float *h_B=(float*)malloc(sizeof(float)*size);
	float *h_next=(float*)malloc(sizeof(float)*size);
memcpy(h_B,h_A0,size*sizeof(float));

#if KEY>0
for (int t = 0; t < iteration;t++){

for (int k = 1; k < nz - 1; k++) {

for (int j = 1; j < ny - 1; j++) {

for (int i = 1; i < nx - 1; i++) {

h_next[Index3D (nx, ny, i, j, k )] = (h_B[Index3D (nx, ny, i, j, k+1 )] + h_B[Index3D (nx, ny, i, j, k-1 )] + h_B[Index3D (nx, ny, i, j+1, k )] + h_B[Index3D (nx, ny, i, j-1, k )] + h_B[Index3D (nx, ny, i+1, j, k )] + h_B[Index3D (nx, ny, i-1, j, k )])*c1 - c0 * h_B[Index3D (nx, ny, i, j, k )];

}

}

}

tmp = h_B;

h_B = h_next;

h_next = tmp;

}
#endif
	pb_SwitchToTimer(&timers, pb_TimerID_COPY);
	//memory allocation
	cudaMalloc((void **)&d_A0, size*sizeof(float));
	cudaMalloc((void **)&d_Anext, size*sizeof(float));
	cudaMemset(d_Anext,0,size*sizeof(float));

	//memory copy
	cudaMemcpy(d_A0, h_A0, size*sizeof(float), cudaMemcpyHostToDevice);
	
//	if (parameters->synchronizeGpu) cudaThreadSynchronize();
	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	float *temp;
	//only use half size of orignal threads
	//only use tx by ty threads
	dim3 block (tx, ty, 1);
	//also change threads size maping from tx by ty to 2tx x ty
	dim3 grid ((nx+tx*2-1)/(tx*2), (ny+ty-1)/ty,1);
	int sh_size = tx*2*ty*sizeof(float);	
	printf(" blockks= %d\n ", grid.x*grid.y);
	//main execution
	pb_SwitchToTimer(&timers, pb_TimerID_GPU);
	for(int t=0;t<iteration;t++)
	{
		block2D_hybrid_coarsen_x<<<grid, block,sh_size>>>(c0,c1, d_A0, d_Anext, nx, ny,  nz);
		temp=d_A0;
		d_A0=d_Anext;
		d_Anext=temp;

	}
    CUERR // check and clear any existing errors

	
	
	
	 cudaDeviceSynchronize();
	pb_SwitchToTimer(&timers, pb_TimerID_COPY);
	cudaMemcpy(h_Anext, d_A0,size*sizeof(float), cudaMemcpyDeviceToHost);
#if KEY>0
for (int i=0;i<size;i++)
	printf("hA0[%d]=%f hnext[%d]=%f\n",i,h_B[i],i,h_Anext[i]);
#endif
//	if (parameters->synchronizeGpu) cudaThreadSynchronize();
	cudaFree(d_A0);
    cudaFree(d_Anext);
 
	if (parameters->outFile) {
		 pb_SwitchToTimer(&timers, pb_TimerID_IO);
		outputData(parameters->outFile,h_Anext,nx,ny,nz);
		
	}
	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
		
	free (h_A0);
	free (h_Anext);
	pb_SwitchToTimer(&timers, pb_TimerID_NONE);

	pb_PrintTimerSet(&timers);
	pb_FreeParameters(parameters);

	return 0;

}
