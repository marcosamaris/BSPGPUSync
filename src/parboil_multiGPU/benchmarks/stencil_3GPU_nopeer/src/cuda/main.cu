
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
#include <stdint.h>
#include "kernels.cu"

#include <sys/time.h>
#include <unistd.h>
cudaStream_t stream1,stream2;
#define KEY 0
#define handle_error_en(en, msg) \
               do { errno = en; perror(msg); exit(EXIT_FAILURE); } while (0)
double wall_time() {
	static bool first_call = true;
	static double start_time;

	struct timeval tv;
	gettimeofday(&tv,0);
	double now = tv.tv_sec + 1e-6*tv.tv_usec;

	if (first_call) {
		first_call = false;
		start_time = now;
	}
	return now - start_time;
}

/// \brief Wrappers around platform dependent timers and performance info

/// Returns the wall time in seconds relative to arbitrary origin
/// As accurate and lightweight as we can get it, but may not
/// be any better than the gettime of day system call.

/// On some machines we have access to a cycle count

/// Otherwise uses wall_time() in nanoseconds.
static inline uint64_t cycle_count() {
	uint64_t x;
	unsigned int a,d;
	__asm__ volatile("rdtsc" : "=a"(a), "=d"(d));
	x = ((uint64_t)a) | (((uint64_t)d)<<32);
	return x;
}

double cpu_frequency() {
	static double freq = -1.0;
	if (freq == -1.0) {
		double used = wall_time();
		uint64_t ins = cycle_count();
		if (ins == 0) return 0;
		while ((cycle_count()-ins) < 100000000);  // 100M cycles at 1GHz = 0.1s
		ins = cycle_count() - ins;
		used = wall_time() - used;
		freq = ins/used;
	}
	return freq;
}
/// Returns the cpu time in seconds relative to arbitrary origin

/// As accurate and lightweight as we can get it, but may not
/// be any better than the clock system call.
static inline double cpu_time() {
	static const double rfreq = 1.0/cpu_frequency();
	return cycle_count()*rfreq;
}


/// Do nothing and especially do not touch memory
inline void cpu_relax() {
	asm volatile("rep;nop" : : : "memory");
}

/// Sleep or spin for specified no. of microseconds

/// Wrapper to ensure desired behavior (and what is that one might ask??)
// usleep(us);


static double tttt, ssss;
#define STARTt_TIMER  tttt=wall_time(); ssss=cpu_time()
#define ENDt_TIMER(msg) tttt=wall_time()-tttt; ssss=cpu_time()-ssss;  printf("timer: %20.20s %8.10fs %8.10fs\n", msg, ssss, tttt)


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
	
//	cudaSetDevice(1);
//	cudaDeviceEnablePeerAccess(0,0);
//	cudaSetDevice(0);
//	cudaDeviceEnablePeerAccess(1,0);
  //  CUERR // check and clear any existing errors
	
	
	float *d_B0,*d_B1;
	printf("CUDA accelerated 7 points stencil codes****\n");
	printf("Author: Li-Wen Chang <lchang20@illinois.edu>\n");
	parameters = pb_ReadParameters(&argc, argv);

	pb_InitializeTimerSet(&timers);
	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	//declaration

#if KEY>0
	int nx=64,ny=64,nz=8;
	int size;
    int tx=16,ty=4;
#else
	int nx=512,ny=512,nz=64;
	int size;
    int tx=32,ty=8;
#endif
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
	float *h_Anext0;
	//device
	float *d_A0,*d_A1;
	float *d_Anext0,*d_Anext1;

float *tmp;	
	size=nx*ny*nz;
	float *h_A=(float*)malloc(sizeof(float)*size);
	float *h_B=(float*)malloc(sizeof(float)*size);
	float *h_next=(float*)malloc(sizeof(float)*size);
	generate_data(h_A, nx,ny,nz);
memcpy(h_B,h_A,size*sizeof(float));
#if KEY <0
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
	//load data from files
	pb_SwitchToTimer(&timers, pb_TimerID_IO);
	/*
	inputData(parameters->inpFiles[0], &nz, &ny, &nz);
		*/
	
	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
	
cudaStream_t stream0, stream1;	
	cudaSetDevice(0);
	cudaStreamCreate(&stream1);
	cudaMalloc((void**)&d_B1,sizeof(float));	
	pb_SwitchToTimer(&timers, pb_TimerID_COPY);

	//memory allocation
	cudaMalloc((void **)&d_A1, (size/2+nx*ny)*sizeof(float));
	cudaMalloc((void **)&d_Anext1, (size/2+nx*ny)*sizeof(float));
	cudaMemset(d_Anext1,0,(size/2+nx*ny)*sizeof(float));

	//memory copy
	cudaMemcpy(d_A1, h_A+size/2-nx*ny, (size/2+nx*ny)*sizeof(float), cudaMemcpyHostToDevice);


	cudaSetDevice(1);
	cudaStreamCreate(&stream0);
	cudaMalloc((void**)&d_B0,sizeof(float));	
	h_Anext0=(float*)malloc(sizeof(float)*size);
	
	pb_SwitchToTimer(&timers, pb_TimerID_COPY);
	//memory allocation
	cudaMalloc((void **)&d_A0, (size/2+nx*ny)*sizeof(float));
	cudaMalloc((void **)&d_Anext0, (size/2+nx*ny)*sizeof(float));
	cudaMemset(d_Anext0,0,(size/2+nx*ny)*sizeof(float));

	//memory copy
	cudaMemcpy(d_A0, h_A,(size/2+nx*ny)*sizeof(float), cudaMemcpyHostToDevice);
	
//	if (parameters->synchronizeGpu) cudaThreadSynchronize();
	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);

	//only use half size of orignal threads
	//only use tx by ty threads
	dim3 block (tx, ty, 1);
	//also change threads size maping from tx by ty to 2tx x ty
	dim3 grid ((nx+tx*2-1)/(tx*2), (ny+ty-1)/ty,1);
	int sh_size = tx*2*ty*sizeof(float);	
	printf(" blockks= %d\n ", grid.x*grid.y);
	//main execution
	pb_SwitchToTimer(&timers, pb_TimerID_GPU);
float *temp1,*temp2;
float *cur_d_Br0, *cur_d_Br1;
float *cur_d_Bw0, *cur_d_Bw1;
//STARTt_TIMER;
	for(int t=0;t<iteration;t++)
	{
	cudaSetDevice(0);
		block2D_hybrid_coarsen_interface<<<grid, block,sh_size,stream1>>>(c0,c1, d_A1, d_Anext1, nx, ny,  nz/2, d_A1,NULL,1);
	cudaSetDevice(1);
		block2D_hybrid_coarsen_interface<<<grid, block,sh_size,stream0>>>(c0,c1, d_A0+nx*ny*((nz/2)-2), d_Anext0, nx, ny,  nz/2, d_A0,NULL,nz/2-1);

	cudaSetDevice(0);
		cudaMemcpyAsync(d_Anext0+nx*ny*(nz/2),d_Anext1+nx*ny,nx*ny*sizeof(float),cudaMemcpyDefault,stream1);
		cur_d_Br1=d_A1;cur_d_Bw1=d_Anext0;
		block2D_hybrid_coarsen_x1<<<grid, block,sh_size,stream1>>>(c0,c1, d_A1, d_Anext1, nx, ny,  nz/2, cur_d_Br1,cur_d_Bw1,1);

	cudaSetDevice(1);
		cudaMemcpyAsync(d_Anext1,d_Anext0+nx*ny*((nz/2)-1),nx*ny*sizeof(float),cudaMemcpyDefault,stream0);
		cur_d_Br0=d_A0;cur_d_Bw0=d_Anext1;
		block2D_hybrid_coarsen_x2<<<grid, block,sh_size,stream0>>>(c0,c1, d_A0, d_Anext0, nx, ny,  nz/2, cur_d_Br0,cur_d_Bw0,1);
	 cudaStreamSynchronize(stream0);
	
	temp1= d_Anext0;
	d_Anext0=d_A0;
	d_A0=temp1;

	temp2= d_Anext1;
	d_Anext1=d_A1;
	d_A1=temp2;
	
	cudaSetDevice(0);
	 cudaStreamSynchronize(stream1);
	}
//ENDt_TIMER("kernel");
	
    CUERR // check and clear any existing errors
	
	
	cudaSetDevice(1);
	pb_SwitchToTimer(&timers, pb_TimerID_COPY);
	cudaMemcpy(h_Anext0, d_A0,size/2*sizeof(float), cudaMemcpyDeviceToHost);
	cudaSetDevice(0);
	cudaMemcpy(h_Anext0+size/2, d_A1+nx*ny,size/2*sizeof(float), cudaMemcpyDeviceToHost);
//	if (parameters->synchronizeGpu) cudaThreadSynchronize();
	cudaSetDevice(1);
	cudaFree(d_A0);
    cudaFree(d_Anext0);
	cudaFree (d_B0);
    cudaFree(d_Anext0);
//int byt;
//if (! (byt=memcmp(h_Anext0,h_B,size*sizeof(float))) ){printf("they are equal");}
//else printf("fuck!! they are not equal at byte =%d\n",byt);

#if KEY <0
for (int i=0;i<size;i++)
	printf("hA0[%d]=%f hnext[%d]=%f\n",i,h_B[i],i,h_Anext0[i]);
#endif
	if (parameters->outFile) {
		 pb_SwitchToTimer(&timers, pb_TimerID_IO);
		outputData(parameters->outFile,h_Anext0,nx,ny,nz);
		
	}
	pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
		
	free (h_A);
	free (h_Anext0);
	cudaSetDevice(0);
	cudaFree (d_B1);
	cudaFree(d_A1);
    cudaFree(d_Anext1);
	pb_SwitchToTimer(&timers, pb_TimerID_NONE);

	pb_PrintTimerSet(&timers);
	pb_FreeParameters(parameters);

	return 0;

}
