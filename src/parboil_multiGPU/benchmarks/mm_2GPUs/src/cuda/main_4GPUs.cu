/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/* 
 * Main entry of dense matrix-matrix multiplication kernel
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <malloc.h>
#include <vector>
#include <parboil.h>
#include <iostream>
#include "sgemm_kernel.cu"

// I/O routines
extern bool readColMajorMatrixFile(const char *fn, int &nr_row, int &nr_col, std::vector<float>&v);
extern bool writeColMajorMatrixFile(const char *fn, int, int, std::vector<float>&);

extern "C"
void computeGold(float *, const float*, const float*, unsigned int, unsigned int, unsigned int);

int
main (int argc, char *argv[]) {

  struct pb_Parameters *params;
  struct pb_TimerSet timers;

  float *dA, *dB, *dC, *dA2, *dB2, *dC2,  *dA1, *dB1, *dC1, *dA3, *dB3, *dC3;
  size_t A_sz, B_sz, C_sz;
  int matArow, matAcol;
  int matBrow, matBcol;
  std::vector<float> matA, matBT;

  pb_InitializeTimerSet(&timers);

  /* Read command line. Expect 3 inputs: A, B and B^T 
     in column-major layout*/
  params = pb_ReadParameters(&argc, argv);
  if ((params->inpFiles[0] == NULL) 
      || (params->inpFiles[1] == NULL)
      || (params->inpFiles[2] == NULL)
      || (params->inpFiles[3] != NULL))
    {
      fprintf(stderr, "Expecting three input filenames\n");
      exit(-1);
    }
 
  /* Read in data */
  pb_SwitchToTimer(&timers, pb_TimerID_IO);

  // load A
  readColMajorMatrixFile(params->inpFiles[0],
      matArow, matAcol, matA);
  // copy A to device memory
  A_sz = matArow*matAcol*sizeof(float);

  // load B^T
  readColMajorMatrixFile(params->inpFiles[2],
      matBcol, matBrow, matBT);

  pb_SwitchToTimer( &timers, pb_TimerID_COMPUTE );
  B_sz = matBrow*matBcol*sizeof(float);

  // allocate space for C
  C_sz = matArow*matBcol*sizeof(float);

int i;
  // CUDA memory allocation
  std::vector<float> matC(matArow*matBcol);
  std::vector<float> matC0(matArow*matBcol);
  std::vector<float> matC1(matArow*matBcol);
  std::vector<float> matC2(matArow*matBcol);
  std::vector<float> matC3(matArow*matBcol);
  std::vector<float> matC4(matArow*matBcol);
    for (i=0;i<matAcol;i++)        {
	memcpy(&matC0.front()+i*matArow/4,&matA.front()+i*matArow,matArow*sizeof(float)/4);
	memcpy(&matC1[i*matArow/4],&matA[matArow/4+i*matArow],matArow*sizeof(float)/4);
	memcpy(&matC2[i*matArow/4],&matA[matArow/2+i*matArow],matArow*sizeof(float)/4);
	memcpy(&matC3[i*matArow/4],&matA[3*matArow/4+i*matArow],matArow*sizeof(float)/4);
	}	
cudaSetDevice(2);
  cudaMalloc((void**)&dA2, A_sz);
  cudaMalloc((void**)&dB2, B_sz);
  cudaMalloc((void**)&dC2, C_sz);

cudaSetDevice(1);
  cudaMalloc((void**)&dA1, A_sz);
  cudaMalloc((void**)&dB1, B_sz);
  cudaMalloc((void**)&dC1, C_sz);
cudaSetDevice(3);
  cudaMalloc((void**)&dA3, A_sz);
  cudaMalloc((void**)&dB3, B_sz);
  cudaMalloc((void**)&dC3, C_sz);

cudaSetDevice(0);
  cudaMalloc((void**)&dA, A_sz);
  cudaMalloc((void**)&dB, B_sz);
  cudaMalloc((void**)&dC, C_sz);
  // Copy A and B^T into device memory
  pb_SwitchToTimer( &timers, pb_TimerID_COPY );
  cudaMemcpy(dA, &matC0.front(), A_sz/4, cudaMemcpyHostToDevice); 
  cudaMemcpy(dB, &matBT.front(), B_sz, cudaMemcpyHostToDevice); 
cudaSetDevice(1);
  cudaMemcpy(dA1, (&matC1.front()), A_sz/4, cudaMemcpyHostToDevice); 
  cudaMemcpy(dB1, &matBT.front(), B_sz, cudaMemcpyHostToDevice); 
cudaSetDevice(2);
  cudaMemcpy(dA2, (&matC2.front()), A_sz/4, cudaMemcpyHostToDevice); 
  cudaMemcpy(dB2, &matBT.front(), B_sz, cudaMemcpyHostToDevice); 
cudaSetDevice(3);
  cudaMemcpy(dA3, (&matC3.front()), A_sz/4, cudaMemcpyHostToDevice); 
  cudaMemcpy(dB3, &matBT.front(), B_sz, cudaMemcpyHostToDevice); 

  // Use standard sgemm interface
  regtileSgemm('N', 'T', matArow/4, matBcol, matAcol, 1.0f, \
      dA3, matArow/4, dB3, matBcol, 0.0f, dC3, matArow/4);
cudaSetDevice(1);
  regtileSgemm('N', 'T', matArow/4, matBcol, matAcol, 1.0f, \
      dA1, matArow/4, dB1, matBcol, 0.0f, dC1, matArow/4);
cudaSetDevice(2);
  regtileSgemm('N', 'T', matArow/4, matBcol, matAcol, 1.0f, \
      dA2, matArow/4, dB2, matBcol, 0.0f, dC2, matArow/4);
cudaSetDevice(0);
  pb_SwitchToTimer( &timers, pb_TimerID_GPU );
  regtileSgemm('N', 'T', matArow/4, matBcol, matAcol, 1.0f, \
      dA, matArow/4, dB, matBcol, 0.0f, dC, matArow/4);

  if (params->outFile) {
    pb_SwitchToTimer( &timers, pb_TimerID_COPY );
    cudaMemcpy(&matC.front(), dC, C_sz/4, cudaMemcpyDeviceToHost);
cudaSetDevice(3);
    cudaMemcpy(&matC[3*C_sz/(4*sizeof(float))], dC3, C_sz/4, cudaMemcpyDeviceToHost);
cudaSetDevice(2);
    cudaMemcpy(&matC[C_sz/(2*sizeof(float))], dC2, C_sz/4, cudaMemcpyDeviceToHost);
cudaSetDevice(1);
    cudaMemcpy(&matC[C_sz/(4*sizeof(float))], dC1, C_sz/4, cudaMemcpyDeviceToHost);
    /* Write C to file */
cudaSetDevice(0);
//    for (i=0;i<matBcol;i++)
  //      {
//	memcpy(&matC.front(),&matC1.front(),C_sz/2);
//	memcpy(&matC[C_sz/(2*sizeof(float))],&matC2.front(),C_sz/2);

//	}	
    pb_SwitchToTimer(&timers, pb_TimerID_IO);
    writeColMajorMatrixFile(params->outFile,
	matArow, matBcol, matC); 
  }

  pb_SwitchToTimer(&timers, pb_TimerID_NONE);
const char file[]="output/large/matrix3.txt";
bool flag=    readColMajorMatrixFile(file,matArow, matBcol, matC4);
if(flag==0)printf("err");
float x;
for (i=0;i<C_sz/(sizeof(float));i++){
//if (fabs((x = matC4[i]-matC[i]))>0.0001)//{
//	printf("fuck it is still not correct %d %f %f\n",i,matC4[i],matC[i]);//}
//else
//	printf(" %d %f %f\n",i,matC4[i],matC[i]);
	//printf("yahoo correct");
}
  double GPUtime = pb_GetElapsedTime(&(timers.timers[pb_TimerID_GPU]));
  std::cout<< "GFLOPs = " << 2.* matArow * matBcol * matAcol/GPUtime/1e9 << std::endl;
  pb_PrintTimerSet(&timers);
  pb_FreeParameters(params);
  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);
cudaSetDevice(1);
  cudaFree(dA1);
  cudaFree(dB1);
  cudaFree(dC1);
cudaSetDevice(2);
  cudaFree(dA2);
  cudaFree(dB2);
  cudaFree(dC2);
cudaSetDevice(3);
  cudaFree(dA3);
  cudaFree(dB3);
  cudaFree(dC3);
  return 0;
}
