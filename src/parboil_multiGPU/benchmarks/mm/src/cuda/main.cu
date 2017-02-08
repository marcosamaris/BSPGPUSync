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
#include <pthread.h>


//pthread parameters
#define NUM_GPUS 4
void *multithreaded_matmul(void *parameter);

struct thread_params{

	std::vector<float> A;
	std::vector<float> BT;
	std::vector<float> C;
	int Arow, Bcol, Acol;
	int id;

};

struct thread_params param[NUM_GPUS];

pthread_barrier_t bar;


// I/O routines
extern bool readColMajorMatrixFile(const char *fn, int &nr_row, int &nr_col, std::vector<float>&v);
extern bool writeColMajorMatrixFile(const char *fn, int, int, std::vector<float>&);

extern "C"
void computeGold(float *, const float*, const float*, unsigned int, unsigned int, unsigned int);

int
main (int argc, char *argv[]) {

  struct pb_Parameters *params;
  struct pb_TimerSet timers;

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
  std::vector<float> matC(matArow*matBcol);



  pthread_t thread[NUM_GPUS];

   pthread_attr_t attr;

   int rc;

   long t;

   void *status;

	

//pthread_barrier_init(&bar, NULL, NUM_GPUS);




   /* Initialize and set thread detached attribute */

   pthread_attr_init(&attr);

   pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
 //pb_SwitchToTimer( &timers, pb_TimerID_GPU );
   for(t=0; t<NUM_GPUS; t++) {

     fprintf(stderr,"Main: creating thread %ld\n", t);

      param[t].id=t;
      param[t].A= matA;
      param[t].BT= matBT;
      param[t].C= matC;
      param[t].Arow=matArow;
      param[t].Bcol=matBcol;
      param[t].Acol=matAcol;

      rc = pthread_create(&thread[t], &attr,multithreaded_matmul,(void *) &param[t]);

      if (rc) {

         printf("ERROR; return code from pthread_create() is %d\n", rc);

         exit(-1);

         }

      }

	

    for(t=0; t<NUM_GPUS; t++) {

      rc = pthread_join(thread[t], &status);

      if (rc) {

         printf("ERROR; return code from pthread_join()  is %d\n", rc);

         exit(-1);

         }

      printf("Main: completed join with thread %ld having a status  of %ld\n",t,(long)status);

      }

cudaSetDevice(0);

for(t=0; t<NUM_GPUS; t++) 
	memcpy(&matC[t*C_sz/(sizeof(float)*NUM_GPUS)],&(param[t].C).front(),C_sz/NUM_GPUS);
    pb_SwitchToTimer(&timers, pb_TimerID_IO);
    writeColMajorMatrixFile(params->outFile,
	matArow, matBcol, matC); 

  //pb_SwitchToTimer(&timers, pb_TimerID_NONE);
  //double GPUtime = pb_GetElapsedTime(&(timers.timers[pb_TimerID_GPU]));
  //std::cout<< "GFLOPs = " << 2.* matArow * matBcol * matAcol/GPUtime/1e9 << std::endl;
  pb_PrintTimerSet(&timers);
  pb_FreeParameters(params);
  return 0;
}



void *multithreaded_matmul(void *paramt){

struct thread_params *parameters= (struct thread_params *)paramt;
cudaSetDevice(parameters->id);
 float *dA, *dB, *dC;
/*std::vector<float> A,BT,C;
A= parameters->A;
BT= parameters->BT;
C= parameters->C;*/
int matArow=parameters->Arow;
int matBcol=parameters->Bcol;
int matAcol=parameters->Acol;
//printf(" %d %d %d\n", matArow,matAcol,matBcol);
int A_sz= matArow * matAcol;
int B_sz= matAcol * matBcol;
int C_sz= matArow * matBcol;
  cudaMalloc((void**)&dA, A_sz*sizeof(float)/NUM_GPUS);
  cudaMalloc((void**)&dB, B_sz*sizeof(float));
  cudaMalloc((void**)&dC, C_sz*sizeof(float)/NUM_GPUS);
//printf("asize =%d bsiz=%d\n",(parameters->A).size(),(parameters->BT).size());
  CHECK_ERROR("mySgemm");
  std::vector<float> A(A_sz/NUM_GPUS);
int i;
    for (i=0;i<matAcol;i++)        
	memcpy(&A[i*matArow/NUM_GPUS],&(parameters->A)[i*matArow+(parameters->id)*matArow/NUM_GPUS],matArow*sizeof(float)/NUM_GPUS);
  cudaMemcpy(dB, &((parameters->BT).front()), B_sz*sizeof(float), cudaMemcpyHostToDevice); 
  CHECK_ERROR("mySgemm");
  
  cudaMemcpy(dA, &A.front(), A_sz*sizeof(float)/NUM_GPUS, cudaMemcpyHostToDevice); 
  CHECK_ERROR("mySgemm");
  regtileSgemm('N', 'T', matArow/NUM_GPUS, matBcol, matAcol, 1.0f, \
      dA, matArow/NUM_GPUS, dB, matBcol, 0.0f, dC, matArow/NUM_GPUS);

  //cudaMemcpy(&(parameters->C)[(parameters->id)*C_sz/(NUM_GPUS)], dC, C_sz*sizeof(float)/NUM_GPUS, cudaMemcpyDeviceToHost);
  cudaMemcpy(&(parameters->C).front(), dC, C_sz*sizeof(float)/NUM_GPUS, cudaMemcpyDeviceToHost);


  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

pthread_exit((void*)&parameters->id);
}
