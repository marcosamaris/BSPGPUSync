/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
 
 /*
 * This sample implements 64-bin histogram calculation
 * of arbitrary-sized 8-bit data array
 */

// Utility and system includes
//#include <shrUtils.h>
//#include <shrQATest.h>
//#include <cuda_runtime.h>
//#include <cutil_inline.h>

// project include
#include "histogram_common.h"

const int numRuns = 16;

static char *sSDKsample = "[histogram]\0";

int main(int argc, char **argv)
{
    uchar *h_Data;
    uint  *h_HistogramCPU, *h_HistogramGPU,*h_HistogramGPU2 ;
    uchar *d_Data,*d_Data2;
    uint  *d_Histogram,*d_Histogram2;
    uint hTimer;
    int PassFailFlag = 1;
    uint byteCount = 64 * 1048576;
    uint uiSizeMult = 1;

            h_Data         = (uchar *)malloc(byteCount);
            h_HistogramCPU = (uint  *)malloc(HISTOGRAM256_BIN_COUNT * sizeof(uint));
            h_HistogramGPU = (uint  *)malloc(HISTOGRAM256_BIN_COUNT * sizeof(uint));
            h_HistogramGPU2 = (uint  *)malloc(HISTOGRAM256_BIN_COUNT * sizeof(uint));

            srand(2009);
            for(uint i = 0; i < byteCount; i++) 
                h_Data[i] = rand() % 256;
cudaSetDevice(2);
            ( cudaMalloc((void **)&d_Data, byteCount/2  ) );
            ( cudaMalloc((void **)&d_Histogram, HISTOGRAM256_BIN_COUNT * sizeof(uint)  ) );
            ( cudaMemcpy(d_Data, h_Data, byteCount/2, cudaMemcpyHostToDevice) );
cudaSetDevice(3);
            ( cudaMalloc((void **)&d_Data2, byteCount/2  ) );
            ( cudaMalloc((void **)&d_Histogram2, HISTOGRAM256_BIN_COUNT * sizeof(uint)  ) );
            ( cudaMemcpy(d_Data2, h_Data+byteCount/2, byteCount/2, cudaMemcpyHostToDevice) );
uint PARTIAL_HISTOGRAM256_COUNT = 120;
//uint *d_PartialHistograms;
//uint *d_PartialHistograms2;


    {
//		 initHistogram256(d_PartialHistograms,PARTIAL_HISTOGRAM256_COUNT);
//            initHistogram256(PARTIAL_HISTOGRAM256_COUNT);
 uint *d_PartialHistograms;
 uint *d_PartialHistograms2;

cudaEvent_t start1, stop1, start2, stop2; 

//Internal memory allocation
cudaSetDevice(2);
    ( cudaMalloc((void **)&d_PartialHistograms, PARTIAL_HISTOGRAM256_COUNT * HISTOGRAM256_BIN_COUNT * sizeof(uint)) );
cudaSetDevice(3);
    ( cudaMalloc((void **)&d_PartialHistograms2, PARTIAL_HISTOGRAM256_COUNT * HISTOGRAM256_BIN_COUNT * sizeof(uint)) );

//Internal memory deallocation                                                

            for(int iter = -1; iter < numRuns; iter++){
                //iter == -1 -- warmup iteration
                if(iter == 0){
cudaSetDevice(2);
cudaEventCreate(&start1); 
cudaEventCreate(&stop1);
cudaEventRecord(start1, 0);
                    ( cudaDeviceSynchronize() );
cudaSetDevice(3);
cudaEventCreate(&start2); 
cudaEventCreate(&stop2);
cudaEventRecord(start2, 0);
                    ( cudaDeviceSynchronize() );
                }

//                histogram256(d_Histogram, d_Data, byteCount);
cudaSetDevice(2);
	  histogram256(d_Histogram,d_PartialHistograms, d_Data, byteCount/2,PARTIAL_HISTOGRAM256_COUNT);
        //            ( cudaDeviceSynchronize() );
cudaSetDevice(3);
	  histogram256(d_Histogram2,d_PartialHistograms2, d_Data2, byteCount/2,PARTIAL_HISTOGRAM256_COUNT);
//	  histogram256(d_Histogram, d_Data, byteCount,PARTIAL_HISTOGRAM256_COUNT);
            }

cudaSetDevice(2);
            ( cudaDeviceSynchronize() );
	    cudaEventRecord(stop1, 0); cudaEventSynchronize(stop1);
cudaSetDevice(3);
            ( cudaDeviceSynchronize() );
	    cudaEventRecord(stop2, 0); cudaEventSynchronize(stop2);
cudaSetDevice(2);
                ( cudaMemcpy(h_HistogramGPU, d_Histogram, HISTOGRAM256_BIN_COUNT * sizeof(uint), cudaMemcpyDeviceToHost) );
cudaSetDevice(3);
                ( cudaMemcpy(h_HistogramGPU2, d_Histogram2, HISTOGRAM256_BIN_COUNT * sizeof(uint), cudaMemcpyDeviceToHost) );

                histogram256CPU(
                    h_HistogramCPU,
                    h_Data,
                    byteCount
                );
	for (int i=0;i<HISTOGRAM256_BIN_COUNT;i++)
	h_HistogramGPU[i]+=h_HistogramGPU2[i];

float elapsedTime2; cudaEventElapsedTime(&elapsedTime2, start2, stop2);
printf("time2=%f\n",  elapsedTime2/numRuns);
cudaSetDevice(2);
float elapsedTime1; cudaEventElapsedTime(&elapsedTime1, start1, stop1);
printf("time1 =%f \n", elapsedTime1/numRuns);

                for(uint i = 0; i < HISTOGRAM256_BIN_COUNT; i++){
	                   if(h_HistogramGPU[i] != h_HistogramCPU[i]) PassFailFlag = 0;
//printf("h_HistogramGPU[%d]=%d h_HistogramCPU[%d]=%d \n",i,h_HistogramGPU[i],i, h_HistogramCPU[i]);

		}
cudaSetDevice(2);
    ( cudaFree(d_PartialHistograms) );                                        
cudaSetDevice(3);
    ( cudaFree(d_PartialHistograms2) );                                        
    //        closeHistogram256();
	  //closeHistogram256(d_PartialHistograms);
    }

cudaSetDevice(2);
        ( cudaFree(d_Histogram) );
        ( cudaFree(d_Data) );
cudaSetDevice(3);
        ( cudaFree(d_Histogram2) );
        ( cudaFree(d_Data2) );
        free(h_HistogramGPU);
        free(h_HistogramGPU2);
        free(h_HistogramCPU);
        free(h_Data);

}
