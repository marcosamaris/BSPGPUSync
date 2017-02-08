/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/*
   Implementing Breadth first search on CUDA using algorithm given in DAC'10
   paper "An Effective GPU Implementation of Breadth-First Search"

   Copyright (c) 2010 University of Illinois at Urbana-Champaign. 
   All rights reserved.

   Permission to use, copy, modify and distribute this software and its documentation for 
   educational purpose is hereby granted without fee, provided that the above copyright 
   notice and this permission notice appear in all copies of this software and that you do 
   not sell the software.

   THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS, IMPLIED OR 
   OTHERWISE.

Author: Lijiuan Luo (lluo3@uiuc.edu)
Revised for Parboil 2 Benchmark Suite by: Geng Daniel Liu (gengliu2@illinois.edu)
 */
#include <errno.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <parboil.h>
#include <deque>
#include <iostream>
#include <pthread.h>
#include <stdint.h>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
//pthread CPU affinity
cpu_set_t cpuset;
cudaStream_t stream1,stream2;

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


#define MAX_THREADS_PER_BLOCK 512
#define NUM_SM 14 //the number of Streaming Multiprocessors; 15 for Fermi architecture 30 for G280 at the moment of this document
#define NUM_BIN 8 //the number of duplicated frontiers used in BFS_kernel_multi_blk_inGPU
#define EXP 3 // EXP = log(NUM_BIN), assuming NUM_BIN is still power of 2 in the future architecture
//using EXP and shifting can speed up division operation 
#define MOD_OP 7 // This variable is also related with NUM_BIN; may change in the future architecture;
//using MOD_OP and "bitwise and" can speed up mod operation
int no_of_nodes; //the number of nodes in the graph
int edge_list_size;//the number of edges in the graph
FILE *fp;

typedef int2 Node;
typedef int2 Edge;
#define NUM_GPUS 2
#include "kernel.cu"
//Somehow "cudaMemset" does not work. So I use cudaMemcpy of constant variables for initialization
const int h_top = 1;
const int zero = 0;

Node *d_graph_nodes,*d_graph_nodes2;
Edge *d_graph_edges,*d_graph_edges2;
int *g_color1,*g_color2;
int *g_cost1,*g_cost2;
int *g_cost3,*g_cost4;
int *g_q1,*g_q3;
int *g_q2,*g_q4;
int *tail1,*tail2;
int *h_cost,*color;
int *num_t1,*num_t2;
int *dup_q1,*dup_q2;
int *dup_q3,*dup_q4;
void *multithreaded_bfs(void *parameter);
struct thread_params{
	Node *h_graph_nodes;
	Edge *h_graph_edges;
	int source;
	int id;
};
struct thread_params param;
pthread_barrier_t bar;

void onethread_bfs(struct thread_params *parameter);
void runGPU(int argc, char** argv);
////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 

	{
	//printf(" device 1 \n");
	cudaSetDevice(1);
	cudaDeviceEnablePeerAccess(0,0);
//	cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
	cudaSetDevice(0);
	cudaDeviceEnablePeerAccess(1,0);
//	cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
	no_of_nodes=0;
	edge_list_size=0;
//	cudaSetDevice(0);
	runGPU(argc,argv);

}
///////////////////////////////
///////////////////////////////
//FUNCTION:only run GPU version 
////////////////////////////////////////////
void runGPU( int argc, char** argv) 
{

	struct pb_Parameters *params;
	struct pb_TimerSet timers;



	pb_InitializeTimerSet(&timers);
	params = pb_ReadParameters(&argc, argv);
	if ((params->inpFiles[0] == NULL) || (params->inpFiles[1] != NULL))
	{
		fprintf(stderr, "Expecting one input filename\n");
		exit(-1);
	}

	pb_SwitchToTimer(&timers, pb_TimerID_IO);
	//printf("Reading File\n");
	//Read in Graph from a file
	fp = fopen(params->inpFiles[0],"r");
	if(!fp)
	{
		printf("Error Reading graph file\n");
		return;
	}
	int source;

	fscanf(fp,"%d",&no_of_nodes);
	// allocate host memory
	Node* h_graph_nodes = (Node*) malloc(sizeof(Node)*no_of_nodes);
	color = (int*) malloc(sizeof(int)*no_of_nodes);
	int start, edgeno;   
	// initalize the memory
	for( unsigned int i = 0; i < no_of_nodes; i++) 
	{
		fscanf(fp,"%d %d",&start,&edgeno);
		h_graph_nodes[i].x = start;
		h_graph_nodes[i].y = edgeno;
		color[i]=WHITE;
	}
	//read the source node from the file
	fscanf(fp,"%d",&source);
	fscanf(fp,"%d",&edge_list_size);
	int id,cost;
	Edge* h_graph_edges = (Edge*) malloc(sizeof(Edge)*edge_list_size);
	for(int i=0; i < edge_list_size ; i++)
	{
		fscanf(fp,"%d",&id);
		fscanf(fp,"%d",&cost);
		h_graph_edges[i].x = id;
		h_graph_edges[i].y = cost;
	}
	if(fp)
		fclose(fp);    

	//	printf("Read File\n");

	// allocate mem for the result on host side
	h_cost = (int*) malloc( sizeof(int)*no_of_nodes);
	for(int i = 0; i < no_of_nodes; i++){
		h_cost[i] = INF;
	}
	h_cost[source] = 0;

	pb_SwitchToTimer(&timers, pb_TimerID_COPY);

	int * temp = NULL;
	( cudaMalloc( (void**) &temp, sizeof(int)*no_of_nodes) );
	( cudaFree( temp) );
	//	unsigned int copy_timer = 0;
	//	cutilCheckError(cutCreateTimer(&copy_timer));
	//	cutilCheckError(cutStartTimer(copy_timer));

	//Copy the Node list to device memory
	/*Node *d_graph_nodes,*d_graph_nodes2;
	  Edge *d_graph_edges,*d_graph_edges2;
	  int *d_color,*d_color2;
	  int *d_cost,*d_cost2;
	  int *d_q1,*d_q3;
	  int *d_q2,*d_q4;
	  int *tail,*tail2;*/

//	for (int i=0;i<NUM_GPUS;i++){
		param.h_graph_nodes=h_graph_nodes;
		param.h_graph_edges=h_graph_edges;
		param.source=source;
//	}
	/*
	   param[1].d_graph_nodes=&d_graph_nodes;
	   param[1].d_graph_nodes2=&d_graph_nodes2;
	   param[1].d_graph_edges=&d_graph_edges;
	   param[1].d_graph_edges2=&d_graph_edges2;
	   param[1].d_color=&d_color;
	   param[1].d_color2=&d_color2;
	   param[1].d_cost=&d_cost;
	   param[1].d_cost2=&d_cost2;
	   param[1].d_q1=&d_q1;
	   param[1].d_q3=&d_q3;
	   param[1].d_q2=&d_q2;
	   param[1].d_q4=&d_q4;
	   param[1].tail=&tail;
	   param[1].tail2=&tail2;
	 */

	pb_SwitchToTimer(&timers, pb_TimerID_GPU);

/*	pthread_t thread[NUM_GPUS];
	pthread_attr_t attr;
	int rc;
	long t;
	void *status;
		CPU_SET(3, &cpuset);
		CPU_SET(4, &cpuset);
		CPU_SET(1, &cpuset);
		CPU_SET(2, &cpuset);

	pthread_barrier_init(&bar, NULL, NUM_GPUS);

	// Initialize and set thread detached attribute 
	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
	for(t=0; t<NUM_GPUS; t++) {
		fprintf(stderr,"Main: creating thread %ld\n", t);
		param[t].id=t;
		rc = pthread_create(&thread[t], &attr,multithreaded_bfs,(void *) &param[t]);
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

	pthread_barrier_destroy(&bar);
*/

	onethread_bfs(&param);
	pb_SwitchToTimer(&timers, pb_TimerID_COPY);

	//Store the result into a file
	pb_SwitchToTimer(&timers, pb_TimerID_IO);
	FILE *fp = fopen(params->outFile,"w");
	fprintf(fp, "%d\n", no_of_nodes);
	for(int i=0;i<no_of_nodes;i++)
		fprintf(fp,"%d %d\n",i,h_cost[i]);
	fclose(fp);
	//printf("Result stored in %s\n", params->outFile);

	// cleanup memory
	free( h_graph_nodes);
	free( h_graph_edges);
	free( color);
	free( h_cost);

	pb_SwitchToTimer(&timers, pb_TimerID_NONE);
	pb_PrintTimerSet(&timers);
	pb_FreeParameters(params);
}




inline bool  BFS_GPUone( Node * d_graph_nodes,Edge * d_graph_edges,
		int * d_color, int * d_cost, int * d_q1, int * d_q2, int * tail, int source, int & global_kt){
	int num_of_blocks; 
	int num_of_threads_per_block;
	/*cudaFuncSetCacheConfig(BFS_in_GPU_kernel, cudaFuncCachePreferShared);    
	cudaFuncSetCacheConfig(BFS_kernel_multi_blk_inGPU, cudaFuncCachePreferShared);    
	cudaFuncSetCacheConfig(BFS_kernel1, cudaFuncCachePreferShared);    
	cudaFuncSetCacheConfig(BFS_kernel3, cudaFuncCachePreferShared);    
	cudaFuncSetCacheConfig(BFS_kernel2, cudaFuncCachePreferShared);    
	cudaFuncSetCacheConfig(BFS_kernel4, cudaFuncCachePreferShared);    
	*/

	/*int *zerro;
	  ( cudaMalloc( (void**) &zerro, sizeof(int)));
	  (cudaMemset(zerro,0,sizeof(int)));
	 */
	(cudaMemcpy(tail,&h_top,sizeof(int),cudaMemcpyHostToDevice));
	(cudaMemcpy(&d_cost[source],&zero,sizeof(int),cudaMemcpyHostToDevice));

	( cudaMemcpy( &d_q1[0], &source, sizeof(int), cudaMemcpyHostToDevice) );
	int num_t;//number of threads
	int k=0;//BFS level index
	//	cudaHostRegister(&num_t,sizeof(int),cudaHostRegisterDefault);
	//whether or not to adjust "k", see comment on "BFS_kernel_multi_blk_inGPU" for more details 
	int * switch_kd;
	( cudaMalloc( (void**) &switch_kd, sizeof(int)));
	int * num_td;//number of threads
	( cudaMalloc( (void**) &num_td, sizeof(int)));

	//whether to stay within a kernel, used in "BFS_kernel_multi_blk_inGPU"
	bool *stay;
	( cudaMalloc( (void**) &stay, sizeof(bool)));
	int switch_k;

	//max number of frontier nodes assigned to a block
	int * max_nodes_per_block_d;
	( cudaMalloc( (void**) &max_nodes_per_block_d, sizeof(int)));
#ifdef DIS_COMP
	int max_nodes_per_block;
#endif
	int *global_kt_d;
	( cudaMalloc( (void**) &global_kt_d, sizeof(int)));
	(cudaMemcpy(global_kt_d,&global_kt, sizeof(int),cudaMemcpyHostToDevice));
	cudaError_t err;

	do
	{
	//	cudaDeviceSynchronize();
		//STARTt_TIMER;
		( cudaMemcpy( &num_t, tail, sizeof(int), cudaMemcpyDeviceToHost) );
		//		(cudaMemcpy(tail,&zero,sizeof(int),cudaMemcpyHostToDevice));
		//(cudaMemset(tail,0,sizeof(int)));

	//	(cudaMemcpy(&global_kt,global_kt_d, sizeof(int),cudaMemcpyDeviceToHost));
//		printf("BFS_LEVEL=%d elements=%d\n",global_kt,num_t);
		//ENDt_TIMER("transfer");

		if(num_t == 0){//frontier is empty
			(cudaFree(stay));
			(cudaFree(switch_kd));
			(cudaFree(num_td));
			(cudaMemcpy(&global_kt,global_kt_d, sizeof(int),cudaMemcpyDeviceToHost));
			//			cudaHostUnregister(&num_t);
			return false;
		}

		num_of_blocks = 1;
		num_of_threads_per_block = num_t;
		if(num_of_threads_per_block <NUM_BIN)
			num_of_threads_per_block = NUM_BIN;
		if(num_t>MAX_THREADS_PER_BLOCK)
		{
			num_of_blocks = (int)ceil(num_t/(double)MAX_THREADS_PER_BLOCK); 
			num_of_threads_per_block = MAX_THREADS_PER_BLOCK;
		}
		if(num_of_blocks == 1)//will call "BFS_in_GPU_kernel" 
			num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
		if(num_of_blocks >1 && num_of_blocks <= NUM_SM)// will call "BFS_kernel_multi_blk_inGPU"
			num_of_blocks = NUM_SM;

		//assume "num_of_blocks" can not be very large
		dim3  grid( num_of_blocks, 1, 1);
		dim3  threads( num_of_threads_per_block, 1, 1);
		if(k%2 == 0){
			if(num_of_blocks == 1){ 
				cudaSetDevice(0);
				(cudaMemcpy(tail,&zero,sizeof(int),cudaMemcpyHostToDevice));
				//STARTt_TIMER;
				BFS_in_GPU_kernel<<< grid, threads >>>(d_q1,d_q2, d_graph_nodes, 
						d_graph_edges, d_color, d_cost,num_t , tail,GRAY0,k );
				cudaDeviceSynchronize();
			//	ENDt_TIMER("kernelll1");
			}
			else if(num_of_blocks <= NUM_SM){
				cudaSetDevice(0);
#ifdef DIS_COMP
				max_nodes_per_block =(int) ceil(float(num_t)/NUM_SM);
				(cudaMemcpy(max_nodes_per_block_d,
					    &max_nodes_per_block,sizeof(int), cudaMemcpyHostToDevice));
#endif
				//		(cudaMemcpy(num_td,&num_t,sizeof(int),
				//			cudaMemcpyHostToDevice));
				(cudaMemcpy(num_td,tail,sizeof(int),cudaMemcpyDeviceToDevice));
				(cudaMemcpy(tail,&zero,sizeof(int),cudaMemcpyHostToDevice));

				//STARTt_TIMER;
				BFS_kernel_multi_blk_inGPU
					<<< grid, threads >>>(d_q1,d_q2, d_graph_nodes, 
							d_graph_edges, d_color, d_cost, num_td, tail,GRAY0,k,
							switch_kd, max_nodes_per_block_d, global_kt_d);
				cudaDeviceSynchronize();
			//	ENDt_TIMER("kernell1");
				(cudaMemcpy(&switch_k,switch_kd, sizeof(int),
					    cudaMemcpyDeviceToHost));
				if(!switch_k){
					k--;
				}
			}
			else{  
				cudaSetDevice(0);
				(cudaMemcpyAsync(tail1,&zero,sizeof(int),cudaMemcpyHostToDevice,stream1));
				cudaSetDevice(1);
				(cudaMemcpyAsync(tail2,&zero,sizeof(int),cudaMemcpyHostToDevice,stream2));
				cudaMemcpyPeerAsync(g_color2,1,g_color1,0,no_of_nodes*sizeof(int),stream2);
				cudaMemcpyPeerAsync(g_q3,1,g_q1,0,num_t*sizeof(int),stream2);	
				cudaMemcpyPeerAsync(g_cost3,1,g_cost1,0,no_of_nodes*sizeof(int),stream2);
			//	STARTt_TIMER;

				if (num_of_blocks%2==0){
					grid= dim3((num_of_blocks/2), 1, 1);

					BFS_kernel4<<< grid, threads,0,stream2 >>>(g_q3,g_q4, d_graph_nodes2,
							d_graph_edges2, g_color2, g_cost3, num_t, tail2, GRAY0,k);
					(cudaMemcpyAsync(num_t2,tail2,sizeof(int),cudaMemcpyDeviceToHost,stream2));
					cudaMemcpyPeerAsync(g_cost2,0,g_cost3,1,no_of_nodes*sizeof(int),stream2);
					cudaStreamSynchronize(stream2);
					cudaDeviceSynchronize();
				cudaSetDevice(0);
					BFS_kernel3<<< grid, threads,0, stream1 >>>(g_q1,g_q2, d_graph_nodes,
							d_graph_edges, g_color1, g_cost1, num_t, tail1, GRAY0,k);

				/*	err=cudaGetLastError();
					if (err != cudaSuccess){
						printf("kernel error= %d ",err);
						exit(-1);
					}
				*/
					(cudaMemcpyAsync(num_t1,tail1,sizeof(int),cudaMemcpyDeviceToHost,stream1));
					cudaMemcpyPeerAsync(g_q2+(*num_t1),0,g_q4,1,(*num_t2)*sizeof(int),stream1);
					cudaStreamSynchronize(stream1);
					cudaDeviceSynchronize();

					int numt=(*num_t1)+(*num_t2);
					cudaMemcpy(tail1,&numt,sizeof(int),cudaMemcpyHostToDevice);
					grid= dim3((int)ceil(*num_t2/(double)512), 1, 1);
					cost_adjust<<<grid,threads >>>(*num_t2,*num_t1,g_q2,g_cost2,g_cost1,g_color1,GRAY0);
					cudaDeviceSynchronize();
		
					//duplicate_detection<<<grid1,threads>>>(num_t2,*num_t1,g_cost2,g_cost1,tail,d_q2,g_color1,GRAY0);
					//cudaDeviceSynchronize();
				/*
					err=cudaGetLastError();
					if (err != cudaSuccess){
						printf("duplicate detectin kernel error= %d ",err);
						exit(-1);
					}*/

				}
				else{

					grid= dim3((int)ceil(num_of_blocks/(double)2), 1, 1);
					BFS_kernel2<<< grid, threads,0,stream2 >>>(g_q3,g_q4, d_graph_nodes2,
							d_graph_edges2, g_color2, g_cost3, num_t, tail2, GRAY0,k);
					cudaDeviceSynchronize();
					(cudaMemcpyAsync(num_t2,tail2,sizeof(int),cudaMemcpyDeviceToHost,stream2));
					cudaMemcpyPeerAsync(g_cost2,0,g_cost3,1,no_of_nodes*sizeof(int),stream2);
					cudaStreamSynchronize(stream2);
					cudaDeviceSynchronize();
				cudaSetDevice(0);
					BFS_kernel1<<< grid, threads,0,stream1 >>>(g_q1,g_q2, d_graph_nodes,
							d_graph_edges, d_color, d_cost, num_t, tail1, GRAY0,k);
					/*
					err=cudaGetLastError();
					if (err != cudaSuccess){
						printf(" 21 kernel error= %d ",err);
						exit(-1);
					}*/
					(cudaMemcpyAsync(num_t1,tail1,sizeof(int),cudaMemcpyDeviceToHost,stream1));
					cudaMemcpyPeerAsync(g_q2+*num_t1,0,g_q4,1,(*num_t2)*sizeof(int),stream1);
					cudaStreamSynchronize(stream1);
					cudaDeviceSynchronize();
					int numt=*num_t1+*num_t2;
					printf("1 num_t1=%d num_t2=%d \n",*num_t1,*num_t2);
					cudaMemcpy(tail1,&numt,sizeof(int),cudaMemcpyHostToDevice);
					grid= dim3((int)ceil(*num_t2/(double)512), 1, 1);
					cost_adjust<<<grid,threads >>>(*num_t2,*num_t1,g_q2,g_cost2,g_cost1,g_color1,GRAY0);
					cudaDeviceSynchronize();

					//duplicate_detection<<<grid1,threads>>>(*num_t2,*num_t1,g_cost2,g_cost1,tail,d_q2,g_color1,GRAY0);
					//cudaDeviceSynchronize();
					/*
					err=cudaGetLastError();
					if (err != cudaSuccess){
						printf("duplicate detectin kernel error= %d ",err);
						exit(-1);
					}*/

				}

				//ENDt_TIMER("kernel1");
			}
		}
		else{
			if(num_of_blocks == 1){  
				(cudaMemcpy(tail,&zero,sizeof(int),cudaMemcpyHostToDevice));
			//	STARTt_TIMER;
				BFS_in_GPU_kernel<<< grid, threads >>>(d_q2,d_q1, d_graph_nodes, 
						d_graph_edges, d_color, d_cost, num_t, tail,GRAY1,k);
				cudaDeviceSynchronize();
			//	ENDt_TIMER("kernelll2");
			}
			else if(num_of_blocks <= NUM_SM){
#ifdef DIS_COMP
				max_nodes_per_block =(int) ceil(float(num_t)/NUM_SM);
				(cudaMemcpy(max_nodes_per_block_d,
					    &max_nodes_per_block,sizeof(int), cudaMemcpyHostToDevice));
#endif
				//		(cudaMemcpy(num_td,&num_t,sizeof(int),
				//			cudaMemcpyHostToDevice));
				(cudaMemcpy(num_td,tail,sizeof(int),cudaMemcpyDeviceToDevice));
				(cudaMemcpy(tail,&zero,sizeof(int),cudaMemcpyHostToDevice));
			//	STARTt_TIMER;
				BFS_kernel_multi_blk_inGPU
					<<< grid, threads >>>(d_q2,d_q1, d_graph_nodes, 
							d_graph_edges, d_color, d_cost, num_td, tail,GRAY1,k,
							switch_kd, max_nodes_per_block_d, global_kt_d);
				cudaDeviceSynchronize();
			//	ENDt_TIMER("kernell2");
				(cudaMemcpy(&switch_k,switch_kd, sizeof(int),
					    cudaMemcpyDeviceToHost));
				if(!switch_k){
					k--;
				}
			}
			else{
				cudaSetDevice(0);
				(cudaMemcpyAsync(tail1,&zero,sizeof(int),cudaMemcpyHostToDevice,stream1));
				cudaSetDevice(1);
				(cudaMemcpyAsync(tail2,&zero,sizeof(int),cudaMemcpyHostToDevice,stream2));
				cudaMemcpyPeerAsync(g_color2,1,g_color1,0,no_of_nodes*sizeof(int),stream2);
				cudaMemcpyPeerAsync(g_q4,1,g_q2,0,num_t*sizeof(int),stream2);	
				cudaMemcpyPeerAsync(g_cost3,1,g_cost1,0,no_of_nodes*sizeof(int),stream2);
					cudaDeviceSynchronize();
					/*err=cudaGetLastError();
					if (err != cudaSuccess){
						printf("43 kernel error= %d ",err);
						exit(-1);
					}*/
printf("numt=%d\n",num_t);
			//	STARTt_TIMER;
				if  (num_of_blocks%2==0){
					grid= dim3((num_of_blocks/2), 1, 1);
					BFS_kernel4<<< grid, threads,0,stream2 >>>(g_q4,g_q3, d_graph_nodes2,
							d_graph_edges2, g_color2, g_cost3, num_t, tail2, GRAY1,k);
					(cudaMemcpyAsync(num_t2,tail2,sizeof(int),cudaMemcpyDeviceToHost,stream2));
					cudaMemcpyPeerAsync(g_cost2,0,g_cost3,1,no_of_nodes*sizeof(int),stream2);
					cudaStreamSynchronize(stream2);
					cudaDeviceSynchronize();
				cudaSetDevice(0);
					BFS_kernel3<<< grid, threads,0,stream1 >>>(g_q2,g_q1, d_graph_nodes,
							d_graph_edges, d_color, d_cost, num_t, tail1, GRAY1,k);

					/*err=cudaGetLastError();
					if (err != cudaSuccess){
						printf("43 kernel error= %d ",err);
						exit(-1);
					}*/
					(cudaMemcpyAsync(num_t1,tail1,sizeof(int),cudaMemcpyDeviceToHost,stream1));
					cudaMemcpyPeerAsync(g_q1+*num_t1,0,g_q3,1,(*num_t2)*sizeof(int),stream1);
					cudaStreamSynchronize(stream1);
					cudaDeviceSynchronize();
					int numt=*num_t1+*num_t2;
					cudaMemcpy(tail1,&numt,sizeof(int),cudaMemcpyHostToDevice);
					grid= dim3((int)ceil(*num_t2/(double)512), 1, 1);
					cost_adjust<<<grid,threads >>>(*num_t2,*num_t1,g_q1,g_cost2,g_cost1,g_color1,GRAY1);
					cudaDeviceSynchronize();
					//duplicate_detection<<<grid1,threads>>>(*num_t2,*num_t1,g_cost2,g_cost1,tail,d_q1,g_color1,GRAY1);
					//cudaDeviceSynchronize();

					/*err=cudaGetLastError();
					if (err != cudaSuccess){
						printf("duplicate detectin kernel error =%d ",err);
						exit(-1);
					}*/
					//pthread_barrier_wait(&bar);

				}
				else{
					grid= dim3((int)ceil(num_of_blocks/(double)2), 1, 1);
				cudaSetDevice(1);
					BFS_kernel2<<< grid, threads,0,stream2 >>>(g_q4,g_q3, d_graph_nodes2,
							d_graph_edges2, g_color2, g_cost3, num_t, tail2, GRAY1,k);
					cudaDeviceSynchronize();
					(cudaMemcpyAsync(num_t2,tail2,sizeof(int),cudaMemcpyDeviceToHost,stream2));
					cudaMemcpyPeerAsync(g_cost2,0,g_cost3,1,no_of_nodes*sizeof(int),stream2);
					cudaStreamSynchronize(stream2);
					cudaDeviceSynchronize();
				cudaSetDevice(0);
					BFS_kernel1<<< grid, threads,0,stream1 >>>(d_q2,d_q1, d_graph_nodes,
							d_graph_edges, d_color, d_cost, num_t, tail1, GRAY1,k);

					/*err=cudaGetLastError();
					if (err != cudaSuccess){
						printf(" kernel error= %d ",err);
						exit(-1);
					}*/
					printf(" num_t1=%d num_t2=%d \n",*num_t1,num_t2);
					(cudaMemcpyAsync(num_t1,tail1,sizeof(int),cudaMemcpyDeviceToHost,stream1));
					cudaMemcpyPeerAsync(g_q1+*num_t1,0,g_q3,1,(*num_t2)*sizeof(int),stream1);
					cudaStreamSynchronize(stream1);
					cudaDeviceSynchronize();
					int numt=*num_t1+*num_t2;
					printf(" num_t1=%d num_t2=%d \n",*num_t1,num_t2);
					cudaMemcpy(tail1,&numt,sizeof(int),cudaMemcpyHostToDevice);
					grid= dim3((int)ceil(*num_t2/(double)512), 1, 1);
					cost_adjust<<<grid,threads >>>(*num_t2,*num_t1,d_q1,g_cost2,g_cost1,g_color1,GRAY1);
					cudaDeviceSynchronize();
					//duplicate_detection<<<grid1,threads>>>(*num_t2,*num_t1,g_cost2,g_cost1,tail,d_q1,g_color1,GRAY1);
					//cudaDeviceSynchronize();

					 /*err=cudaGetLastError();
					if (err != cudaSuccess){
						printf("duplicate detectin kernel error=%d ",err);
						exit(-1);
					}*/

				}

			//	ENDt_TIMER("kernel2");
			}
		}

		// check if kernel execution generated any error
		//ENDt_TIMER("kernel");
		//CUT_CHECK_ERROR("Kernel execution failed");
		/*cudaError_t b= cudaGetLastError();
		if (b!= cudaSuccess)
		{ printf(" Error has occured = %d",b);
			exit(-1);}
		*/	
		k++;
	}
	while(1);
}


void onethread_bfs( struct thread_params *parameter){


	struct thread_params *parameters= (struct thread_params *)parameter;

	Node *h_graph_nodes=parameters->h_graph_nodes;
	Edge *h_graph_edges=parameters->h_graph_edges;
	int source=parameters->source;
	cudaMallocHost((void**)&num_t1,sizeof(int));	
	cudaMallocHost((void**)&num_t2,sizeof(int));	
		cudaSetDevice(0);
//		sleep(30);
		cudaStreamCreate(&stream1 ) ;

		( cudaMalloc( (void**) &d_graph_nodes, sizeof(Node)*no_of_nodes) );
		( cudaMalloc( (void**) &d_graph_edges, sizeof(Edge)*edge_list_size) );
		( cudaMalloc( (void**) &g_color1, sizeof(int)*no_of_nodes) );
		( cudaMalloc( (void**) &g_cost1, sizeof(int)*no_of_nodes));
		( cudaMalloc( (void**) &g_cost2, sizeof(int)*no_of_nodes));
		( cudaMalloc( (void**) &g_q1, sizeof(int)*no_of_nodes));
		( cudaMalloc( (void**) &g_q2, sizeof(int)*no_of_nodes));
		( cudaMalloc( (void**) &tail1, sizeof(int)));

		( cudaMemcpyAsync( d_graph_nodes, h_graph_nodes, sizeof(Node)*no_of_nodes, cudaMemcpyHostToDevice,stream1 ));
		( cudaMemcpyAsync( d_graph_edges, h_graph_edges, sizeof(Edge)*edge_list_size, cudaMemcpyHostToDevice,stream1) );
		( cudaMemcpyAsync( g_color1, color, sizeof(int)*no_of_nodes, cudaMemcpyHostToDevice,stream1 ));
		( cudaMemcpyAsync( g_cost1, h_cost, sizeof(int)*no_of_nodes, cudaMemcpyHostToDevice,stream1 ));
		
		cudaSetDevice(1);

		cudaStreamCreate(&stream2 ) ;

		( cudaMalloc( (void**) &d_graph_nodes2, sizeof(Node)*no_of_nodes) );
		( cudaMalloc( (void**) &d_graph_edges2, sizeof(Edge)*edge_list_size) );
		( cudaMalloc( (void**) &g_color2, sizeof(int)*no_of_nodes) );
		( cudaMalloc( (void**) &g_cost3, sizeof(int)*no_of_nodes));
		( cudaMalloc( (void**) &g_cost4, sizeof(int)*no_of_nodes));
		( cudaMalloc( (void**) &g_q3, sizeof(int)*no_of_nodes));
		( cudaMalloc( (void**) &g_q4, sizeof(int)*no_of_nodes));
		( cudaMalloc( (void**) &tail2, sizeof(int)));

		( cudaMemcpyAsync( d_graph_nodes2, h_graph_nodes, sizeof(Node)*no_of_nodes, cudaMemcpyHostToDevice,stream2 ));
		( cudaMemcpyAsync( d_graph_edges2, h_graph_edges, sizeof(Edge)*edge_list_size, cudaMemcpyHostToDevice,stream2) );
		( cudaMemcpyAsync( g_color2, color, sizeof(int)*no_of_nodes, cudaMemcpyHostToDevice,stream2 ));
		( cudaMemcpyAsync( g_cost3, h_cost, sizeof(int)*no_of_nodes, cudaMemcpyHostToDevice,stream2 ));
		
		cudaSetDevice(0);
		cudaStreamSynchronize(stream1);

		(cudaBindTexture(0,g_graph_node_ref,d_graph_nodes, sizeof(Node)*no_of_nodes));
		(cudaBindTexture(0,g_graph_edge_ref,d_graph_edges,sizeof(Edge)*edge_list_size));


		cudaSetDevice(1);
		cudaStreamSynchronize(stream2);
		(cudaBindTexture(0,g_graph_node_ref2,d_graph_nodes2, sizeof(Node)*no_of_nodes));
		(cudaBindTexture(0,g_graph_edge_ref2,d_graph_edges2,sizeof(Edge)*edge_list_size));


		int cur_count = 0;
                printf("Starting bfs GPU kernel\n");
                STARTt_TIMER;
                BFS_GPUone( d_graph_nodes,d_graph_edges,
                                g_color1, g_cost1, g_q1, g_q2,  tail1, source, cur_count);
                ENDt_TIMER("bfs");

		( cudaMemcpy( h_cost, g_cost1, sizeof(int)*no_of_nodes, cudaMemcpyDeviceToHost) );
		( cudaMemcpy( color, g_color1, sizeof(int)*no_of_nodes, cudaMemcpyDeviceToHost) );




		cudaSetDevice(0);
      	  ( cudaUnbindTexture(g_graph_node_ref) );
          ( cudaUnbindTexture(g_graph_edge_ref) );

                (cudaFree(d_graph_nodes));
                (cudaFree(d_graph_edges));
                (cudaFree(g_color1));
                (cudaFree(g_cost1));
                (cudaFree(g_cost2));
                (cudaFree(g_q1));
                (cudaFree(g_q2));
                (cudaFree(tail1));

                cudaStreamDestroy( stream1) ;



		cudaSetDevice(1);
	 ( cudaUnbindTexture(g_graph_node_ref2) );
         ( cudaUnbindTexture(g_graph_edge_ref2) );
                (cudaFree(d_graph_nodes2));
                (cudaFree(d_graph_edges2));
                (cudaFree(g_color2));
                (cudaFree(g_cost3));
                (cudaFree(g_cost4));
                (cudaFree(g_q3));
                (cudaFree(g_q4));
                (cudaFree(tail2));

                cudaStreamDestroy( stream2) ;
	
}
