NVCC= nvcc
CFLAGS= -c -arch sm_20
LDFLAGS= -lcuda -L/lib -L/usr/local/cuda/lib64

histogram :cuda_prog1.o cuda_prog2.o gold.o 
	$(NVCC) $(LDFLAGS) cuda_prog1.o cuda_prog2.o gold.o -o $@
cuda_prog1.o : 
	$(NVCC) $(CFLAGS) histogram256.cu -o $@
cuda_prog2.o : 
	$(NVCC) $(CFLAGS) main.cu -o $@
gold.o :  
	g++ -c  histogram_gold.cpp -o $@
clean:
	rm -rf *o histogram




#histogram: histogram.o
 #   nvcc -fPIC -o $@ $< -L /usr/local/cuda/lib -lcudart

#histogram.o: main.cu histogram256.cu histogram_gold.cpp histogram_common.h
 #   /usr/local/cuda/bin/nvcc --compiler-options -fno-strict-aliasing \
    -I/usr/local/cuda/include \
    -DUNIX -O2 -o $@ -c $<

#matrixMul: matrixMul.o
#    g++ -fPIC -o $@ $< -L /usr/local/cuda/lib -lcudart

# It MUST be named .cu or nvcc compiles as regular C !!! (no __global__)
#matrixMul.o: matrixMul.cu
#    /usr/local/cuda/bin/nvcc --compiler-options -fno-strict-aliasing \
    -I/usr/local/cuda/include \
    -DUNIX -O2 -o $@ -c $<
