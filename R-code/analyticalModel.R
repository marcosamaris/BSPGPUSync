
library(ggplot2)
library(reshape2)
library(plyr)

cbbPalette <- gray(1:9/ 12)#c("red", "blue", "darkgray", "orange","black","brown", "lightblue","violet")
dirpath <- "~/Dropbox/Doctorate/GIT/BSyncGPU/"
setwd(paste(dirpath, sep=""))

gpus <- read.table("./data/deviceInfo.csv", sep=",", header=T)
NoGPU <- dim(gpus)[1]
namesTraces <- read.csv("./data/tracesNames.csv",header = T, sep = ",")


# apps <- c("matMul_gpu_uncoalesced","matMul_gpu", "matMul_gpu_sharedmem_uncoalesced", "matMul_gpu_sharedmem",
#          "matrix_sum_normal", "matrix_sum_coalesced", 
#          "dotProd", "vectorAdd",  "subSeqMax")

apps <- c("backprop") #bpnn_layerforward_CUDA
          
flopsTheoreticalPeak <- gpus['max_clock_rate']*gpus['num_of_cores']
lambda <- matrix(nrow = NoGPU, ncol = length(apps), 0, dimnames = gpus['gpu_name'])

latencySharedMemory <- 5; #Cycles per processor
latencyGlobalMemory <- latencySharedMemory* 100; #Cycles per processor

latencyL1 <- latencySharedMemory; #Cycles per processor
latencyL2 <- latencyGlobalMemory*0.5; #Cycles per processor


for (app in 1:length(apps)){
temp <- read.csv(paste("./data/", gpus[2,'gpu_name'],"/" , apps[app], "/backprop-traces.csv", sep=""), header=F,  col.names = names(namesTraces))

bpnn_layerforward_CUDA <- Data <- subset(temp, Registers.Per.Thread == 11) 

N <- seq(8192, 65536, 1024)

numberMultiplication <- 1;
numberSum <- 4;

tileWidth <- 16;
threadsPerBlock <- tileWidth*tileWidth;

gridsizes <- as.integer((N +  tileWidth -1)/tileWidth);

numberthreads <- threadsPerBlock * gridsizes;


timeComputationKernel <- ((numberMultiplication * 20 + numberSum * 10) ) * numberthreads;

L1Effect <- 0
L2Effect <- 0

CommGM <- ((numberthreads*1 - L1Effect - L2Effect)*latencyGlobalMemory + L1Effect*latencyL1 + L2Effect*latencyL2);
CommSM <- ((numberthreads*(tileWidth + tileWidth + log2(N)))*latencyGlobalMemory + L1Effect*latencyL1 + L2Effect*latencyL2);
lambda=1
timeKernel <- ( lambda*(timeComputationKernel + CommGM + CommSM)/(flopsTheoreticalPeak[2,]*10^6));

}

