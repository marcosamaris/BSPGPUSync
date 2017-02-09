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

apps <- c("backprop", "heartwall",  "hotspot", "hotspot3D", "lavaMD", "lud", "nw") #bpnn_layerforward_CUDA

flopsTheoreticalPeak <- gpus['max_clock_rate']*gpus['num_of_cores']
lambda <- matrix(nrow = NoGPU, ncol = length(apps), 0, dimnames = gpus['gpu_name'])

latencySharedMemory <- 5; #Cycles per processor
latencyGlobalMemory <- latencySharedMemory* 100; #Cycles per processor

latencyL1 <- latencySharedMemory; #Cycles per processor
latencyL2 <- latencyGlobalMemory*0.5; #Cycles per processor

temp <- list()
for (app in 1:length(apps)){
    temp[[app]] <- read.csv(paste("./data/", gpus[2,'gpu_name'],"/" , apps[app], "/", apps[app], "-traces.csv", sep=""), header=F,  col.names = names(namesTraces))
}

temp <- read.csv(paste("./data/", gpus[2,'gpu_name'],"/" , "backprop", "/backprop-traces.csv", sep=""), header=F,  col.names = names(namesTraces))
bpnn_layerforward_CUDA <- Data <- subset(temp, Registers.Per.Thread == 11)
bpnn_adjust_weights_cuda <- Data <- subset(temp, Registers.Per.Thread == 21)


temp <- read.csv(paste("./data/", gpus[2,'gpu_name'],"/" , "gaussianS", "/gaussian-traces.csv", sep=""), header=F,  col.names = names(namesTraces))
fan1S <- Data <- subset(temp, Registers.Per.Thread == 13)
fan2S <- Data <- subset(temp, Registers.Per.Thread == 14) 

temp <- read.csv(paste("./data/", gpus[2,'gpu_name'],"/" , "gaussianC", "/gaussian-traces.csv", sep=""), header=F,  col.names = names(namesTraces))
fan1C <- Data <- subset(temp, Registers.Per.Thread == 13)
fan2C <- Data <- subset(temp, Registers.Per.Thread == 14) 

temp <- read.csv(paste("./data/", gpus[2,'gpu_name'],"/" , "heartwall", "/heartwall-traces.csv", sep=""), header=F,  col.names = names(namesTraces))
kernel <- Data <- subset(temp, Registers.Per.Thread == 31)

temp <- read.csv(paste("./data/", gpus[2,'gpu_name'],"/" , "hotspot", "/hotspot-traces.csv", sep=""), header=F,  col.names = c("Input.Size", names(namesTraces)))
calculate_temp <- Data <- subset(temp, Registers.Per.Thread == 38)

temp <- read.csv(paste("./data/", gpus[2,'gpu_name'],"/" , "hotspot3D", "/hotspot3D-traces.csv", sep=""), header=F,  col.names = c("Input.Size", names(namesTraces)))
hotspotOpt1 <- subset(temp, Registers.Per.Thread == 30)

temp <- read.csv(paste("./data/", gpus[2,'gpu_name'],"/" , "lavaMD", "/lavaMD-traces.csv", sep=""), header=F,  col.names = c(names(namesTraces)))
lud_diagonal <- subset(temp, Registers.Per.Thread == 58)

temp <- read.csv(paste("./data/", gpus[2,'gpu_name'],"/" , "lud", "/lud-traces.csv", sep=""), header=F,  col.names = c(names(namesTraces)))
lud_diagonal <- subset(temp, Registers.Per.Thread == 1024)
lud_perimeter <- subset(temp, Registers.Per.Thread == 3072)
lud_internal <- subset(temp, Registers.Per.Thread == 2048)

temp <- read.csv(paste("./data/", gpus[2,'gpu_name'],"/" , "nw", "/nw-traces.csv", sep=""), header=F,  col.names = c("Input.Size", names(namesTraces)))
needle_cuda_shared_1 <- subset(temp, Name == 26)
needle_cuda_shared_2 <-  temp[grep("_2", temp$Name),]

for (app in 1:length(apps)){
    N <- seq(8192, 65536, 1024)
    
    tileWidth <- 16;
    threadsPerBlock <- tileWidth*tileWidth;
    gridsizes <- as.integer((N +  tileWidth -1)/tileWidth);
    numberthreads <- threadsPerBlock * gridsizes;
    
    numberMultiplication <- 1;
    numberSum <- 4;
    
    L1Effect <- 0
    L2Effect <- 0
    
    timeComputationKernel <- ((numberMultiplication * 24 + numberSum * 10) ) * numberthreads;
    CommGM <- ((numberthreads*1 - L1Effect - L2Effect)*latencyGlobalMemory + L1Effect*latencyL1 + L2Effect*latencyL2);
    CommSM <- ((numberthreads*(tileWidth + tileWidth + log2(N)))*latencySharedMemory);
    lambda=.425
    timeKernel <- ( lambda*(timeComputationKernel + CommGM + CommSM)/(flopsTheoreticalPeak[2,]*10^6));
    
    Accuracy1 <- timeKernel/bpnn_layerforward_CUDA["Duration"]
    
    ###########################
    
    numberMultiplication <- 1;
    numberSum <- 4;
    
    L1Effect <- 0
    L2Effect <- 0
    
    timeComputationKernel <- ((numberMultiplication * 24 + numberSum * 10) ) * numberthreads;
    CommGM <- ((numberthreads*1 - L1Effect - L2Effect)*latencyGlobalMemory + L1Effect*latencyL1 + L2Effect*latencyL2);
    CommSM <- ((numberthreads*(tileWidth + tileWidth + log2(N)))*latencySharedMemory);
    lambda=1
    timeKernel <- ( lambda*(timeComputationKernel + CommGM + CommSM)/(flopsTheoreticalPeak[2,]*10^6));
    
    Accuracy2 <- timeKernel/bpnn_adjust_weights_cuda["Duration"]
    
    
    ###########################
    
    numberMultiplication <- 1;
    numberSum <- 4;
    
    L1Effect <- 0
    L2Effect <- 0
    
    timeComputationKernel <- ((numberMultiplication * 24 + numberSum * 10) ) * numberthreads;
    CommGM <- ((numberthreads*1 - L1Effect - L2Effect)*latencyGlobalMemory + L1Effect*latencyL1 + L2Effect*latencyL2);
    CommSM <- ((numberthreads*(tileWidth + tileWidth + log2(N)))*latencySharedMemory);
    lambda=1
    timeKernel <- ( lambda*(timeComputationKernel + CommGM + CommSM)/(flopsTheoreticalPeak[2,]*10^6));
    
    Accuracy2 <- timeKernel/fan1["Duration"]
    
    
    
    
    
}

Accuracy1
Accuracy2