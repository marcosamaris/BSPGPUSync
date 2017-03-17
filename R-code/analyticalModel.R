library(ggplot2)
library(reshape2)
library(plyr)

cbbPalette <- gray(1:9/ 12) #c("red", "blue", "darkgray", "orange","black","brown", "lightblue","violet")
dirpath <- "~/Dropbox/Doctorate/GIT/BSyncGPGPU/"
setwd(paste(dirpath, sep=""))

gpus <- read.table("./data/deviceInfo.csv", sep=",", header=T)
NoGPU <- dim(gpus)[1]
namesTraces <- read.csv("./data/Tesla-K40/tracesNames.csv",header = T, sep = ",")


# apps <- c("matMul_gpu_uncoalesced","matMul_gpu", "matMul_gpu_sharedmem_uncoalesced", "matMul_gpu_sharedmem",
#          "matrix_sum_normal", "matrix_sum_coalesced", 
#          "dotProd", "vectorAdd",  "subSeqMax")

apps <- c("backprop", "gaussian", "heartwall",  "hotspot", "hotspot3D", "lavaMD", "lud", "nw") #bpnn_layerforward_CUDA

flopsTheoreticalPeak <- gpus['max_clock_rate']*gpus['num_of_cores']
lambda <- matrix(nrow = NoGPU, ncol = length(apps), 0, dimnames = gpus['gpu_name'])


appList <- list()

appList$backprop <- data.frame(read.csv(paste("./data/", gpus[8,'gpu_name'],"/" , "traces", "/backprop-traces.csv", sep=""), header=F,  col.names = names(namesTraces)))

appList$gaussian <- data.frame(read.csv(paste("./data/", gpus[8,'gpu_name'],"/" , "traces", "/gaussian-traces.csv", sep=""), header=F,  col.names = names(namesTraces)))

appList$heartwall <- data.frame(read.csv(paste("./data/", gpus[8,'gpu_name'],"/" , "traces", "/heartwall-traces.csv", sep=""), header=F,  col.names = names(namesTraces)))

appList$hotspot <- data.frame(read.csv(paste("./data/", gpus[8,'gpu_name'],"/" , "traces", "/hotspot-traces.csv", sep=""), header=F,  col.names = c("Input.Size", names(namesTraces))))

appList$hotspot3D <- data.frame(read.csv(paste("./data/", gpus[8,'gpu_name'],"/" , "traces", "/hotspot3D-traces.csv", sep=""), header=F,  col.names = c("Input.Size", names(namesTraces))))

appList$lavaMD <- data.frame(read.csv(paste("./data/", gpus[8,'gpu_name'],"/" , "traces", "/lavaMD-traces.csv", sep=""), header=F,  col.names = c(names(namesTraces))))

appList$lud <- data.frame(read.csv(paste("./data/", gpus[8,'gpu_name'],"/" , "traces", "/lud-traces.csv", sep=""), header=F,  col.names = c(names(namesTraces))))

appList$nw <- data.frame(read.csv(paste("./data/", gpus[8,'gpu_name'],"/" , "traces", "/nw-traces.csv", sep=""), header=F,  col.names = c("Input.Size", names(namesTraces))))

latencySharedMemory <- 5; #Cycles per processor
latencyGlobalMemory <- latencySharedMemory* 100; #Cycles per processor

latencyL1 <- latencySharedMemory; #Cycles per processor
latencyL2 <- latencyGlobalMemory*0.5; #Cycles per processor


##### Back Propagation bpnn_layerforward_CUDA

    N <- seq(8192, 65536, 1024)
    
    tileWidth <- 16;
    threadsPerBlock <- tileWidth*tileWidth;
    blocksPerGrid <- as.integer((N +  tileWidth -1)/tileWidth);
    numberthreads <- threadsPerBlock * blocksPerGrid;
    
    numberMultiplication <- 1;
    pow2 <- 4;
    numberSum <- 4;
    ComputationKernel <- ((numberMultiplication * 36 + numberSum * 20 + pow2*36));
    
    L1Effect <- 0
    L2Effect <- 0
    
    CommGM <- ((1 - L1Effect - L2Effect)*latencyGlobalMemory + L1Effect*latencyL1 + L2Effect*latencyL2);
    CommSM <- (((tileWidth * tileWidth))*latencySharedMemory);
    lambda <- 6.6
    timeKernel <- numberthreads*(ComputationKernel + CommGM + CommSM)/((flopsTheoreticalPeak[2,]*10^6)*lambda);
    
    backpropLayer <- timeKernel/subset(appList$backprop, Registers.Per.Thread == 11)["Duration"]
    

    ##### Back Propagation bpnn_adjust_weights_cuda

    numberMultiplication <- 1;
    numberSum <- 4;
    ComputationKernel <- ((numberMultiplication * 24 + numberSum * 10));
    
    L1Effect <- 0
    L2Effect <- 0
    
    CommGM <- ((1 - L1Effect - L2Effect)*latencyGlobalMemory + L1Effect*latencyL1 + L2Effect*latencyL2);
    CommSM <- (((tileWidth + tileWidth + log2(N)))*latencySharedMemory);
    lambda=1.1
    timeKernel <- numberthreads*(ComputationKernel + CommGM + CommSM)/((flopsTheoreticalPeak[2,]*10^6)*lambda);
    
    backpropWeights <- timeKernel/subset(appList$backprop, Registers.Per.Thread == 21)["Duration"]
    
    # boxplot(backpropWeights$Duration,backpropLayer$Duration)
    ########################### Gaussian Fan1
    
    N <- c(16,32, 64, 128, seq(256, 2048, 16))
    
    numberthreads <- NULL
    for (i in 1:length(N)){
        numberthreads <- c(numberthreads, seq(N[i]-1,1,-1))
    }
    
    numberMultiplication <- 0;
    numberDivision <- 1;
    numberSum <- 0;
    ComputationKernel <- ((numberMultiplication * 24 + numberSum * 10 + numberDivision * 36) );
    
    L1Effect <- 0
    L2Effect <- 0
    
    CommGM <- ((1 - L1Effect - L2Effect)*latencyGlobalMemory + L1Effect*latencyL1 + L2Effect*latencyL2);
    CommSM <- 0
    lambda <- .0005
    timeKernel <- numberthreads*(ComputationKernel + CommGM + CommSM)/((flopsTheoreticalPeak[2,]*10^6)*lambda);
    
    fan1 <- timeKernel/subset(appList$gaussian, Registers.Per.Thread == 13)["Duration"]
    
    ########################### Gaussian Fan2
    
    # tileSize = 4
    # threadsPerBlock <- tileSize*tileSize
    # blocksPerGrid <- ceiling((N/tileSize))^2
    # #gridsizes <- ceiling((N/threadsPerBlock) + (!(N %% threadsPerBlock)));
    # 
    # numberthreads <- threadsPerBlock * blocksPerGrid
    # for (i in 4:length(N)){
    #     size <- N[i]
    #     if(size >= 128) {
    #         for(j in size:65){
    #         numberthreads <- c(numberthreads, 4096)
    #         }
    #         numberthreads <- c(numberthreads, tail(numberthreads, n=1) - 64)
    #         accumulator <- 126
    #         for(j in 63:2){
    #             numberthreads <- c(numberthreads, tail(numberthreads, n=1) - accumulator)
    #             accumulator <- accumulator - 2
    #         }
    #     }
    # }
    #     
    #     
    #     numberthreads <- NULL
        # numberthreads <- c(numberthreads, 240) 
        # accumulator <- 30
        # for(j in 14:1){
        #     numberthreads <- c(numberthreads, tail(numberthreads, n=1) - accumulator)
        #     accumulator <- accumulator - 2
        # }        
        # 
        
    # size=2048
    # for(j in size:1){
    #     numberthreads <- c(numberthreads, 4096)
    # }
    # numberthreads <- c(numberthreads, tail(numberthreads, n=1) - 64)
    # accumulator <- 126
    # for(j in 63:2){
    #     numberthreads <- c(numberthreads, tail(numberthreads, n=1) - accumulator)
    #     accumulator <- accumulator - 2
    # }
        
    
    numberMultiplication <- 1;
    numberDivision <- 0;
    numberSum <- 1;
    ComputationKernel <- ((numberMultiplication * 24 + numberSum * 10 + numberDivision * 36) );
    
    L1Effect <- 0
    L2Effect <- 0
    
    CommGM <- ((1 - L1Effect - L2Effect)*latencyGlobalMemory + L1Effect*latencyL1 + L2Effect*latencyL2);
    CommSM <- 0
    lambda = 0.4
    # for (i in 1:length(N)){
    #     timeKernel <- N[i]*(ComputationKernel + CommGM + CommSM)/((flopsTheoreticalPeak[2,]*10^6)*lambda);
    #     fan2 <- timeKernel/subset(appList$gaussian, Registers.Per.Thread == 14 & Input.Size == N[i])["Duration"]
    #     boxplot(fan2$Duration, main=N[i])
    # }
    
    
    # plot(1:2047, 
    #      subset(appList$gaussian, Registers.Per.Thread == 14 & Input.Size == 2048)["Duration"])

    timeKernel <- 4096*(ComputationKernel + CommGM + CommSM)/((flopsTheoreticalPeak[2,]*10^6)*lambda);
    fan2 <- timeKernel/subset(appList$gaussian, Registers.Per.Thread == 14 & Input.Size == N[i])["Duration"]
    
    
    TempPlot <- data.frame(row.names =  c("App", "Accuracy"))
    
    TempPlot <- rbind(TempPlot,data.frame("App"=rep("backpropLayer",length(backpropLayer)),"Accuracy"=backpropLayer$Duration))
    TempPlot <- rbind(TempPlot,data.frame("App"=rep("backpropWeights",length(backpropWeights)),"Accuracy"=backpropWeights$Duration))
    TempPlot <- rbind(TempPlot,data.frame("App"=rep("fan1",length(fan1)),"Accuracy"=fan1$Duration))
    TempPlot <- rbind(TempPlot,data.frame("App"=rep("fan2",length(fan2)),"Accuracy"=fan2$Duration))
    
    
    Graph <- ggplot(data=TempPlot, aes(x=App, y=Accuracy, group=App, col=App)) + 
        geom_boxplot( size=1) + stat_boxplot(geom ='errorbar') +
        ggtitle(paste("Accuracy of 4 kernels of Rodinia in ", gpus[8,'gpu_name'], sep="")) +
        theme(plot.title = element_text(hjust = 0.5)) +
        theme(plot.title = element_text(family = "Times", face="bold", size=30)) +
        theme(axis.title = element_text(family = "Times", face="bold", size=30)) +
        theme(axis.text  = element_text(family = "Times", face="bold", size=30, colour = "Black")) +
        theme(legend.title  = element_text(family = "Times", face="bold", size=0)) +
        theme(legend.text  = element_text(family = "Times", face="bold", size=25)) +
        theme(legend.key.size = unit(1, "cm")) +
        theme(legend.position = "none")
    
    ggsave(paste("./images/AnalyticalModel-", gpus[8,'gpu_name'], ".pdf",sep=""), Graph, device = pdf, height=21, width=29)
    write.csv(TempPlot, file = paste("./results/AnalyticalModel-", gpus[8,'gpu_name'], ".csv",sep=""))
    
        ########################### Hearthwall kernel
    
    N <- seq(20, 104)
    N <- rep(N, N)
    
    threadsPerBlock <- 256
    blocksPerGrid <- 51
    
    numberthreads <- threadsPerBlock * blocksPerGrid
    
    
    numberMultiplication <- 0;
    numberDivision <- 0;
    numberSum <- 0;
    ComputationKernel <- ((1000) );
    
    L1Effect <- 0
    L2Effect <- 0
    frame <- 656*744
    CommGM <- (frame)*latencyGlobalMemory + L1Effect*latencyL1 + L2Effect*latencyL2;
    CommSM <- (frame*N)*latencySharedMemory;
    lambda=0.21
    timeKernel <- numberthreads*(ComputationKernel + CommGM + CommSM)/((flopsTheoreticalPeak[2,]*10^6)*lambda);
    
    kernel <- timeKernel/subset(appList$heartwall, Registers.Per.Thread == 31)["Duration"]
    
    ########################### Hotspot calculate_temp
    
    N_i <- c(64, 512, 1024)
    N_j <- seq(32, 4096, 32)
    
    N = rep(N_j, N_j/2)
    N = rep(N, 3)
    
    threadsPerBlock <- 16 * 16
    blocksPerGrid <- 6 * 6
    
    numberthreads <- threadsPerBlock * blocksPerGrid
    
    
    numberMultiplication <- 6;
    numberDivision <- 0;
    numberSum <- 9;
    ComputationKernel <- (numberMultiplication * 24 + numberSum * 10 + numberDivision * 36 );
    
    L1Effect <- 0
    L2Effect <- 0

    CommGM <- (1)*latencyGlobalMemory + L1Effect*latencyL1 + L2Effect*latencyL2;
    CommSM <- (N)*latencySharedMemory;
    lambda = 1
    timeKernel <- numberthreads*(ComputationKernel + CommGM + CommSM)/((flopsTheoreticalPeak[2,]*10^6)*lambda);
    
    calculate_temp <- timeKernel/subset(appList$hotspot, Registers.Per.Thread == 38)["Duration"]
    
    
    ########################### Hotspot_3D hotspotOpt1

    N_i <- c(2, 4, 8)
    N_j <- seq(100, 1000, 100)
    
    N = rep(N_j, N_j/2)
    N = rep(N, 3)
    
    threadsPerBlock <- 64 * 4
    blocksPerGrid <- 8 * 128
    
    numberthreads <- threadsPerBlock * blocksPerGrid
    
    nnumberMultiplication <- 6;
    numberDivision <- 0;
    numberSum <- 9;
    ComputationKernel <- (numberMultiplication * 24 + numberSum * 10 + numberDivision * 36 );
    
    L1Effect <- 0
    L2Effect <- 0
    
    CommGM <- (1)*latencyGlobalMemory + L1Effect*latencyL1 + L2Effect*latencyL2;
    CommSM <- (N)*latencySharedMemory;
    lambda = 1
    timeKernel <- numberthreads*(ComputationKernel + CommGM + CommSM)/((flopsTheoreticalPeak[2,]*10^6)*lambda);
    
    hotspotOpt1 <- timeKernel/subset(appList$hotspot3D, Registers.Per.Thread == 30)["Duration"]
    
    ########################### lavaMD kernel_gpu_cuda
    
    N <- 5:100
    
    threadsPerBlock <- 64 * 4
    blocksPerGrid <- 128
    
    numberthreads <- threadsPerBlock * blocksPerGrid
    
    nnumberMultiplication <- 5;
    numberDivision <- 0;
    numberSum <- 10;
    ComputationKernel <- (numberMultiplication * 24 + numberSum * 10 + numberDivision * 36);
    
    L1Effect <- 0
    L2Effect <- 0
    
    CommGM <- (N*N)*latencyGlobalMemory + L1Effect*latencyL1 + L2Effect*latencyL2;
    CommSM <- (N*N)*latencySharedMemory;
    lambda = 0.01
    timeKernel <- numberthreads*(ComputationKernel + CommGM + CommSM)/((flopsTheoreticalPeak[2,]*10^6)*lambda);
    
    kernel_gpu_cuda <- timeKernel/subset(appList$lavaMD, Registers.Per.Thread == 58)["Duration"]
    
    ########################### LU decomposition - lud_diagonal
    N <- seq(256, 8192, 256)
    
    N_diagonal <- rep(N, N/16)
    
    threadsPerBlock <- 16
    blocksPerGrid <- 1
    
    numberthreads <- threadsPerBlock * blocksPerGrid
    
    nnumberMultiplication <- 1;
    numberDivision <- 0;
    numberSum <- 1;
    ComputationKernel <- (numberMultiplication * 24 + numberSum * 10 + numberDivision * 36);
    
    L1Effect <- 0
    L2Effect <- 0
    
    CommGM <- (N_diagonal)*latencyGlobalMemory + L1Effect*latencyL1 + L2Effect*latencyL2;
    CommSM <- (N_diagonal)*latencySharedMemory;
    lambda = 0.261
    timeKernel <- numberthreads*(ComputationKernel + CommGM + CommSM)/((flopsTheoreticalPeak[2,]*10^6)*lambda);
    
    lud_diagonal <- timeKernel/subset(appList$lud, Registers.Per.Thread == 36)["Duration"]
    
    ######################## LUD lud_perimeter
    N_perimeter <- rep(N, N/16 -1)
    
    threadsPerBlock <- 16
    blocksPerGrid <- 1
    
    numberthreads <- threadsPerBlock * blocksPerGrid
    
    nnumberMultiplication <- 1;
    numberDivision <- 0;
    numberSum <- 1;
    ComputationKernel <- (numberMultiplication * 24 + numberSum * 10 + numberDivision * 36);
    
    L1Effect <- 0
    L2Effect <- 0
    
    CommGM <- (N_diagonal)*latencyGlobalMemory + L1Effect*latencyL1 + L2Effect*latencyL2;
    CommSM <- (N_diagonal)*latencySharedMemory;
    lambda = 0.261
    timeKernel <- numberthreads*(ComputationKernel + CommGM + CommSM)/((flopsTheoreticalPeak[2,]*10^6)*lambda);
    
    lud_perimeter <- timeKernel/subset(appList$lud, Registers.Per.Thread == 46)["Duration"]
    
    ############################ LUD lud_internal
    N_perimeter <- rep(N, N/16 -1)
    
    threadsPerBlock <- 16
    blocksPerGrid <- 1
    
    numberthreads <- threadsPerBlock * blocksPerGrid
    
    nnumberMultiplication <- 1;
    numberDivision <- 0;
    numberSum <- 1;
    ComputationKernel <- (numberMultiplication * 24 + numberSum * 10 + numberDivision * 36);
    
    L1Effect <- 0
    L2Effect <- 0
    
    CommGM <- (N_diagonal)*latencyGlobalMemory + L1Effect*latencyL1 + L2Effect*latencyL2;
    CommSM <- (N_diagonal)*latencySharedMemory;
    lambda = 0.261
    timeKernel <- numberthreads*(ComputationKernel + CommGM + CommSM)/((flopsTheoreticalPeak[2,]*10^6)*lambda);
    
    lud_internal <- timeKernel/subset(appList$lud, Registers.Per.Thread == 21)["Duration"]
    
    lud_diagonal <- subset(temp, Registers.Per.Thread == 36)
    lud_perimeter <- subset(temp, Registers.Per.Thread == 46)
    lud_internal <- subset(temp, Registers.Per.Thread == 21)
    
    
    ######################### Needleman-Wunsch needle_cuda_shared_1

    N_i <- seq(256, 4096, 256)
    N_j <- 1:10
    
    threadsPerBlock <- 16
    blocksPerGrid <- NULL
    for (i in 1:length(N_i)){
        blocksPerGrid <- c(blocksPerGrid, seq(1,N_i[i]/16))
    }
    
    numberthreads <- threadsPerBlock * blocksPerGrid
    
    nnumberMultiplication <- 6;
    numberDivision <- 0;
    numberSum <- 9;
    ComputationKernel <- (numberMultiplication * 24 + numberSum * 10 + numberDivision * 36 );
    
    L1Effect <- 0
    L2Effect <- 0
    
    CommGM <- (1)*latencyGlobalMemory + L1Effect*latencyL1 + L2Effect*latencyL2;
    CommSM <- (N)*latencySharedMemory;
    lambda = 0.01
    timeKernel <- numberthreads*(ComputationKernel + CommGM + CommSM)/((flopsTheoreticalPeak[2,]*10^6)*lambda);
    
    needle_cuda_shared_1 <- timeKernel/subset( appList$nw[grep("_1", appList$nw$Name),])["Duration"]
    
    ######################### Needleman-Wunsch needle_cuda_shared_2
    
    N_i <- seq(256, 4096, 256)
    N_j <- 1:10
    
    threadsPerBlock <- 16
    blocksPerGrid <- NULL
    for (i in 1:length(N_i)){
        blocksPerGrid <- c(blocksPerGrid, seq(N_i[i]/16-1 , 1))
    }
    
    numberthreads <- threadsPerBlock * blocksPerGrid
    
    nnumberMultiplication <- 6;
    numberDivision <- 0;
    numberSum <- 9;
    ComputationKernel <- (numberMultiplication * 24 + numberSum * 10 + numberDivision * 36 );
    
    L1Effect <- 0
    L2Effect <- 0
    
    CommGM <- (1)*latencyGlobalMemory + L1Effect*latencyL1 + L2Effect*latencyL2;
    CommSM <- (N)*latencySharedMemory;
    lambda = 0.01
    timeKernel <- numberthreads*(ComputationKernel + CommGM + CommSM)/((flopsTheoreticalPeak[2,]*10^6)*lambda);
    
    needle_cuda_shared_2 <- timeKernel/subset( appList$nw[grep("_2", appList$nw$Name),])["Duration"]
    
    appAllKernel <- data.frame()
    appAllKernel <- rbind(appAllKernel, cbind("backpropLayer", backpropLayer[,]))
    appAllKernel <- rbind(appAllKernel, cbind("backpropWeights", backpropWeights[,]))
    appAllKernel <- rbind(appAllKernel, cbind("fan1", fan1[,]))
    appAllKernel <- rbind(appAllKernel, cbind("fan2", fan2[,]))
    appAllKernel <- rbind(appAllKernel, cbind("kernel", kernel[,]))
    appAllKernel <- rbind(appAllKernel, cbind("calculate_temp", calculate_temp[,]))
    appAllKernel <- rbind(appAllKernel, cbind("hotspotOpt1", hotspotOpt1[,]))
    appAllKernel <- rbind(appAllKernel, cbind("kernel_gpu_cuda", kernel_gpu_cuda[,]))
    appAllKernel <- rbind(appAllKernel, cbind("lud_diagonal", lud_diagonal[,]))
    appAllKernel <- rbind(appAllKernel, cbind("lud_perimeter", lud_perimeter[,]))
    appAllKernel <- rbind(appAllKernel, cbind("lud_internal", lud_internal[,]))
    appAllKernel <- rbind(appAllKernel, cbind("needle_cuda_shared_1", needle_cuda_shared_1[,]))
    appAllKernel <- rbind(appAllKernel, cbind("needle_cuda_shared_2", needle_cuda_shared_2[,]))
    
    names(appAllKernel) <- c("Kernel", "Accuracy")
    
    appAllKernel$Accuracy <- as.numeric(levels(appAllKernel$Accuracy))[appAllKernel$Accuracy]
    
    Graph <- ggplot(data=appAllKernel, aes(x=Kernel, y=Accuracy, group=Kernel, color=Kernel)) + 
        ggtitle("Accuracy of the BSP-based model on Rodinia ") +
        geom_boxplot() + scale_y_log10(breaks = scales::trans_breaks("log10", function(x) 10^x),
                                       labels = scales::trans_format("log10", scales::math_format(10^.x))) + 
        annotation_logticks(sides = "l") + 
        stat_boxplot(geom ='errorbar') +
        theme(legend.position ="none") +
        theme(plot.title = element_text(family = "Times", face="bold", size=50)) +
        theme(axis.title = element_text(family = "Times", face="bold", size=50)) +
        theme(axis.text  = element_text(family = "Times", face="bold", size=50, colour = "Black")) +
        theme(axis.text.x=element_blank()) 
        
        
    ggsave(paste("./images/Rodinia-BSP-Modeling.pdf",sep=""), Graph, device = pdf, height=21, width=29)
    write.csv(appAllKernel, file = "./results/Rodinia-BSP-Modeling.csv")
    