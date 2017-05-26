library("foreach")
# library("e1071")
# require("nnet")
# library("randomForest")
library("caret")
library("caretEnsemble")
# library("ggplot2")
dirpath <- "~/Dropbox/Doctorate/GIT/BSyncGPGPU/"
setwd(paste(dirpath, sep=""))

gpus <- read.table("./data/deviceInfo.csv", sep=",", header=T)
NoGPU <- dim(gpus)[1]

apps <- c("backprop", "gaussian", "heartwall",  "hotspot", "hotspot3D", "lavaMD", "lud", "nw") #bpnn_layerforward_CUDA

kernelsDict <- vector(mode="list", length=13)
names(kernelsDict) <- c("bpnn_layerforward_CUDA",
                        "bpnn_adjust_weights_cuda",
                        "Fan1",
                        "Fan2",
                        "kernel",
                        "calculate_temp",
                        "hotspotOpt1",
                        "kernel_gpu_cuda",
                        "lud_diagonal",
                        "lud_perimeter",
                        "lud_internal",
                        "needle_cuda_shared_1",
                        "needle_cuda_shared_2"
)

kernelsDict[[1]] <- apps[1]
kernelsDict[[2]] <- apps[1]
kernelsDict[[3]] <- apps[2]
kernelsDict[[4]] <- apps[2]
kernelsDict[[5]] <- apps[3]
kernelsDict[[6]] <- apps[4]
kernelsDict[[7]] <- apps[5]
kernelsDict[[8]] <- apps[6]
kernelsDict[[9]] <- apps[7]
kernelsDict[[10]] <- apps[7]
kernelsDict[[11]] <- apps[7]
kernelsDict[[12]] <- apps[8]
kernelsDict[[13]] <- apps[8]

kernel_1_parameter <- c(1,2,3,4,5,8,9,10,11)
kernel_2_parameter <- c(6,7,12,13)

for(kernelApp in c(1)) {
    tempFeatures <- data.frame()
    for(gpu in c(1, 2, 3, 6, 7, 9)){
        tempAppGpu <- cbind(read.csv(file = paste("./datasets/",names(kernelsDict[kernelApp]), "-", gpus[gpu,'gpu_name'], ".csv", sep=""),stringsAsFactors = FALSE ), gpus[gpu,])        
        tempFeatures <- rbind(tempFeatures, data.frame(tempAppGpu))   
    }
    
    tempFeatures <-tempFeatures[names(tempFeatures) != "X"]
    
    tempGpuData <- tempFeatures[, names(tempFeatures) %in% c(names(gpus))]
    tempFeatures <- tempFeatures[,!names(tempFeatures) %in% c(names(gpus))]
    
    nums <- sapply(tempFeatures, is.numeric)
    tempFeatures <- tempFeatures[,nums]
    
    nums <- sapply(tempGpuData, is.numeric)
    tempGpuData <- tempGpuData[,nums]
    
    tempFeatures[apply(tempFeatures, 2, is.infinite)] <- 0
    tempFeatures[apply(tempFeatures, 2, is.na)] <- 0
    
    tempDevice <- tempFeatures$device
    tempFeatures$device <- NULL
    tempDuration <- tempFeatures$duration
    
    tempFeatures <- tempFeatures[,apply(tempFeatures, 2, function(v) var(v, na.rm=TRUE)!=0)]
    
    corFeatures <- cor(getElement(tempFeatures, "duration"), tempGpuData, method = "pearson", use = "complete.obs")
    
    tempFeatures$duration <- NULL
    corFeatures <- corFeatures[, colnames(corFeatures) != "duration"]
    
    Result <- data.frame()
    
}