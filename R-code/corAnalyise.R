library("Hmisc")
library("corrplot")

dirpath <- "~/Dropbox/Doctorate/GIT/BSyncGPGPU/"
setwd(paste(dirpath, sep=""))

gpus <- read.table("./data/deviceInfo.csv", sep=",", header=T)
NoGPU <- dim(gpus)[1]

namesMetrics35_1 <- read.csv("./data/metricsNames-3.5-1.csv",header = T, sep = ",")
namesTraces_1 <- read.csv("./data/tracesNames-1.csv",header = T, sep = ",")

namesMetrics35_2 <- read.csv("./data/metricsNames-3.5-2.csv",header = T, sep = ",")
namesTraces_2 <- read.csv("./data/tracesNames-2.csv",header = T, sep = ",")

apps <- c("backprop", "gaussian", "heartwall",  "hotspot", "hotspot3D", "lavaMD", "lud", "nw") #bpnn_layerforward_CUDA


# names(kernelsDict)
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


result <- data.frame()

for (kernel in c(1)){
    kernel=1
    tracesTemp <- data.frame(read.csv(paste("./data/", gpus[2,'gpu_name'],"/" , "traces/", kernelsDict[[kernel]],"-traces.csv", sep=""), 
                                      header=FALSE,stringsAsFactors = FALSE, strip.white = FALSE, na.strings = c("<OVERFLOW>")))
    metricsTemp <- data.frame(read.csv(paste("./data/", gpus[2,'gpu_name'],"/" , "metrics/", kernelsDict[[kernel]],"-metrics.csv", sep=""), 
                                       header=FALSE,stringsAsFactors = FALSE, strip.white = FALSE, na.strings = c("<OVERFLOW>")))
    
    names(tracesTemp) <- names(namesTraces_1)
    names(metricsTemp) <- names(namesMetrics35_1)
    
    
    if (kernel == 3 | kernel == 4){
        temp <- cbind(subset(tracesTemp[grep(names(kernelsDict[kernel]), tracesTemp$Name),], Input.Size.1 <= 4096),
                      subset(metricsTemp[grep(names(kernelsDict[kernel]), metricsTemp$Kernel),]))
    } else {
        temp <- cbind(subset(tracesTemp[grep(names(kernelsDict[kernel]), tracesTemp$Name),]),
                      subset(metricsTemp[grep(names(kernelsDict[kernel]), metricsTemp$Kernel),]))
    }
    dim(temp)
    
    nums <- sapply(temp, is.numeric)
    temp <- temp[,nums]
    temp[mapply(is.na, temp)] <- 0
    
    temp$Size <- NULL
    temp$Throughput <- NULL
    temp$elapsed_cycles_sm <- NULL
    
    # for(threshold in c(-1, .5, .75, .9, .975)){
    corTemp <- array()
    for (i in 1:length(temp)){
            if (temp[i][1,] != 0){
                corTemp[i] <- cor(temp$Duration, temp[i])
            }
    }
    
    
    
    
    
    hcTemp <- hclust(as.dist(cor(temp[which(abs(corTemp) >= .975)]),upper = TRUE),method = "complete") 
    # plot(hcTemp)
    
    dendoTemp <- as.dendrogram(hcTemp)
    # plot(dendoTemp)
    
    cutedTree <- cutree(hcTemp, k=2) 
    sample(names(cutedTree[cutedTree == 1]), 1)
    
    
    
    
    
    
    
    
    plot(clusterTemp)
    
    for(i in c(0, 0.5, 0.75, 0.9, 0.95)){
        # corMatrix <- cor(temp[which(abs(corTemp) >= .9995)], method = "pearson", use = "complete.obs")
        
        
    }
    

    col <- colorRampPalette(c("blue", "white", "green"))(20)
    png(file = paste("./images/correlation/HeatMap-",names(kernelsDict[kernel]), "-cor-", TitleFig, "-", gpus[2,'gpu_name'], ".png", sep=""),width = 8192,height = 8192)    
        heatmap(x = corTemp, col = col, symm = TRUE)
    dev.off()
    
    png(file = paste("./images/correlation/Matrix-",names(kernelsDict[kernel]), "-cor-", TitleFig, "-", gpus[2,'gpu_name'], ".png", sep=""),width = 8192,height = 8192)
    corrplot(corTemp, type = "upper", order = "hclust", 
             tl.col = "black", tl.srt = 45)
    dev.off()
    
    res2 <- rcorr(as.matrix(temp))
    flattenCorrMatrix(res2$r, res2$P)


    for(threshold in c(.5, .75, .9, .95)){
    corTemp <- array()
    
    
    
    
    }
    names(kernelsDict[kernel])
    corTemp[corTemp >= threshold]
    corTemp
png(file = paste("./images/",names(kernelsDict[kernel]), "-cor-", TitleFig, "-", gpus[2,'gpu_name'], ".png", sep=""))
    plot(corTemp[corTemp >= threshold],
         main = paste(names(kernelsDict[kernel]), " | Correlation = ", threshold, "| No. = ", length(names(temp[which(abs(corTemp) >= threshold)])),
                      " | GPU = ", gpus[2,'gpu_name'], sep=""),
         ylim=c(-1,1))
dev.off()
TitleFig <- TitleFig + 1
}


# }



# length(names(temp[which(abs(corTemp) >= 0.75)]))

