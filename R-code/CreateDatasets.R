library(zoo)
library(e1071)

dirpath <- "~/Dropbox/Doctorate/GIT/BSyncGPGPU/"
setwd(paste(dirpath, sep=""))

gpus <- read.table("./data/deviceInfo.csv", sep=",", header=T)
NoGPU <- dim(gpus)[1]

apps <- c()

namesMetrics35_1 <- read.csv("./data/metricsNames-3.5-1.csv",header = T, sep = ",")
namesTraces_1 <- read.csv("./data/tracesNames-1.csv",header = T, sep = ",")

namesMetrics35_2 <- read.csv("./data/metricsNames-3.5-2.csv",header = T, sep = ",")
namesTraces_2 <- read.csv("./data/tracesNames-2.csv",header = T, sep = ",")

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

# names(kernelsDict)

metricsCorrelation <- matrix(c(names(namesTraces_1), names(namesMetrics35_1)))

result <- data.frame()
for (kernel in c(1,2,3,4,5,8,9,10,11)){

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
    # dim(temp)
    
    nums <- sapply(temp, is.numeric)
    temp <- temp[,nums]
    temp <- log(temp)
    temp[mapply(is.infinite, temp)] <- 0
    
    temp$Size <- NULL
    temp$Throughput <- NULL
    temp$elapsed_cycles_sm <- NULL
    
    # Create Training and Test data -
    set.seed(length(temp))  # setting seed to reproduce results of random sampling
    trainingRowIndex <- sample(1:nrow(temp), 0.8*nrow(temp))  # row indices for training data
    trainingData <- temp[trainingRowIndex, ]  # model training data
    # DurationTemp <- trainingData$Duration
    # trainingData$Duration <- NULL
    testData  <- temp[-trainingRowIndex, ]   # test data
    dim(trainingData)
    dim(testData)
    
    TestDuration <- 2^testData$Duration
    testData$Duration <- NULL
    
    fit <- svm(trainingData$Duration ~ ., data = trainingData, kernel="linear", scale=FALSE) 
    # fit <- glm(trainingData$Duration ~ ., data = trainingData)
    # summary(base)
    
    predictions <- predict(fit, testData)
    predictions <- 2^predictions
    mse <- mean((predictions/TestDuration - 1)^2)
    mae <- mean(abs(as.matrix(TestDuration)  - predictions))
    mape <- mean(abs(as.matrix(TestDuration)  - predictions/predictions))
    
    Acc <- predictions/TestDuration
    AccMin <- min(Acc)
    AccMean <- mean(as.matrix(Acc))
    AccMedian <- median(as.matrix(Acc))
    AccMax <- max(Acc)
    AccSD <- sd(as.matrix(Acc))
    
    tempResult <- data.frame(kernel, TestDuration, predictions, Acc, AccMin, AccMax, AccMean, AccMedian, AccSD,mse, mae,mape)
    
    result <- rbind(result, tempResult)
    
    
    png(file = paste("./images/accuracySVM/","Boxplot-Accurcy-", names(kernelsDict[kernel]), "-", gpus[2,'gpu_name'], ".png", sep=""))
            boxplot(Acc,
                 main = paste("Accuracy of ", names(kernelsDict[kernel]), " over ", gpus[2,'gpu_name'], sep=""))
        dev.off()
        
}
colnames(result) <-c("Kernel", "Measured", "Predicted",  "accuracy", "Min", "max", "Mean", "Median", "SD", "mse", "mae", "mape")
write.csv(result, file = "./results/SVM.csv")    
    # temp$L1.Shared.Memory.Utilization
    # temp$L2.Cache.Utilization
    # temp$Texture.Cache.Utilization
    # temp$Device.Memory.Utilization
    # temp$System.Memory.Utilization
    # temp$Load.Store.Function.Unit.Utilization
    # temp$Arithmetic.Function.Unit.Utilization
    # temp$Control.Flow.Function.Unit.Utilization
    # temp$Texture.Function.Unit.Utilization
    
    # na.locf(DataAppGPU30$Global.Load.Transactions.Per.Request)

    
    
    
    # for(threshold in c(-1, .5, .75, .9, .975)){
    #     corTemp <- array()
    #     for (i in 1:length(temp)){
    #         if(is.numeric(temp[[i]])){
    #             corTemp[i] <- cor(temp$Duration, temp[[i]])
    #         }
    #     }
    #     # names(kernelsDict[kernel])
    #     # corTemp[corTemp >= threshold]
    #     # corTemp
    #     
    #     # png(file = paste("./images/",names(kernelsDict[kernel]), "-cor-", TitleFig, "-", gpus[2,'gpu_name'], ".png", sep=""))
    #     #     plot(corTemp[corTemp >= threshold], 
    #     #          main = paste(names(kernelsDict[kernel]), " | Correlation = ", threshold, "| No. = ", length(names(temp[which(abs(corTemp) >= threshold)])),
    #     #                       " | GPU = ", gpus[2,'gpu_name'], sep=""),
    #     #          ylim=c(-1,1))
    #     # dev.off()
    #     # TitleFig <- TitleFig + 1
    # }
    
    
# }

    
    
# length(names(temp[which(abs(corTemp) >= 0.75)]))

    