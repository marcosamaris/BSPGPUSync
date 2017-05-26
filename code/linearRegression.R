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

for(kernelApp in c(1:5, 8:13)) {
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
    
    tempFeatures[apply(tempFeatures, 2, is.infinite)] <- 0
    tempFeatures[apply(tempFeatures, 2, is.na)] <- 0
    
    tempDevice <- tempFeatures$device
    tempFeatures$device <- NULL
    tempDuration <- tempFeatures$duration
    
    tempFeatures <- tempFeatures[,apply(tempFeatures, 2, function(v) var(v, na.rm=TRUE)!=0)]
    corFeatures <- cor(getElement(tempFeatures, "duration"), tempFeatures, method = "pearson", use = "complete.obs")
    
    tempFeatures$duration <- NULL
    corFeatures <- corFeatures[, colnames(corFeatures) != "duration"]
    
    Result <- data.frame()
    
    for(threshCorr in c(0, 0.25, 0.5)){
        tempDataset <- data.frame()
        tempDataset <- tempFeatures[which(abs(corFeatures) >= threshCorr)]
        
        
        #   PCR - http://www.milanor.net/blog/performing-principal-components-regression-pcr-in-r/
        
        if(length(tempDataset) > 20){
            hcFeatures <- hclust(as.dist(1 - abs(cor(tempDataset, 
                                         method = "pearson", use = "complete.obs")), 
                                         upper = TRUE), method = "single")
            
            for(numberFeatures in c(20, 10, 5)){
                cutedTree <- cutree(hcFeatures, k=numberFeatures)
                # table(cutedTree)
                
                
                parNameTemp <- vector()
                for(numberCluster in 1:numberFeatures){
                    parNameTemp[numberCluster] <- names(tempDataset[cutedTree == numberCluster][1])
                }
                
                Data <- tempDataset[parNameTemp]
                Data <- cbind(Data, duration=tempDuration, tempGpuData)
                
                Data$compute_version <- NULL
                Data$gpu_name <- NULL
                
                
                for(gpu in c(1, 2, 3, 6, 7, 9)) {
                    
                    trainingData <- log(subset(Data, gpu_id !=  gpu) + 0.000000000000001)  # training data
                    testData  <- log(subset(Data, gpu_id ==  gpu) + 0.000000000000001)   # test data
                    
                    trainingDuration <- trainingData$duration
                    trainingData$duration <- NULL
                    trainingData$gpu_id <- NULL
                    
                    testDuration <- testData$duration
                    testData$duration <- NULL
                    testData$gpu_id <- NULL
                    
                    # Example of Stacking algorithms
                    # create submodels
                    control <- trainControl(method="repeatedcv", number=10, repeats=3)
                    algorithmList <- c('glm', 'lm', 'svmLinear')
                    set.seed(5)
                    
                    models <- caretList(x=trainingData, y=trainingDuration, trControl=control, methodList=algorithmList)
                    
                    fit <- caretStack(models, method=c("glm", 'lm', 'svmLinear'))
                    
                    # fit <- step(lm(trainingDuration ~ ., data = trainingData )
                    predictions <- predict(fit, testData, comps=3)
                
                    predictions <- 2^predictions - 0.000000000000001
                    testDuration <- 2^testDuration - 0.000000000000001
                    accuracy <- predictions/testDuration
                    accuracy
                    
                    tempResult <- data.frame(gpus[gpu,'gpu_name'], names(kernelsDict[kernelApp]), testDuration, predictions, accuracy, threshCorr, numberFeatures)
                        
                    Result <- rbind(Result, tempResult)

                }
            }
        }
    }
    colnames(Result) <-c("Gpus", "Kernels", "Measured", "Predicted",  "Accuracy", "threshCorr", "numberFeatures")
    Result$threshCorr <- as.character(Result$threshCorr)
    Result$numberFeatures <- as.character(Result$numberFeatures)
    
    Graph <- ggplot(data=Result, aes(x=Gpus, y=Accuracy, group=Gpus, col=Gpus)) +
        geom_boxplot(size=1, outlier.size = 2.5) + scale_y_continuous(limits =  c(0, 4)) +
        stat_boxplot(geom ='errorbar') +
        xlab(" ") + 
        theme_bw() +        
        ylab(expression(paste("Accuracy ",T[k]/T[m] ))) +
        facet_grid(numberFeatures~threshCorr, scales="fixed") 
    ggsave(paste("./images/accuracyGLM/",names(kernelsDict[kernelApp]), ".png",sep=""), Graph, height=10, width=20)
}

