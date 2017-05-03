library("nnet")
library("robust")
library("robustbase")
library("MASS")
library("randomForest")
library("caret")
library("e1071")
library("ggplot2")
library("data.table")
library("ff")
library("doParallel")

dirpath <- "~/Dropbox/Doctorate/GIT/BSyncGPGPU/"
setwd(paste(dirpath, sep=""))

source("./R-code/common.R")
set.seed(5)
for(kernelApp in c(1:7, 9:13)) {
    tempFeatures <- data.frame()
    for(gpu in c(1, 2, 3, 6, 7, 9)){
        tempAppGpu <- data.frame(cbind(fread(file = paste("./datasets/",names(kernelsDict[kernelApp]), "-", gpus[gpu,'gpu_name'], ".csv", sep=""),check.names = TRUE,stringsAsFactors = FALSE), gpus[gpu,]))
        tempFeatures <- rbind(tempFeatures, tempAppGpu[sample(nrow(tempAppGpu), 46),])
        
    }
    # ggplot(tempFeatures, aes(duration, colour = gpu_name)) +
    #     geom_density( position = "stack")
    # ggplot(data=tempFeatures, aes(x=gpu_name, y=duration, col = gpu_name)) +
    #     geom_boxplot( )
    
    tempFeatures <- tempFeatures[names(tempFeatures) != "V1"]
    
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
    
    
    # corFeatures <- apply()
    corFeatures <- cor(getElement(tempFeatures, "duration"), tempFeatures, method = "spearman", use = "pairwise.complete.obs")
    
    tempFeatures$duration <- NULL
    corFeatures <- corFeatures[, colnames(corFeatures) != "duration"]
    
    Result <- data.frame()
    # , "glm", "svm", "rf"
    for(iML in c("ann")){
        
        for(threshCorr in c(0, 0.25, 0.5)){
            tempData <- data.frame()
            tempData <- subset(tempFeatures[which(abs(corFeatures) >= threshCorr)])
            
            if(length(tempData) > 20){
                hcFeatures <- hclust(as.dist(cor(tempData, 
                                                 method = "spearman", use = "all.obs"), 
                                             upper = FALSE), method = "average")
                
                for(numberFeatures in c(20, 10, 5)){

                    cutedTree <- cutree(hcFeatures, k = numberFeatures)
                    
                    parNameTemp <- vector()
                    
                    for(numberCluster in 1:numberFeatures){
                        parNameTemp[numberCluster] <- names(tempData[cutedTree == numberCluster][1])
                    }
                    
                    Data <- tempData[parNameTemp]
                    
                    Data <- cbind(Data, duration=tempDuration, gpu_id=tempGpuData$gpu_id, num_of_cores=tempGpuData$num_sp_per_sm, num_of_sm=tempGpuData$num_of_sm)
                    
                    # Data$duration <- Data$duration/max(Data$duration)
                    
                    Data$compute_version <- NULL
                    Data$gpu_name <- NULL
                    
                     
                    
                    for(gpu in c(1, 2, 3,  6, 7, 9)) {
                        
                        trainingData <- log(subset(Data, gpu_id !=  gpu)  + 0.000000000000001) # training data
                        testData  <- log(subset(Data, gpu_id ==  gpu) + 0.000000000000001)   # test data
                        
                        trainingDuration <- trainingData$duration
                        trainingData$duration <- NULL
                        trainingData$gpu_id <- NULL
                        
                        testDuration <- testData$duration
                        testData$duration <- NULL
                        testData$gpu_id <- NULL
                        
                        # cl <- makeCluster(detectCores())
                        # registerDoParallel(cl)
                        my.grid <- expand.grid(.decay = c(0.5, 0.1), .size = c(5, 6, 7))
                        fit <- nnet(trainingDuration ~ ., data = trainingData,size = 35, rang = 0.001, decay = 5e-2, maxit = 20, linout=TRUE, trace=FALSE)
                        # stopCluster(cl)
                        
                        predictions <- predict(fit, testData)
                        predictions <- 2^predictions - 0.000000000000001
                        testDuration <- 2^testDuration - 0.000000000000001
                        

                        accuracy <- predictions/testDuration
                        
                        
                        tempResult <- data.frame(gpus[gpu,'gpu_name'], names(kernelsDict[kernelApp]), testDuration, predictions, accuracy, threshCorr, numberFeatures)
                        
                        Result <- rbind(Result, tempResult)
                        
                    }
                }
            }
        }
        
        if (iML == "lm") ML= "LM"
        
        if (iML == "glm") ML= "GLM"
        
        if (iML == "ann") ML= "ANN"
        
        if (iML == "svm") ML= "SVM"
        
        if (iML == "rf") ML= "RF"
        
        colnames(Result) <-c("Gpus", "Kernels", "Measured", "Predicted",  "Accuracy", "threshCorr", "numberFeatures")
        Result$threshCorr <- as.character(Result$threshCorr)
        Result$numberFeatures <- as.character(Result$numberFeatures)
        
        Graph <- ggplot(data=Result, aes(x=Gpus, y=Accuracy, group=Gpus, col=Gpus)) +
            geom_boxplot(size=1, outlier.size = 2.5) + 
            stat_boxplot(geom ='errorbar') +
            xlab(" ") + 
            theme_bw() +        
            ylab(expression(paste("Accuracy ",T[k]/T[m] ))) +
            facet_grid(numberFeatures~threshCorr, scales="fixed") 
        ggsave(paste("./images/phase1/", ML, "/",names(kernelsDict[kernelApp]), ".png",sep=""), Graph, height=10, width=20)
    }
}

