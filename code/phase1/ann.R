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
NroSamples <- c(57,57, rep(100, 11))
for(kernelApp in c(1:7, 9:13)) {
    tempFeatures <- data.frame()
    for(gpu in c(1, 2, 3, 6, 7, 9, 10)){
        tempAppGpu <- data.frame(cbind(fread(file = paste("./datasets/",names(kernelsDict[kernelApp]), "-", gpus[gpu,'gpu_name'], ".csv", sep=""),check.names = TRUE,stringsAsFactors = FALSE), gpus[gpu,]))
        tempFeatures <- rbind(tempFeatures, tempAppGpu[sample(nrow(tempAppGpu), NroSamples[kernelApp]),])
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
        
        for(threshCorr in c(0.5, 0.75)){
            tempData <- data.frame()
            tempData <- subset(tempFeatures[which(abs(corFeatures) >= threshCorr)])
            
            if(length(tempData) > 10){
                hcFeatures <- hclust(as.dist(abs(cov(tempData, 
                                                 method = "spearman", use = "all.obs")), 
                                             upper = FALSE), method = "average")
                
                for(numberFeatures in c(10, 5)){

                    cutedTree <- cutree(hcFeatures, k = numberFeatures)
                    
                    parNameTemp <- vector()
                    
                    for(numberCluster in 1:numberFeatures){
                        Tempvariance <- sapply(tempData[cutedTree == numberCluster], var)
                        parNameTemp[numberCluster] <- names(tempData[names(Tempvariance == max(Tempvariance))])
                    }
                    
                    Data <- tempData[parNameTemp]
                    
                    Data <- cbind(Data, duration=tempDuration, gpu_id=tempGpuData$gpu_id, num_of_cores=tempGpuData$num_sp_per_sm, num_of_sm=tempGpuData$num_of_sm)
                    # Data$duration <- Data$duration/max(Data$duration)
                    
                    Data$compute_version <- NULL
                    Data$gpu_name <- NULL
                    
                    for(gpu in c(1, 2, 3,  6, 7, 9, 10)) {
                        
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
                        my.grid <- expand.grid(.decay = c(0.5, 0.1), .size = c(1, 5, 6, 7))
                        fit <- nnet(trainingDuration ~ ., data = trainingData, size = 10, maxit=5000, linout=TRUE, trace=FALSE)
                        #train(trainingDuration ~ ., data = trainingData,
                               #      method = "nnet", maxit = 1000, tuneGrid = my.grid, trace = F, linout = 1) 
                            #
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
        
    
        
        colnames(Result) <-c("Gpus", "Kernels", "Measured", "Predicted",  "Accuracy", "threshCorr", "numberFeatures")
        Result$threshCorr <- as.character(Result$threshCorr)
        Result$numberFeatures <- as.character(Result$numberFeatures)
        
        Graph <- ggplot(data=Result, aes(x=Gpus, y=Accuracy, group=Gpus, col=Gpus)) +
            geom_boxplot(size=1, outlier.size = 2.5) + 
            stat_boxplot(geom ='errorbar') +
            xlab(" ") + 
            theme_bw() +        
            ylab(expression(paste("Accuracy ",T[k]/T[m] ))) +
            facet_wrap(numberFeatures~threshCorr, scales="free", ncol = 2) 
        ggsave(paste("./images/phase1/", iML, "/",names(kernelsDict[kernelApp]), ".png",sep=""), Graph, height=10, width=20)
    }
}

