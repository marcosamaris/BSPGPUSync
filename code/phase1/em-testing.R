library("foreach")
library("caret")
library("caretEnsemble")
library("ggplot2")


dirpath <- "~/Dropbox/Doctorate/GIT/BSyncGPGPU/"
setwd(paste(dirpath, sep=""))

source("./R-code/common.R")

for(kernelApp in c(1:2)) {
    nroSamples <- c(57,57,rep(100, 11))
    tempFeatures <- data.frame()
    
    for(gpu in c(1, 2, 3, 6, 7, 9, 10)){
        tempAppGpu <- data.frame(cbind(fread(file = paste("./datasets/",names(kernelsDict[kernelApp]), "-", gpus[gpu,'gpu_name'], ".csv", sep=""),check.names = TRUE,stringsAsFactors = FALSE), gpus[gpu,]))
        tempFeatures <- rbind(tempFeatures, tempAppGpu[sample(nrow(tempAppGpu), nroSamples[kernelApp]),])
        
    }
    # ggplot(tempFeatures, aes(duration, colour = gpu_name)) +
    #     geom_density( position = "stack")
    # ggplot(data=tempFeatures, aes(x=gpu_name, y=duration, col = gpu_name)) +
    #     geom_boxplot( )
    
    tempFeatures <- tempFeatures[names(tempFeatures) != "V1"]
    nums <- sapply(tempFeatures, is.numeric)
    tempFeatures <- tempFeatures[,nums]
    
    tempFeatures[apply(tempFeatures, 2, is.infinite)] <- 0
    tempFeatures[apply(tempFeatures, 2, is.na)] <- 0
    tempFeatures <- tempFeatures[,apply(tempFeatures, 2, function(v) var(v, na.rm=TRUE)!=0)]
    
    
    Result <- data.frame()
    
    # "lm", "glm", "svm", "rf", "ann"
    for(iML in c("lm")){
        training <- log(subset(tempFeatures, gpu_id !=  gpu) + 0.000000000000001)  # training data
        testing  <- log(subset(tempFeatures, gpu_id ==  gpu) + 0.000000000000001)   # test data
        
        trainingDuration <- training$duration
        training$duration <- NULL
        training$gpu_id <- NULL
        
        testDuration <- testing$duration
        testing$duration <- NULL
        testing$gpu_id <- NULL
        
        require(caret)
        
        fit <- train(y = trainingDuration,
                                 training,
                                 method = "lm"
        )
        coef.icept <- 
            fit$modelInfo
        
    }
    
        # for(gpu in c(1, 2, 3,  6, 7, 9, 10)) {
        #     
        #         tempData <- data.frame()
        #         tempData <- subset(tempFeatures[which(abs(corFeatures) >= threshCorr)])
        #             
        #         Data <- tempData[parNameTemp]
        #         Data <- cbind(Data, duration=tempDuration, gpu_id=tempGpuData$gpu_id, num_of_cores=tempGpuData$num_sp_per_sm, num_of_sm=tempGpuData$num_of_sm, bw=tempGpuData$bandwith)
        #         
        #         Data$compute_version <- NULL
        #         Data$gpu_name <- NULL
        #         
        #             
        # }    
}