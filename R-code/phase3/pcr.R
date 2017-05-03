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

# tempFeatures <- data.frame(cbind(fread(file = paste("./datasets/AllAppGPUs.csv", sep=""),check.names = TRUE)))        
NroSamples <- c(57, 57, rep(100, 11))
for(gpu in c(1, 2, 6, 7, 9, 10)) {
    tempFeatures <- data.frame()
    
    for(kernelApp in c(1:7, 9:13)){
        tempAppGpu <- data.frame(cbind(fread(file = paste("./datasets/", 
                                                          names(kernelsDict[kernelApp]), "-", 
                                                          gpus[gpu, 'gpu_name'], ".csv", sep = ""), 
                                             check.names = TRUE, stringsAsFactors = FALSE),
                                       gpus[gpu, ]))
        tempFeatures <- rbind(tempFeatures, 
                              tempAppGpu[sample(nrow(tempAppGpu), 
                                                NroSamples[kernelApp]), ])
    }
}
tempFeatures$V1 <- NULL
tempKernel <- tempFeatures$kernel

nums <- sapply(tempFeatures, is.numeric)
tempFeatures <- tempFeatures[,nums]

tempFeatures[apply(tempFeatures, 2, is.infinite)] <- 0
tempFeatures[apply(tempFeatures, 2, is.na)] <- 0

# tempDevice <- tempFeatures
# tempFeatures$device <- NULL
tempDuration <- tempFeatures$duration

tempFeatures <- tempFeatures[,apply(tempFeatures, 2, function(v) var(v, na.rm=TRUE)!=0)]

corFeatures <- cor(getElement(tempFeatures, "duration"), tempFeatures, method = "spearman", use = "complete.obs")

Result <- data.frame()

# "lm" , "glm", "svm", "rf"
for(iML in c("pcr")){
    for(threshCorr in c(0.5, 0.75)){
        Data <- data.frame(log(subset(tempFeatures[which(abs(corFeatures) >= threshCorr)]) +  0.000000000000001))
        Data <- cbind(Data, kernel=tempKernel)
        
        if(length(Data) >= 5){
            for(numberFeatures in c(9, 5, 3, 2, 1)){
                for(kernelApp in c(1:7, 9:13)) {
                    
                    trainingData <- subset(Data, kernel !=  kernelApp)  # training data
                    testData  <- subset(Data, kernel ==  kernelApp)    # test data
                    
                    trainingDuration <- trainingData$duration
                    trainingData$duration <- NULL
                    trainingData$kernel <- NULL
                    
                    testDuration <- testData$duration
                    testData$duration <- NULL
                    testData$kernel <- NULL
                    
                    if (iML == "pcr") fit <- pcr(trainingDuration ~ ., data = trainingData, scale=TRUE, validation = "CV")
                    
                    predictions <- predict(fit, testData, comps=numberFeatures)
                    
                    predictions <- 2^predictions - 0.000000000000001
                    testDuration <- 2^testDuration - 0.000000000000001
                    accuracy <- predictions/testDuration
                    
                    tempResult <- data.frame(gpus[gpu,'gpu_name'], names(kernelsDict[kernelApp]), testDuration, predictions, accuracy, as.character(threshCorr), as.character(numberFeatures))
                    
                    Result <- rbind(Result, tempResult)
                }
            }
        }
    }
    
    colnames(Result) <-c("Gpus", "Kernels", "Measured", "Predicted",  "Accuracy", "threshCorr", "numberFeatures")
    # Result$threshCorr <- as.character(Result$threshCorr)
    
    Graph <- ggplot(data=Result, aes(x=Kernels, y=Accuracy, group=Kernels, col=Kernels)) +
        geom_boxplot(size=1, outlier.size = 2.5) + 
        stat_boxplot(geom ='errorbar') +
        xlab(" ") + 
        theme_bw() +
        theme(axis.text.x=element_blank()) +
        ylab(expression(paste("Accuracy ",T[k]/T[m] ))) +
        facet_wrap(numberFeatures~threshCorr, scales="free", ncol = 2) 
    ggsave(paste("./images/phase3/", iML, "/AllAppGPUs.png",sep=""), Graph, height=10, width=20)
}


