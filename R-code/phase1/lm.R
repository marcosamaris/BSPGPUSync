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

corFeatures <- cor(getElement(tempFeatures, "duration"), tempFeatures, method = "spearman", use = "all.obs")

tempFeatures$duration <- NULL
corFeatures <- corFeatures[, colnames(corFeatures) != "duration"]

Result <- data.frame()
# "lm", "step", "glm", "svm", "rf"
for(iML in c("step")){
    
for(threshCorr in c(0.5, 0.75)){
    tempData <- data.frame()
    tempData <- subset(tempFeatures[which(abs(corFeatures) >= threshCorr)])
    
    if(length(tempData) > 10){
        hcFeatures <- hclust(as.dist(1-abs(cor(tempData, 
                                         method = "spearman", use = "all.obs")), 
                                     upper = FALSE), method = "complete")
        
        for(numberFeatures in c(10, 5)){

            cutedTree <- cutree(hcFeatures, k = numberFeatures)
            
            parNameTemp <- vector()

            for(numberCluster in 1:numberFeatures){
                Tempvariance <- sapply(tempData[cutedTree == numberCluster], var)
                parNameTemp[numberCluster] <- names(tempData[Tempvariance == max(Tempvariance)])
            }
            
            Data <- tempData[parNameTemp]
            Data <- cbind(Data, duration=tempDuration, tempGpuData)
            
            Data$compute_version <- NULL
            Data$gpu_name <- NULL
            
            for(gpu in c(1, 2, 3,  6, 7, 9, 10)) {
                
                trainingData <- log(subset(Data, gpu_id !=  gpu) + 0.000000000000001)  # training data
                testData  <- log(subset(Data, gpu_id ==  gpu) + 0.000000000000001)   # test data
                
                trainingDuration <- trainingData$duration
                trainingData$duration <- NULL
                trainingData$gpu_id <- NULL
                
                testDuration <- testData$duration
                testData$duration <- NULL
                testData$gpu_id <- NULL
                
                # cl <- makeCluster(8)
                # registerDoParallel(cl)
                
                if (iML == "lm") fit <- lm(trainingDuration ~ ., data = trainingData)
                
                if (iML == "step") fit <- step(steps = 100, scale = TRUE, direction = "both", trace = FALSE,
                                             lm(trainingDuration ~ ., data = trainingData))
                
                if (iML == "glm") fit <- glm(trainingDuration ~ ., data = trainingData )
            
                if (iML == "rlm") fit <- rlm(trainingDuration ~ ., data = trainingData)
                
                if (iML == "svm") fit <- svm(trainingDuration ~ ., data = trainingData, kernel="linear", scale=TRUE)
                
                if (iML == "rf") fit <- randomForest(trainingDuration ~ ., data = trainingData, mtry=5,ntree=50)
                # stopCluster(cl)
                
                # print(paste("kernel=", names(kernelsDict[kernelApp]), " GPU=", gpus[gpu,'gpu_name'],  " threshCorr=", threshCorr, " numberFeatures=", numberFeatures))
                # print(coefficients(fit)) # model coefficients
                # print(confint(fit, level=0.95)) # CIs for model parameters 
                # print(fitted(fit)) # predicted values
                # print(residuals(fit)) # residuals
                # print(anova(fit)) # anova table 
                # print(vcov(fit)) # covariance matrix for model parameters 
                # print(influence(fit)) # regression diagnostics
                
                
                predictions <- predict(fit, testData)
                
                predictions <- 2^predictions - 0.000000000000001
                testDuration <- 2^testDuration - 0.000000000000001
                accuracy <- predictions/testDuration
                
                maxAccuracy <- max(accuracy)
                minAccuracy <- min(accuracy)
                sdAccuracy <- sd(accuracy)
                
                
                
                tempResult <- data.frame(gpus[gpu,'gpu_name'], names(kernelsDict[kernelApp]), testDuration, predictions, accuracy, threshCorr, numberFeatures, maxAccuracy, minAccuracy, sdAccuracy)
                
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
    facet_wrap(numberFeatures~threshCorr, scales="free") 
ggsave(paste("./images/phase1/", iML, "/",names(kernelsDict[kernelApp]), ".png",sep=""), Graph, height=10, width=20)
write.csv(Result, file(paste("./results/phase1/", iML, "-",names(kernelsDict[kernelApp]), ".csv",sep="")))
}
}

