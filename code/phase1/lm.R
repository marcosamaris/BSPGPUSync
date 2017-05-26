library("robust")
library("robustbase")
library("MASS")
library("randomForest")
library("caret")
library("e1071")
library("ggplot2")
library("data.table")
library("ff")
library("nnet")
library("doParallel")
library("corrplot")
library("magrittr")
library("cluster")
library("dendextend")
library("Hmisc")
library("car")


dirpath <- "~/Dropbox/Doctorate/GIT/BSyncGPGPU/"
setwd(paste(dirpath, sep=""))

source("./R-code/include/common.R")
source("./R-code/include/sharedFunctions.R")

NroSamples <- c(57,57, rep(100, 11))
for(kernelApp in c(1:7)) {
    tempFeatures <- data.frame()
    for(gpu in c(1, 2, 3, 5, 6, 7, 9, 10)){
        tempAppGpu <- data.frame(cbind(fread(file = paste("./datasets/",names(kernelsDict[kernelApp]), "-", gpus[gpu,'gpu_name'], ".csv", sep=""),check.names = TRUE,stringsAsFactors = FALSE), gpus[gpu,]))
        tempFeatures <- rbind(tempFeatures, tempAppGpu[sample(nrow(tempAppGpu), NroSamples[kernelApp]),])
    }

tempGpuData <- tempFeatures[, names(tempFeatures) %in% c(names(gpus))]
tempFeatures <- tempFeatures[,!names(tempFeatures) %in% c(names(gpus))]
tempDuration <- tempFeatures$duration

nums <- sapply(tempFeatures, is.numeric)
tempFeatures <- tempFeatures[,nums]

tempFeatures$system_memory_write_transactions <- NULL
tempFeatures$system_memory_write_throughput <-  NULL
tempFeatures$system_memory_utilization <- NULL
tempFeatures$device <- NULL
tempFeatures$texture_function_unit_utilization <-  NULL
tempFeatures$l2_subp0_write_sysmem_sector_queries  <- NULL
tempFeatures$l2_subp1_write_sysmem_sector_queries <- NULL



# tempFeatures[apply(tempFeatures, 2, is.infinite)] <- 0
# tempFeatures[apply(tempFeatures, 2, is.na)] <- 0

# tempFeatures$multiprocessor_activity <- NULL
# tempFeatures$achieved_occupancy <- NULL

# tempFeatures$inst_executed	<- NULL
# tempFeatures$active_cycles	<- NULL
# tempFeatures$active_warps	<- NULL

tempFeatures <- tempFeatures[,apply(tempFeatures, 2, function(v) var(v, na.rm=TRUE)!=0)]

corFeaturesGPU <- cor(normalizeLogMax(getElement(tempFeatures, "duration")), apply(tempGpuData[,!names(tempGpuData) %in%
                    c("compute_version", "gpu_id","gpu_name","l1_cache_used")], 2, normalizeLogMax),
                    method = "spearman", use = "complete.obs")

corFeatures <- cor(normalizeLogMax(getElement(tempFeatures, "duration")), apply(tempFeatures, 2, normalizeLogMax), method = "spearman", use = "complete.obs")

# tempFeatures$duration <- NULL
corFeatures <- corFeatures[, colnames(corFeatures) != "duration"]
    
Result <- data.frame()
# "lm", "step", "glm", "svm", "rf", "ann"
for(iML in c("rf")){
for(threshCorr in c(0.5, .75)){
    tempData <- data.frame()
    tempData <- subset(tempFeatures[which(abs(corFeatures) >= threshCorr)])
    
    # varImp(tempData)
    if(length(tempData) > 10){
        hcFeatures <- hclust(as.dist(1-abs(cor(apply(tempData, 2, normalizeLogMax),
                                         method = "spearman", use = "complete.obs"))), method = "average")
        
        # roc_imp <- filterVarImp(x = tempData, y = tempDuration)
        
        for(numberFeatures in c(10, 5)){
            
            cutedTree <- cutree(hcFeatures, k = numberFeatures)
            
            parNameTemp <- vector()

            for(numberCluster in 1:numberFeatures){
                Tempvariance <-  apply(apply(tempData[cutedTree == numberCluster],2, normalizeLogMax), 2,var)
                parNameTemp[numberCluster] <- names(sort(Tempvariance)[length(Tempvariance)])            
            }
            # for(numberCluster in 1:numberFeatures){
            #     parNameTemp[numberCluster] <- names(tempData[order(roc_imp)[numberCluster]])
            # }
            
            Data <- tempData[parNameTemp]
            Data <- data.frame(
                normalizeLogMax(Data), 
                normalizeLogMax(tempGpuData[names(corFeaturesGPU[,order(abs(corFeaturesGPU),decreasing = TRUE)][1:3])]),
                duration = normalizeLogMax(tempDuration), 
                gpu_id = tempGpuData$gpu_id)
            
            for(gpu in c(1,2,3,5, 6, 7, 9,10)) {
                trainingData <- subset(Data, gpu_id !=  gpu)  # training data
                testData  <- subset(Data, gpu_id ==  gpu)   # test data
                
                trainingDuration <- trainingData$duration 
                trainingData$duration  <- NULL
                trainingData$gpu_id  <- NULL
                
                testDuration <- testData$duration
                testData$duration  <- NULL
                testData$gpu_id  <- NULL
                
                # cl <- makeCluster(2)
                # registerDoParallel(cl)
                
                if (iML == "lm") {
                    
                    # length_divisor<-6
                    # iterations <- 
                    # predictions<-foreach(m=1:iterations,.combine=cbind) %do% {
                    #     training_positions <- sample(nrow(training), size=floor((nrow(training)/length_divisor)))
                    #     train_pos<-1:nrow(training) %in% training_positions
                    #     lm_fit<-lm(y~x1+x2+x3,data=training[train_pos,])
                    #     predict(lm_fit,newdata=testing)
                    # }
                    # predictions<-rowMeans(predictions)
                    # error<-sqrt((sum((testing$y-predictions)^2))/nrow(testing))
                    
                    fit <- lm(trainingDuration ~ ., data = trainingData, method = "qr", qr=TRUE)
                    predictions <- predict(fit, testData)
                    # control <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
                    # algorithmList <- c('lm', 'glm')
                    # set.seed(5)
                    # fit <- train(trainingDuration~., data=trainingData, trControl=control, methodList=algorithmList)
                    
                    # results <- resamples(fit)
                    # summary(results)
                    # dotplot(results)
                    # 
                    # stackControl <- trainControl(method="repeatedcv", number=10, repeats=3, savePredictions=TRUE, classProbs=TRUE)
                    # set.seed(5)
                    # fit <- caretStack(models, method="lm", metric="rmse", trControl=stackControl)
                    
                    # fit_cv_model <- train(trainingData, trainingDuration, method = "lm", trControl = rctrl1)
                    # predict(fit_cv_model, testData)
                    
                    # test_reg_loo_model <- train(trainingData, trainingDuration, method = "lm", trControl = rctrl2)
                    # predictions <- predict(test_reg_loo_model, testData)
                }
                
                if (iML == "step") fit <- step(steps = 100, scale = TRUE, direction = "both", trace = FALSE,
                                             lm(trainingDuration ~ ., data = trainingData))
                
                if (iML == "glm") fit <- glm(trainingDuration ~ ., data = trainingData )
            
                if (iML == "ann") fit <- nnet(trainingDuration ~ ., data = trainingData, size = 5, maxit=5000, linout=TRUE, trace=FALSE)
                
                if (iML == "svm") fit <- svm(trainingDuration ~ ., data = trainingData, kernel="linear", scale=TRUE)
                
                if (iML == "rf") fit <- randomForest(trainingDuration ~ ., data = trainingData, mtry=5,ntree=50)
                # stopCluster(cl)
                


                # lines(predictions, col= "blue", lwd=5)
                
                # print(paste("kernel=", names(kernelsDict[kernelApp]), " GPU=", gpus[gpu,'gpu_name'],  " threshCorr=", threshCorr, " numberFeatures=", numberFeatures))
                # print(summary(fit))
                # print(coefficients(fit)) # model coefficients
                # print(confint(fit, level=0.95)) # CIs for model parameters 
                # print(fitted(fit)) # predicted values
                # print(residuals(fit)) # residuals
                # print(anova(fit)) # anova table 
                # print(vcov(fit)) # covariance matrix for model parameters 
                # print(influence(fit)) # regression diagnostics
                
                # predictions <- (2^predictions - 0.000000000000001)
                # testDuration <- (2^testDuration - 0.000000000000001)
                
                # predictions <- (2^predictions - 0.000000000000001)*max(log(tempDuration + 0.000000000000001))
                # testDuration <- (2^testDuration - 0.000000000000001)*max(log(tempDuration + 0.000000000000001))
                
                accuracy <- predictions/testDuration
                
                maxAccuracy <- max(accuracy)
                minAccuracy <- min(accuracy)
                sdAccuracy <- sd(accuracy)
                
                tempResult <- cbind(data.frame(Gpus=as.character(gpus[gpu,'gpu_name']), 
                                         Kernels=names(kernelsDict[kernelApp]), 
                                         Measured=testDuration, 
                                         Predicted=predictions, 
                                         Accuracy=accuracy, 
                                         threshCorr=threshCorr, 
                                         numberFeatures=numberFeatures))
                
                Result <- rbind(Result, tempResult)
                
            }
            # if(numberFeatures ==  10){
            #     write.csv(cbind(names(Data), c(meanAcc, rep(0,4))), file(paste("./results/phase1/", iML, "-",names(kernelsDict[kernelApp]), "-Thresh_", threshCorr, "-NoFeatures_",numberFeatures,".csv",sep="")))
            # } else{
            #     write.csv(cbind(c(names(Data), 0), meanAcc), file(paste("./results/phase1/", iML, "-",names(kernelsDict[kernelApp]), "-Thresh_", threshCorr, "-NoFeatures_",numberFeatures,".csv",sep="")))                
            # }
        }
    }
}
    Result$threshCorr <- as.character(Result$threshCorr)
    Result$numberFeatures <- as.character(Result$numberFeatures)

Graph <- ggplot(data=Result, aes(x=Gpus, y=Accuracy, group=Gpus, col=Gpus)) +
    geom_boxplot(size=1, outlier.size = 2.5) +
    stat_boxplot(geom ='errorbar') +
    xlab(" ") + 
    theme_bw() +        
    ylab(expression(paste("Accuracy ",T[k]/T[m] ))) +
    facet_wrap(numberFeatures~threshCorr, scales="free") 
ggsave(paste("./images/phase1/", iML, "-",names(kernelsDict[kernelApp]), ".png",sep=""), Graph, height=10, width=20)
write.csv(Result, file(paste("./results/phase1/", iML, "-",names(kernelsDict[kernelApp]), ".csv",sep="")))
}
}

