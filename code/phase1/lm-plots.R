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

source("./code/include/common.R")
source("./code/include/sharedFunctions.R")

scale_colour_manual(values=cbbPalette)
set.seed(5)

NroSamples <- c(57,57, rep(100, 11))
for(kernelApp in c(1:7, 9:13)) {
    tempFeatures <- data.frame()
    for(gpu in c(1, 2, 3, 5, 6, 7, 9, 10)){
        tempAppGpu <- data.frame(cbind(fread(file = paste("./datasets/",names(kernelsDict[kernelApp]), "-", gpus[gpu,'gpu_name'], ".csv", sep=""),check.names = TRUE,stringsAsFactors = FALSE), gpus[gpu,]))
        tempFeatures <- rbind(tempFeatures, tempAppGpu[sample(nrow(tempAppGpu), NroSamples[kernelApp]),])
    }
    
    tempGpuData <- tempFeatures[, names(tempFeatures) %in% c(names(gpus))]
    tempFeatures <- tempFeatures[,!names(tempFeatures) %in% c(names(gpus))]
    tempDuration <- tempFeatures$duration
    
    # tempFeatures$gst_request <- NULL
    # tempFeatures$gld_request <- NULL
    
    nums <- sapply(tempFeatures, is.numeric)
    tempFeatures <- tempFeatures[,nums]
    
    tempFeatures[apply(tempFeatures, 2, is.infinite)] <- 0
    tempFeatures[apply(tempFeatures, 2, is.na)] <- 0
    
    tempFeatures <- tempFeatures[,apply(tempFeatures, 2, function(v) var(v, na.rm=TRUE)!=0)]
    
    
    # 
    # if (kernelApp == 1) tempFeatures$executed_control.flow_instructions <- NULL
    # 
    if (kernelApp == 2) tempFeatures$executed_load.store_instructions <- NULL
    # 
    # if (kernelApp == 3) tempFeatures$control.flow_instructions <- NULL
    # 
    if (kernelApp == 4) tempFeatures$executed_control.flow_instructions <-  NULL
    # 
    # if (kernelApp == 5) {
    #     tempFeatures$shared_memory_load_throughput <-  NULL
    #     tempFeatures$shared_memory_store_throughput <-  NULL
    #     # tempFeatures$l2_throughput_.reads. <- NULL
    #     tempFeatures$l2_throughput_.writes. <- NULL
    # }
    
    if (kernelApp == 7) {
        tempFeatures$executed_load.store_instructions <-  NULL
        tempFeatures$control.flow_function_unit_utilization <- NULL
        tempFeatures$gst_request <- NULL
        tempFeatures$gld_request <- NULL
    }
    
    
    if (kernelApp == 9) {
        tempFeatures$device_memory_write_throughput[tempFeatures$device_memory_write_throughput == 0] <- 
        mean(tempFeatures$device_memory_write_throughput)
    
        tempFeatures$device_memory_read_throughput[tempFeatures$device_memory_read_throughput == 0] <- 
        mean(tempFeatures$device_memory_read_throughput)
    }
    
    if (kernelApp == 10) {
        # tempFeatures$executed_load.store_instructions <-  NULL
        # tempFeatures$executed_control.flow_instructions <-  NULL
        # tempFeatures$issued_control.flow_instructions <-  NULL
        # tempFeatures$sm_cta_launched <-  NULL
        # tempFeatures$shared_store <-  NULL
    }
    
    NumberGPUParameters <- 1
    # if (kernelApp == 17) {
    #     NumberGPUParameters <- 1
    # } else {
    #     NumberGPUParameters <- 2
    # }
     
    
    
    corFeaturesGPU <- abs(cor(normalizeLogMax(getElement(tempFeatures, "duration")), apply(tempGpuData[,!names(tempGpuData) %in%
                       c("compute_version", "gpu_id","gpu_name","l1_cache_used")], 2, normalizeLogMax),
                          method = "spearman", use = "complete.obs"))
    
    
    
    GPUParameters <- tempGpuData[names(corFeaturesGPU[,order(corFeaturesGPU,decreasing = TRUE)][1:NumberGPUParameters])]
    
    
    corFeatures <- cor(normalizeLogMax(getElement(tempFeatures, "duration")), 
        apply(tempFeatures, 2, normalizeLogMax), method = "spearman", use = "complete.obs")
    
    
    # tempFeatures$multiprocessor_activity<- NULL
    tempFeatures$duration <- NULL
    
    # tempFeatures[,featuresTransLog] <- apply(tempFeatures[,featuresTransLog],2, function(x){(x - mean(x))/sd(x)})
    
    corFeatures <- corFeatures[, colnames(corFeatures) != "duration"]
    
    Result <- data.frame()
    # "lm", "step", "glm", "svm", "rf", "em"
    for(iML in c("lm")){
        
        for(threshCorr in c(0.5)){
            tempData <- data.frame()
            tempData <- subset(tempFeatures[which(abs(corFeatures) >= threshCorr)])
            
            # varImp(tempData)
            
            col <- colorRampPalette(c("blue", "yellow", "red"))(20)
            png(filename = paste("./images/phase1/correlation/heatMap-", names(kernelsDict[kernelApp]), " Thresh=", threshCorr, ".png", sep=""), width = 1600, height = 800)
            heatmap(x = cor(apply(tempData, 2, normalizeLogMax),
                            method = "spearman", use = "complete.obs"),
                    col = col, symm = TRUE)
            dev.off()
            
            png(filename = paste("./images/phase1/correlation/corClustring-", names(kernelsDict[kernelApp]), " Thresh=", threshCorr, ".png", sep=""), width = 1600, height = 800)
            corrplot(cor(apply(tempData, 2, normalizeLogMax),
                         method = "spearman", use = "complete.obs"), type = "upper", order = "hclust", hclust.method="average")
            dev.off()
            
            if(length(tempData) > 10){
                hcFeatures <- hclust(as.dist(1-abs(cor(apply(tempData, 2, normalizeLogMax),
                                                       method = "spearman", use = "complete.obs"))), method = "average")
                
                # plot(hcFeatures)
                
                # roc_imp <- filterVarImp(x = tempData, y = tempDuration)
                
                for(numberFeatures in c(5, 10)){
                    
                    cutedTree <- cutree(hcFeatures, k = numberFeatures)
                    
                    png(filename = paste("./images/phase1/cluster/", names(kernelsDict[kernelApp]), 
                         " Thresh=", threshCorr, " NParam=", numberFeatures, ".png", sep=""), 
                        width = 1600, height = 800)
                    
                    dend <- as.dendrogram(hcFeatures)
                    dend %>% color_branches(k=numberFeatures) %>% plot(horiz=TRUE, 
                            main = paste( names(kernelsDict[kernelApp]), " Thresh=", 
                            threshCorr, " NParam=", numberFeatures, sep=""))
                    
                    # add horiz rect
                    dend %>% rect.dendrogram(k=numberFeatures,horiz=TRUE)
                    # add horiz (well, vertical) line:
                    abline(v = heights_per_k.dendrogram(dend)[paste(numberFeatures, sep = "")], 
                           lwd = 2, lty = 2, col = "blue")
                    # text(50, 50, table(cutedTree))
                    dev.off()
                    
                    parNameTemp <- vector()
                    
                    for(numberCluster in 1:numberFeatures){
                        Tempvariance <-  apply(apply(tempData[cutedTree == numberCluster],2, normalizeLogMax), 2,var)
                        parNameTemp[numberCluster] <- names(sort(Tempvariance)[length(Tempvariance)])
                    }
                    # for(numberCluster in 1:numberFeatures){
                    #     parNameTemp[numberCluster] <- names(tempData[order(roc_imp)[numberCluster]])
                    # }
                    
                    Data <- tempData[parNameTemp]
                    # Data[names(Data[names(Data)  %in%  featuresTransLog])] <- apply(Data[names(Data[names(Data)  %in%  featuresTransLog])], 2,
                    #                                                                 normalizeLogMax)
                    # Data[names(Data[names(Data)  %in%  featuresTransNorm])] <- apply(Data[names(Data[names(Data)  %in%  featuresTransNorm])], 2,
                    #                                                                  normalizeLogMax)
                    # Data[names(Data[names(Data)  %in%  featuresTransOther])] <- apply(Data[names(Data[names(Data)  %in%  featuresTransOther])], 2,
                    #                                                                   normalizeLogMax)
                    
                    # Data[names(Data[names(Data)  %in%  "issued_ipc"])] <- normalizeMaxSqrt(Data[names(Data[names(Data)  %in%  "issued_ipc"])])
                    if ( kernelApp == 10 & length(Data[names(Data[names(Data)  %in%  "issued_control.flow_instructions"])]) != 0){
                        Data <- apply(Data[names(Data) != "issued_control.flow_instructions"], 2, normalizeLogMax)
                        Data[names(Data[names(Data)  %in%  "issued_control.flow_instructions"])] <- normalizeLog(Data[names(Data[names(Data)  %in%  "issued_control.flow_instructions"])])    
                        
                    } else {
                        Data <- apply(Data, 2, normalizeLogMax)
                    }
                    
                    
                    # issued_control.flow_instructions
                    
                    
                    
                    # if (kernelApp == 10) {
                    #     issued_control.flow_instructions
                        
                    GPUParameters <- apply(GPUParameters, 2, normalizeLogMax)
                    
                    # median(sort(tempFeatures$device_memory_write_throughput))
                    # summary(sort(tempFeatures$device_memory_write_throughput))
                    
                    Data <- data.frame(
                                    Data, 
                                    GPUParameters,
                                    duration= normalizeLogMax(tempDuration), 
                                    gpu_id = tempGpuData$gpu_id
                                    )
                    
                    # png(filename = paste("./images/phase1/scatterPlot/", names(kernelsDict[kernelApp]), "-Thresh=", threshCorr, "-NParam=", numberFeatures, ".png", sep=""), width = 1600, height = 800)
                    # scatterplotMatrix(Data,cex.labels =  1.5)
                    # dev.off()
                    
                    meanAcc <- array()
                    png(filename = paste("./images/phase1/fitModels/", iML, "-", names(kernelsDict[kernelApp]), " Thresh=", threshCorr, " NParam=", numberFeatures, ".png", sep=""), width = 1600, height = 800)
                    par(family = "Times", mfrow=c(2,4), mai = c(1, 1, 0.5, 0.5))
                    for(gpu in c(1,2,3,5, 6,7,9,10)) {
                        trainingData <- subset(Data, gpu_id !=  gpu)  # training data
                        testData  <- subset(Data, gpu_id ==  gpu)   # test data
                        
                        trainingDuration <- trainingData$duration 
                        trainingData$duration  <- NULL
                        trainingData$gpu_id <- NULL
                        
                        testDuration <- testData$duration
                        testData$duration  <- NULL
                        testData$gpu_id <- NULL
                        
                        # cl <- makeCluster(8)
                        # registerDoParallel(cl)
                        
                        if (iML == "lm") fit <- lm(trainingDuration ~ ., data = trainingData)
                        
                        if (iML == "step") fit <- step(steps = 100, scale = TRUE, direction = "both", trace = FALSE,
                                                       lm(trainingDuration ~ ., data = trainingData))
                        
                        if (iML == "glm") fit <- glm(trainingDuration ~ ., data = trainingData )
                        
                        if (iML == "svm") fit <- svm(trainingDuration ~ ., data = trainingData, kernel="linear", scale=TRUE)
                        
                        if (iML == "rf") fit <- randomForest(trainingDuration ~ ., data = trainingData, mtry=5,ntree=50)
                        # stopCluster(cl)
                        
                        if (iML == "em") {
                            fit_lm <- lm(trainingDuration ~ ., data = trainingData)
                            fit_step <- step(steps = 100, scale = TRUE, direction = "both", trace = FALSE,
                                                       lm(trainingDuration ~ ., data = trainingData))
                            fit_glm <- glm(trainingDuration ~ ., data = trainingData )
                            fit_svm <- svm(trainingDuration ~ ., data = trainingData, kernel="linear", scale=TRUE)
                            fit_rf <- randomForest(trainingDuration ~ ., data = trainingData, mtry=5,ntree=50)
                            
                            predictions_lm <- predict(fit_lm, testData)
                            predictions_step <- predict(fit_step, testData)
                            predictions_glm <- predict(fit_glm, testData)
                            predictions_svm <- predict(fit_svm, testData)
                            predictions_rf <- predict(fit_rf, testData)
                            
                            predictions <- rowMedians(as.matrix(cbind(predictions_lm, predictions_step, predictions_glm, predictions_svm, predictions_rf))) 
                            
                        } else{
                            predictions <- predict(fit, testData)
                        }
                        
                        
                        # predictions <- predict(fit, testData)
                        
                        
                        base <- residuals(fit)
                        qqnorm(base, ylab="Studentized Residual (Fitted Model)",
                               xlab="t Quantiles",
                               main=paste(gpus[gpu,'gpu_name'], " Thresh= ", threshCorr, " NParam= ", numberFeatures, sep=""), cex.lab = 2.5, cex.main=2.5,cex=1.5,cex.axis=2)
                        qqline(base, col = 2,lwd=5)
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
                        meanAcc[gpu] <- mean(accuracy)
                    }
                    dev.off()
                    # PlotFeatures(Data,numberFeatures)
                    
                    png(filename = paste("./images/phase1/features/", names(kernelsDict[kernelApp]),"-Thresh_", threshCorr, "-NoFeatures_",numberFeatures, ".png", sep=""), width = 1200, height = 2800)
                    par(mfrow=c(numberFeatures,3))
                    cex.Size <- 2
                    for(featureSelected in 1:(numberFeatures)){
                        names(Data)
                        tempFeature <- Data[,featureSelected]
                        plot(Data$gpu_id, tempFeature, xlab="GPUs", cex.lab=cex.Size, cex.axis=cex.Size, ylab=" ")
                        boxplot(tempFeature~Data$gpu_id, xlab="GPUs", main=paste(names(Data[featureSelected]), sep=""), 
                                cex.lab=cex.Size, cex.axis=cex.Size, cex.main=cex.Size)
                        # hist(Data[,i], main = "", xlab=names(Data[i]), cex.lab=cex.Size, cex.axis=cex.Size)
                        
                        # names(tempFeatures[i] "FAN2"
                        dens <- apply(as.data.frame(matrix(tempFeature, ncol = 8, 
                                                           nrow = 100)), 2, density)
                        plot(NA, xlim=range(sapply(dens, "[", "x")), ylim=range(sapply(dens, "[", "y")),
                             cex.lab=cex.Size, cex.axis=cex.Size)
                        mapply(lines, dens, col=cbbPalette[1:length(dens)] , lwd=5)
                        legend("topright", legend=c(1, 2, 3, 5, 6, 7, 9, 10), fill=1:length(dens), cex = 1.5)
                        
                    }
                    dev.off()
                    
                    if(numberFeatures ==  10){
                        write.csv(cbind(names(Data), c(meanAcc, rep(0,4))), file(paste("./results/phase1/", iML, "-",names(kernelsDict[kernelApp]), "-Thresh_", threshCorr, "-NoFeatures_",numberFeatures,".csv",sep="")))
                    } else{
                        write.csv(cbind(c(names(Data), 0), meanAcc), file(paste("./results/phase1/", iML, "-",names(kernelsDict[kernelApp]), "-Thresh_", threshCorr, "-NoFeatures_",numberFeatures,".csv",sep="")))                
                    }
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
            scale_colour_manual(values=cbbPalette) +
            facet_wrap(numberFeatures~threshCorr, scales="free") 
        ggsave(paste("./images/phase1/", iML, "-",names(kernelsDict[kernelApp]), ".png",sep=""), Graph, height=10, width=20)
        write.csv(Result, file(paste("./results/phase1/", iML, "-",names(kernelsDict[kernelApp]), ".csv",sep="")))
    }
}

