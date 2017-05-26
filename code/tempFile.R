library(plyr)

allFeatures <- vector()
for(kernelApp in c(1:7, 9:13)) {
    for(threshCorr in c(0.5)){
        for(numberFeatures in c(5, 10)){
            allFeatures <- rbind(allFeatures, data.frame(read.csv(file = paste("./results/phase1/", iML, "-",names(kernelsDict[kernelApp]), "-Thresh_", threshCorr, "-NoFeatures_",numberFeatures,".csv",sep=""))[,2]))
        }
    }
}

freqFeatures <- count(allFeatures[allFeatures != 0])
png(filename = paste("./images/phase1/barPlotFeatures.png", sep=""), width = 1600, height = 800)
par(las=2) # make label text perpendicular to axis
par(mar=c(5,30,0,2)) # increase y-axis margin.
barplot(freqFeatures[order(freqFeatures[,2]),]$freq,names.arg = freqFeatures[order(freqFeatures[,2]),]$x, 
        horiz = TRUE, cex.names = 1,xlab = "Frequency of the Features",cex.axis = 1.5)
grid()
dev.off()



NroSamples <- c(57,57, rep(1000, 11))
for(kernelApp in c(1:7, 9:13)) {
    tempFeatures <- data.frame()
    for(gpu in c(1, 2, 3, 5, 6, 7, 9, 10)){
        tempAppGpu <- data.frame(cbind(fread(file = paste("./datasets/",names(kernelsDict[kernelApp]), "-", gpus[gpu,'gpu_name'], ".csv", sep=""),check.names = TRUE,stringsAsFactors = FALSE), gpus[gpu,]))
        tempFeatures <- rbind(tempFeatures, tempAppGpu[sample(nrow(tempAppGpu), NroSamples[kernelApp]),])
    }
tempFeatures <- tempFeatures[,!names(tempFeatures) %in% c(names(gpus))]
nums <- sapply(tempFeatures, is.numeric)
temp <- tempFeatures[, nums]

tempFeatures[apply(tempFeatures, 2, is.infinite)] <- 0
tempFeatures[apply(tempFeatures, 2, is.na)] <- 0

tempFeatures <- tempFeatures[,apply(tempFeatures, 2, function(v) var(v, na.rm=TRUE)!=0)]


library(sm)

PlotFeatures <- function(){
    cex.Size <- 2
    for(i in 2:length(tempFeatures)){
        tempFeature <- log(tempFeatures[,i] + 0.000000000000001)
        png(filename = paste("./images/phase1/features/", names(tempFeatures[i]), "-", names(kernelsDict[kernelApp]),".png", sep=""), width = 1600, height = 800)
        par(mfrow=c(1,3))
        plot(tempFeatures$device, tempFeature, xlab="GPUs", cex.lab=cex.Size, cex.axis=cex.Size, ylab=" ")
        boxplot(tempFeature~tempFeatures$device, xlab="GPUs", main=paste(names(tempFeatures[i]), sep=""), 
                cex.lab=cex.Size, cex.axis=cex.Size, cex.main=cex.Size)
    # hist(tempFeatures[,i], main = "", xlab=names(tempFeatures[i]), cex.lab=cex.Size, cex.axis=cex.Size)
    
    # names(tempFeatures[i] "FAN2"
    dens <- apply(as.data.frame(matrix(tempFeature, ncol = 8, 
                                       nrow = 100)), 2, density)
    plot(NA, xlim=range(sapply(dens, "[", "x")), ylim=range(sapply(dens, "[", "y")),
         cex.lab=cex.Size, cex.axis=cex.Size)
    mapply(lines, dens, col=1:length(dens), lwd=5)
    legend("topright", legend=c(1, 2, 3, 5, 6, 7, 9, 10), fill=1:length(dens), cex = cex.Size)
    dev.off()
    }
}
            
# fit <- svm(trainingData$Duration ~ ., data = trainingData, kernel="linear", epsilon=10)
# fit <- lm(trainingData$Duration ~ ., data = trainingData)


# mygrid <- expand.grid(.decay=c(0.5, 0.1), .size=c(4,5,6))
# fit <- train(training_duration ~., data=trainingData, method="nnet", tuneGrid=mygrid)

# fit <- nnet(training_duration ~ ., data=trainingData, size=50, weights=2)
# fit <- randomForest(training_duration ~ ., data = trainingData)




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














# ## ROBUST REGRESSION 
# tempData$issued_load.store_instructions <- NULL
# tempData$executed_control.flow_instructions<- NULL
# tempData$executed_load.store_instructions <- NULL
# tempData$issue_slots <- NULL
# tempData$issued_control.flow_instructions <- NULL
# tempData$l2_write_transactions <- NULL
# tempData$sm_cta_launched <- NULL
# tempData$gld_request <- NULL
# tempData$gst_request <- NULL
# tempData$inst_issued2 <- NULL
# tempData$inst_issued1 <- NULL
# tempData$active_cycles <- NULL
# tempData$inst_executed <- NULL
# tempData$global_store_transactions <- NULL
# tempData$l2_read_transactions <- NULL
# tempData$gst_inst_32bit <- NULL
# tempData$warps_launched <- NULL
# tempData$global_load_transactions <- NULL
# tempData$floating_point_operations.single_precision. <- NULL
# tempData$floating_point_operations.single_precision_add. <- NULL
# tempData$integer_instructions <- NULL
# tempData$misc_instructions <- NULL
# tempData$l2_throughput_.reads. <- NULL
# tempData$device_memory_read_transactions <- NULL
# tempData$global_load_throughput <- NULL
# tempData$active_warps <- NULL
# tempData$executed_ipc <- NULL
# tempData$gld_inst_32bit <- NULL
# tempData$fp_instructions.single. <- NULL
# tempData$control.flow_instructions <- NULL
# tempData$load.store_instructions <- NULL
# 

