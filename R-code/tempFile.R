temp <- read.csv(file = paste("./datasets/AllAppGPUs.csv", sep = "" )) 

# View(temp)
# temp$device_memory_utilization <- gsub("[^0-9]", "", temp$device_memory_utilization)
# temp$control.flow_function_unit_utilization <- gsub("[^0-9]", "",
#          temp$control.flow_function_unit_utilization)
# 
# temp$texture_function_unit_utilization <- gsub("[^0-9]", "",
#          temp$texture_function_unit_utilization)
# 
# temp$l1.shared_memory_utilization <- gsub("[^0-9]", "",
#          temp$l1.shared_memory_utilization)
# 
# temp$system_memory_utilization <- gsub("[^0-9]", "", 
#          temp$system_memory_utilization)


nums <- sapply(temp, is.numeric)
temp <- temp[, nums]

temp[apply(temp, 2, is.infinite)] <- 0
temp[apply(temp, 2, is.na)] <- 0

dim(temp)
tempSV <-
    temp[, apply(temp, 2, function(v)
        var(v, na.rm = TRUE) != 0)]


for(i in 1:length(tempSV)){
    png(filename = paste("./images/dataAnalysis/Features/", names(tempSV[i]), ".png", sep=""))
    par(mfrow=c(1,3))
    plot(tempSV[,i])
    boxplot(tempSV[,i], main=paste(names(tempSV[i]), sep=""))
    hist(tempSV[,i])
    dev.off()
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

