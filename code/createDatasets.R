dirpath <- "~/Dropbox/Doctorate/GIT/BSyncGPGPU/"
setwd(paste(dirpath, sep = ""))

source("./code/include/configCreatedatasets.R")

NroSamples <- c(57, 57, rep(1000, 11))

kernelsGPU <- data.frame()
for (gpu in c(1,2,3,5, 6,7,9,10)) {
    tempComm <- data.frame()
    kernels <- data.frame()
    for (kernelApp in c(10)) {
        tracesTemp <- data.frame(read.csv( paste( "./data/", gpus[gpu, 'gpu_name'], "/" , "traces/", 
                                                kernelsDict[[kernelApp]], "-traces.csv", sep = "" ), header = FALSE ))
        metricsTemp <- data.frame(read.csv(paste("./data/", gpus[gpu, 'gpu_name'], "/" , "metrics/", 
                                                kernelsDict[[kernelApp]], "-metrics.csv", sep = ""), header = FALSE ))
        
        if (kernelApp %in% kernel_1_parameter) {
            if (gpus[gpu, 'compute_version'] == 3.0) {
                names(metricsTemp) <- c("input.size.1", names(metricNames30))
            }
            if (gpus[gpu, 'compute_version'] == 3.5) {
                names(metricsTemp) <- c("input.size.1", names(metricNames35))
            }
            if (gpus[gpu, 'compute_version'] == 5.0) {
                names(metricsTemp) <- c("input.size.1", names(metricNames50))
            }
            if (gpus[gpu, 'compute_version'] == 5.2) {
                if(gpus[gpu, 'gpu_name']  == "GTX-970"){
                    names(metricsTemp) <- c("input.size.1", names(metricNames52.970))
                } else{
                    names(metricsTemp) <- c("input.size.1", names(metricNames52))
                }
            }
            names(tracesTemp) <- c("input.size.1", names(traceNames))
            
            tracesTemp <-
                as.data.frame(append(tracesTemp, list(input.size.2 = 0), after = 1))
            metricsTemp <-
                as.data.frame(append(metricsTemp, list(input.size.2 = 0), after = 1))
            
            if (kernelApp == 3 | kernelApp == 4) {
                tempFeatures <-
                    cbind(subset(tracesTemp[grep(names(kernelsDict[kernelApp]), tracesTemp$name), ], input.size.1 <= 4096),
                          subset(metricsTemp[grep(names(kernelsDict[kernelApp]), metricsTemp$kernel), ]))
            } else {
                tempFeatures <-
                    cbind(subset(tracesTemp[grep(names(kernelsDict[kernelApp]), tracesTemp$name), ]),
                          subset(metricsTemp[grep(names(kernelsDict[kernelApp]), metricsTemp$kernel), ]))
            }
        } else if (kernelApp %in% kernel_2_parameter) {
            if (gpus[gpu, 'compute_version'] == 3.0) {
                names(metricsTemp) <-
                    c("input.size.1",
                      "input.size.2",
                      names(metricNames30))
            }
            if (gpus[gpu, 'compute_version'] == 3.5) {
                names(metricsTemp) <-
                    c("input.size.1",
                      "input.size.2",
                      names(metricNames35))
            }
            if (gpus[gpu, 'compute_version'] == 5.0) {
                names(metricsTemp) <-
                    c("input.size.1",
                      "input.size.2",
                      names(metricNames50))
            }
            if (gpus[gpu, 'compute_version'] == 5.2) {
                if(gpus[gpu, 'gpu_name']  == "GTX-970"){
                    names(metricsTemp) <- c("input.size.1", "input.size.2", names(metricNames52.970))
                } else{
                    names(metricsTemp) <- c("input.size.1", "input.size.2", names(metricNames52))
                }
            }
            names(tracesTemp) <- c("input.size.1", "input.size.2", names(traceNames))
            
            tempFeatures <-
                cbind(subset(tracesTemp[grep(names(kernelsDict[kernelApp]), tracesTemp$name), ]),
                      subset(metricsTemp[grep(names(kernelsDict[kernelApp]), metricsTemp$kernel), ]))
        }
        tempComm <- rbind(tempComm, subset(tracesTemp[grep("memcpy", tracesTemp$name), ]))
        
        tempFeatures <- tempFeatures[, c("input.size.1", "input.size.2", names(traceNames), selectedFeatures)]
        
        tempFeatures$device <- gpus[gpu, "gpu_id"]
        
        tempFeatures$device.1 <- NULL
        tempFeatures$start <- NULL
        tempFeatures <- tempFeatures[, unique(colnames(tempFeatures))]
        
        tempFeatures$size <- NULL 
        tempFeatures$throughput <- NULL
        tempFeatures$name <- NULL
        tempFeatures$kernel <- kernelApp
        
        
        tempFeatures$device_memory_utilization <-
            as.numeric(gsub("[^0-9]", "", tempFeatures$device_memory_utilization))
        tempFeatures$control.flow_function_unit_utilization <-
            as.numeric(gsub("[^0-9]", "", tempFeatures$control.flow_function_unit_utilization))
        tempFeatures$texture_function_unit_utilization <-
            as.numeric(gsub("[^0-9]",
                 "",
                 tempFeatures$texture_function_unit_utilization))
        tempFeatures$l1.shared_memory_utilization <-
            as.numeric(gsub("[^0-9]",
                 "",
                 tempFeatures$l1.shared_memory_utilization))
        tempFeatures$system_memory_utilization <-
            as.numeric(gsub("[^0-9]", "", tempFeatures$system_memory_utilization))
        
        
        
        dim(tempFeatures)
        # elapsed_cycles_sm
        # multiprocessor_activity
        # flop_efficiency.peak_single.
        # flop_efficiency.peak_double.
        #
        #
        # active_warps
        # achieved_occupancy
        #
        # active_cycles
        #
        # executed_ipc
        # flop_efficiency.peak_single.
        # flop_efficiency.peak_double.
        
        tempFeatures <- tempFeatures[, -which(names(tempFeatures) %in% c("grid.z", "block.z", "dynamic.smem", "context",  "stream",  "context.1",
                                               "stream.1", "l2_subp0_read_sysmem_sector_queries", "l2_subp1_read_sysmem_sector_queries", "gld_inst_8bit",
                                               "gld_inst_16bit", "gld_inst_64bit", "gld_inst_128bit", "gst_inst_8bit", "gst_inst_16bit", "gst_inst_64bit", 
                                               "gst_inst_128bit", "prof_trigger_00", "prof_trigger_01", "prof_trigger_02", "prof_trigger_03", "prof_trigger_04",
                                               "prof_trigger_05", "prof_trigger_06", "prof_trigger_07", "local_load", "local_store", "local_load_transactions", 
                                               "local_store_transactions",  "atom_count", "gred_count", "local_memory_load_transactions_per_request",
                                               "local_memory_store_transactions_per_request", "system_memory_read_transactions", "local_memory_load_throughput", 
                                               "local_memory_store_throughput", "system_memory_read_throughput", "l2_throughput_.atomic_requests.", "atomic_transactions", "atomic_transactions_per_request",
                                               "l2_transactions_.atomic_requests."))]
        
        tempFeatures$local_memory_overhead <- NULL
        tempFeatures$instruction_replay_overhead <- NULL
        tempFeatures$local_memory_load_throughput <- NULL
        tempFeatures$local_memory_store_throughput <- NULL
        
        tempFeatures$tex0_cache_sector_queries <- NULL
        tempFeatures$tex1_cache_sector_queries <- NULL
        tempFeatures$tex0_cache_sector_misses <- NULL
        tempFeatures$tex1_cache_sector_misses <- NULL
        tempFeatures$l2_subp0_read_tex_sector_queries <- NULL
        tempFeatures$l2_subp1_read_tex_sector_queries <- NULL
        tempFeatures$l2_subp0_read_tex_hit_sectors <- NULL
        tempFeatures$l2_subp1_read_tex_hit_sectors <- NULL
        tempFeatures$l2_subp0_read_sysmem_sector_queries <- NULL
        tempFeatures$l2_subp1_read_sysmem_sector_queries <- NULL
        tempFeatures$texture_cache_transactions <- NULL
        tempFeatures$texture_function_unit_utilization <-  NULL
        tempFeatures$fb_subp0_read_sectors <- NULL
        tempFeatures$fb_subp1_read_sectors <- NULL
        tempFeatures$fb_subp0_write_sectors <- NULL
        tempFeatures$fb_subp1_write_sectors <- NULL
        tempFeatures$l2_subp0_write_sector_misses <- NULL
        tempFeatures$l2_subp1_write_sector_misses <- NULL
        tempFeatures$l2_subp0_read_sector_misses  <- NULL
        tempFeatures$l2_subp1_read_sector_misses  <- NULL
        tempFeatures$l2_subp0_total_read_sector_queries <- NULL
        tempFeatures$l2_subp1_total_read_sector_queries <- NULL
        tempFeatures$l2_subp0_total_write_sector_queries <- NULL
        tempFeatures$l2_subp1_total_write_sector_queries <- NULL
        tempFeatures$l2_subp0_write_sysmem_sector_queries  <- NULL
        tempFeatures$l2_subp1_write_sysmem_sector_queries <- NULL
        
        tempFeatures$system_memory_write_transactions <- NULL
        tempFeatures$system_memory_write_throughput <-  NULL
        tempFeatures$system_memory_utilization <- NULL
        
        tempFeatures$inst_issued2 <- NULL
        
        
        tempFeatures$device <- NULL
        
        # tempFeatures$floating_point_operations.single_precision_special. <- NULL
        # tempFeatures$fp_instructions.double. <- NULL
        # tempFeatures$fp_instructions.single. <- NULL
        
        # tempPosFeatures <- tempFeatures$static.smem > 500
        # tempFeatures$static.smem[tempPosFeatures] <- tempFeatures$static.smem[tempPosFeatures]/1024
        # 
        # tempPosFeatures <- tempFeatures$requested_global_load_throughput > 500
        # tempFeatures$requested_global_load_throughput[tempPosFeatures] <- tempFeatures$requested_global_load_throughput[tempPosFeatures]/1024
        # 
        # tempPosFeatures <- tempFeatures$requested_global_store_throughput > 500
        # tempFeatures$requested_global_store_throughput	<- tempFeatures$requested_global_store_throughput/1024
        # 
        # tempPosFeatures <- tempFeatures$device_memory_read_throughput > 500
        # tempFeatures$device_memory_read_throughput	<- tempFeatures$device_memory_read_throughput/1024
        # 
        # tempPosFeatures <- tempFeatures$device_memory_write_throughput > 500
        # tempFeatures$device_memory_write_throughput	<- tempFeatures$device_memory_write_throughput/1024
        # 
        # tempPosFeatures <- tempFeatures$global_store_throughput > 500
        # tempFeatures$global_store_throughput	<- tempFeatures$global_store_throughput/1024
        # 
        # tempPosFeatures <- tempFeatures$global_load_throughput > 500
        # tempFeatures$global_load_throughput	<- tempFeatures$global_load_throughput/1024
        
        tempFeaturesTemporal <- tempFeatures[sample(nrow(tempFeatures), NroSamples[kernelApp]),]
        
        write.csv(tempFeaturesTemporal, file = paste("./datasets/", names(kernelsDict[kernelApp]), "-", gpus[gpu, 'gpu_name'], ".csv", sep = "" ))
        kernels <- rbind(kernels, tempFeaturesTemporal)
        
    }
    kernelsGPU <- rbind(kernelsGPU, kernels)
    
    write.csv(kernels, file = paste("./datasets/", gpus[gpu, 'gpu_name'], ".csv", sep = "" ))
    
    
    write.csv(tempComm[, c(
        "name",
        "input.size.1",
        "input.size.2",
        "start",
        "duration",
        "size",
        "throughput",
        "device",
        "context",
        "stream"
    )],
    file = paste("./datasets/Apps-", gpus[gpu, 'gpu_name'], "-comm.csv", sep =
                     ""))
    # write.csv(kernels, file = paste("./datasets/Apps-", gpus[gpu,'gpu_name'], ".csv", sep=""))
    
    # compressFile <- gzfile( paste("./datasets/Apps-", gpus[gpu,'gpu_name'], ".csv.gz", sep=""))
    # write.csv(kernels[,selectedFeatures], compressFile)
}

write.csv(kernelsGPU, file=paste("./datasets/AllAppGPUs.csv", sep = "" ))

# kernelsGPU <- data.frame()
# for (kernelApp in c(1:7, 9:13)) {
#     for (gpu in c(1, 2, 3, 5, 6, 7, 9, 10)) {
#         kernelsGPU <- rbind(kernelsGPU, read.csv(paste("./datasets/", names(kernelsDict[kernelApp]), "-", gpus[gpu, 'gpu_name'], ".csv", sep = "" ), header = T))
#     }
# }
# write.csv(kernelsGPU, file=paste("./datasets/AllAppGPUs.csv", sep = "" ))



for (kernelApp in c(1:7, 9:13)) {
    KernelGPUs <- data.frame()
    for (gpu in c(1,2,3,5, 6,7,9,10)) {
        KernelGPUs <- rbind(KernelGPUs, read.csv(paste("./datasets/", names(kernelsDict[kernelApp]), "-", gpus[gpu, 'gpu_name'], ".csv", sep = "" ), header = T))
    }
    write.csv(KernelGPUs, file = paste("./datasets/", names(kernelsDict[kernelApp]), ".csv", sep = "" ))
}
