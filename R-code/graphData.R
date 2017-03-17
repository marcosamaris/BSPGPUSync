library(ggplot2)

cbbPalette <- gray(1:9/ 12)#c("red", "blue", "darkgray", "orange","black","brown", "lightblue","violet")
dirpath <- "~/Dropbox/Doctorate/GIT/BSyncGPGPU/"
setwd(paste(dirpath, sep=""))

gpus <- read.table("./data/deviceInfo.csv", sep=",", header=T)
NoGPU <- dim(gpus)[1]
metricsNames <- read.csv("./data/Tesla-K40/metricsName.csv",header = T, sep = ",")

# apps <- c("backprop", "gaussian", "heartwall",  "hotspot", "hotspot3D", "lavaMD", "lud", "nw") 

appListMetrics <- list()
appListMetrics$gaussian <- data.frame(read.csv(paste("./data/", gpus[2,'gpu_name'],"/" , "metrics", "/gaussian-metrics.csv", sep=""), header=F,  col.names = c("Input.Size",names(metricsNames))))


TempFan2 <- appListMetrics$gaussian[grep("^Fan2", appListMetrics$gaussian$Kernel), ]
# head(TempFan2)["Kernel"]


MetricPlot <- "Achieved.Occupancy"
TempFan2Plot <- data.frame(row.names =  c("Input.Size", MetricPlot))

TempFan2Plot <- rbind(TempFan2Plot,data.frame("Input.Size"= rep("16", length(subset(TempFan2, Input.Size == 16)[MetricPlot][,])), 
                      subset(TempFan2, Input.Size == 16)[MetricPlot]))

TempFan2Plot <- rbind(TempFan2Plot,data.frame("Input.Size"= rep("64", length(subset(TempFan2, Input.Size == 64)[MetricPlot][,])), 
                                 subset(TempFan2, Input.Size == 64)[MetricPlot]))

TempFan2Plot <- rbind(TempFan2Plot,data.frame("Input.Size"= rep("256", length(subset(TempFan2, Input.Size == 256)[MetricPlot][,])), 
                                 subset(TempFan2, Input.Size == 256)[MetricPlot]))

TempFan2Plot <- rbind(TempFan2Plot,data.frame("Input.Size"= rep("1024", length(subset(TempFan2, Input.Size == 1024)[MetricPlot][,])), 
                                 subset(TempFan2, Input.Size == 1024)[MetricPlot]))

TempFan2Plot <- rbind(TempFan2Plot, data.frame("Input.Size"= rep("2048", length(subset(TempFan2, Input.Size == 2048)[MetricPlot][,])), 
                                 subset(TempFan2, Input.Size == 2048)[MetricPlot]))

View(TempFan2Plot)


Graph <- ggplot(data=TempFan2Plot, aes(x=Input.Size, y=Achieved.Occupancy, group=Achieved.Occupancy, col=Input.Size)) + 
    geom_boxplot( size=1) 

Graph


Achieved.Occupancy
Executed.IPC
L2.Hit.Rate..L1.Reads.


