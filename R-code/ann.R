library(ggplot2)
library(reshape2)
library(plyr)

cbbPalette <- gray(1:9/ 12) #c("red", "blue", "darkgray", "orange","black","brown", "lightblue","violet")
dirpath <- "~/Dropbox/Doctorate/GIT/BSyncGPGPU/"
setwd(paste(dirpath, sep=""))


library(ggplot2)
library(reshape2)
library(plyr)

cbbPalette <- gray(1:9/ 12) #c("red", "blue", "darkgray", "orange","black","brown", "lightblue","violet")
dirpath <- "~/Dropbox/Doctorate/GIT/BSyncGPGPU/"
setwd(paste(dirpath, sep=""))

gpus <- read.table("./data/deviceInfo.csv", sep=",", header=T)
NoGPU <- dim(gpus)[1]

namesTraces <- read.csv("./data/Tesla-K40/metricsNames.csv",header = T, sep = ",")

