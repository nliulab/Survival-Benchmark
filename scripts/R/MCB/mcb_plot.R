remove(list = ls())
library(tidyr)
library(dplyr)
set.seed(1)
nn <- 20

source("scripts/R/functions.R")
new_table <- read.csv("results/MCB_cindex.csv",stringsAsFactors = TRUE)
new_table_cindex = subset(new_table, select = -c(1))

sample_data_cindex <- apply(new_table_cindex, 2, FUN = function(x) generate_sample(nn, x[1], x[2], x[3]))
rank_ew_cindex <- as.data.frame(sample_data_cindex)

#=========================================================================================
# Integrate IBS 
#=========================================================================================
new_table_ibs <- read.csv("results/MCB_ibs.csv",stringsAsFactors = TRUE)
new_table_ibs = subset(new_table_ibs, select = -c(1))

sample_data_ibs <- apply(new_table_ibs, 2, FUN = function(x) generate_sample(nn, x[1], x[2], x[3]))
rank_ew_ibs <- as.data.frame(sample_data_ibs)


#===============================================================
# Plot SUM 
#===============================================================
win.graph(h=6, w=12,pointsize = 12)
par(mfrow=c(1,2))
SUM_cindex <- tsutils::nemenyi(as.matrix(1-rank_ew_cindex), conf.level = 0.95, plottype = "mcb", main = "", ylab = "", ylim=c(0, 16))
title("MCB plot for C-index measure", line = 0.8)
mtext(expression(paste( plain("Mean rank"))), side=2, line=2.8, padj=1, at=8, cex=1.2)
mtext(expression(paste( plain("A"))),  line=1.5, at=-0.5, cex=1.2)

SUM_ibs <- tsutils::nemenyi(as.matrix(rank_ew_ibs), conf.level = 0.95, plottype = "mcb", main = "", ylab = "", ylim=c(0, 16))
title("MCB plot for IBS measure", line = 0.8)
mtext(expression(paste( plain("Mean rank"))), side=2, line=2.8, padj=1, at=8, cex=1.2)
mtext(expression(paste( plain("B"))),  line=1.5, at=-0.5, cex=1.2)

