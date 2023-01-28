library("FrF2")
library("daewr")
library("leaps")

# first round
# 32 run with 3 resoulution
setwd("/Users/zongfan/Downloads")
des1 <- FrF2(nruns=32, nfactors=20, res.min=3)

# load result of first resoulution
data <- read.csv("casestudy2_result_round1.csv", skip=1)
data <- data[,-1:-1]
# high fitness threshold
thres<-0.5
data_p <- na.omit(data[data$fitness>thres,])
data_n <- na.omit(data[data$fitness<thres,])

# change column names
names(data)[1:20] <- names(des1)
block <- c(rep(1,32), rep(2,64))
model <- lm( fitness ~ block+(.)^2, data=data)
cfs <- na.omit(coef(model))[-1:-1]
labels <- names(cfs)
daewr::halfnorm(cfs, labels, alpha=0.25, refline=FALSE)

# select parameters with exhaustive 
modpbr <- regsubsets(fitness~(.)^2, data=data, method="exhaustive", nvmax=4, nbest=4, really.big=TRUE)
rs <- summary(modpbr) 
plot(c(rep(1:5,each=4)), rs$adjr2)
plot(modpbr, scale="r2")

# second round
# select A, B,M,L,K,R, set F=1, and rest values as -1
# resolution V
des2 <- FrF2(32, 6, res.min=5)
 