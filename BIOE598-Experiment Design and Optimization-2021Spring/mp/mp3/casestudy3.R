# ref: https://cran.r-project.org/web/packages/rsm/vignettes/rsm.pdf

library("rsm")
library("daewr")

data <- read.csv("/Users/zongfan/Downloads/test_data_hoke_4.csv")
data_c <- coded.data(data, x1~(FH-0.45)/0.15,x2~(CM-12.5)/4,x3~(SL-0.4)/0.1)
SO_model <- rsm(ERR~SO(x1,x2,x3),data=data_c)
SO_model_sim <- rsm(ERR~FO(x1,x2,x3)+TWI(x1,x3),TWI(x2,x3)+PQ(x2,x3),data=data_c)
# variation function plot
varfcn(data_c, ~SO(x1,x2,x3), dist=seq(0,3,0.1))
varfcn(data_c, ~SO(x1,x2,x3), dist=seq(0,3,0.1), contour=TRUE)
summary(SO_model)
summary(SO_model_sim)
# steepest(SO_model, descent=TRUE)
# steepest(SO_model, descent=TRUE, dist=seq(0,2,0.2))
# surface response contour plot
contour(SO_model_sim, ~x1+x2+x3, image=TRUE)
# contour(SO_model, ~x1+x2+x3,image=TRUE,varfcn(data, ~SO(x1,x2,x3), dist=seq(0,3,0.1))at=summary(SO_model)$canonical$xs)
contour(SO_model_sim, x2~x3,image=TRUE,at=data.frame(x1=0.1634500))
contour(SO_model_sim, x1~x3,image=TRUE,at=data.frame(x2=0.2932412))
contour(SO_model_sim, x1~x2,image=TRUE,at=data.frame(x3=-0.2711756))

# stationary point
sp <- data.frame(x1=c(-0.1634500), x2=c(0.2932412), x3=c(-0.2711756))
sp_code <- code2val(sp, codings(data_c))
# predict stationay point value
sp_value <- predict(SO_model_sim, sp)
# canonical path
canonical.path(SO_model_sim)

# test code
inf_fh <- c(0.42, 0.42, 0.42, 0.42, 0.42, 0.41, 0.41, 0.41) 
inf_cm <- c(13.7, 13.68, 13.69, 13.7, 13.7, 13.7, 13.7, 13.7)
inf_sl <- c(0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37, 0.37)
inf_data <- data.frame(FH=inf_fh, CM=inf_cm, SL=inf_sl)
inf_code <- val2code(inf_data, codings(data_c))

# analyze error 
yhat <- c(0.7, 0.96, 0.97, 0.2, 1.06, 0.53, 0.36, 1.3)
mean(yhat)
sd(yhat)
# statistic distance distriction in training data
library("matrixStats")
d_counts <- binCounts(data$D, bx=seq(10,16,1))
stat_bin <- c("10-11", "11-12", "12-13", "13-14", "14-15", "15-16")
stat_count <- data.frame(bin=stat_bin, count=d_counts)
barplot(height=stat_count$count, names=stat_count$bin,
        main="Statistics of Distance Distribution",
        xlab="Distance Bin",
        ylab="Count")
