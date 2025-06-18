library(readr)
library(vars)
library(tsDyn)

y <- read_csv("data.csv")

y <- y[,2:9]
y <- data.frame(y)
# rownames(df) <- dates
y <- ts(y, frequency = 12, start = c(1987,4))

plot(y)

########################################

var1 <- VAR(y, p=6, type="const")
summary(var1)

B1<-matrix(c(0.7, 0.2, 0.2, 0.7), 2)
y1 <- VAR.sim(B=B1, n=387, include="none")
y2 <- VAR.sim(B=B1, n=387, include="none")

B2<-rbind(c(0.5, 0.5, 0.5), c(0, 0.5, 0.5))
varcov<-matrix(c(1,0.2, 0.3, 1),2)
y3 <- VAR.sim(B=B2, n=387, include="const", varcov=varcov)
y4 <- VAR.sim(B=B2, n=387, include="const", varcov=varcov)

y_sim <- data.frame(y1,y2,y3,y4)
y_sim <- y_sim[c(1,2,5,4,6,7,8,3)]

colnames(y_sim) <- colnames(y)
y_sim <- ts(y_sim, frequency = 12, start = c(1987,4))
plot.ts(y_sim)
write.csv(y_sim, '/home/jiraya/NonParametric_IRF_kNN/sim.csv')

## VAR.boot: Bootstrap a VAR 

# mod <- lineVar(data=df,lag=1)
irf_y <- irf(var1,impulse="cpu_index",  n.ahead=40, ortho=TRUE,boot=FALSE)
plot(irf_y)

#############################################

A = cbind(c(0.2,0),c(-0.3,0.4))
B = cbind(c(-0.1,0.1),c(0.2,-0.3))

varstep <- function(A,B,x,y) {
  e = rnorm(2,0,1)
  A%*%x +B%*%y + e
}

x1 = c(1,1) 
y2 = c(.5,5)


results = cbind(y2,x1)
for (t in seq(1,100))
{
  temp <- x1
  x1 <- varstep(A,B,x1,y2)
  results <- cbind(results, x1)
  y2 <- temp
}

xt = results[1,1:100]
yt = results[2,1:100]
plot(1:100,xt,type = "line")
lines(1:100,yt,col="red")