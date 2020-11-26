#REPORT STATISTICAL LEARNING CAMPISI-CASELLA
#RED WINE QUALITY DATASET

# LIBRARIES:
library(ISLR)
library(labstatR)
library(EnvStats)
library(corrplot)
library(MASS)
library(caret)
library(ROCR)
# install.packages("tree")
library(tree)
# install.packages("randomForest")
library(randomForest)
# install.packages("gbm")
library(gbm)
library(ggplot2)
library(gamlss)
library(gridExtra)
library(tidyverse)    # data manipulation and visualization
library(kernlab)      # SVM methodology
library(e1071)        # SVM methodology
library(ISLR)         # contains example data set "Khan"
library(RColorBrewer) # customized coloring of plots
library(mlbench)
library(caret)        # for easy machine learning workflow
# install.packages("neuralnet")
library(neuralnet)
library(visdat)       # visdat package: at-a-glance ggplot object of what is inside a dataframe
library(caTools)      # for data partition into training and test set
# install.packages("ggthemes")
library(ggthemes)     # for additional plotting themes
# install.packages("DataExplorer")
library(DataExplorer)


# ------------------ #
#### PREPARE DATA ####
# ------------------ #

data_train = read.csv("winequality_red_train.csv")
data_train = data_train[,-1]     # remove the first column with the index of the wine
str(data_train)
sum(!complete.cases(data_train)) # number of NA values = 0
plot_missing(data_train)
attach(data_train)




# ----------------------------- #
#### 1) DESCRIPTIVE ANALYSIS ####
# ----------------------------- #


#### A) Univariate Analysis ####

# FIXED ACIDITY:
summary(fixed.acidity)
frequencyFixedAcidity <- table(fixed.acidity) # range of values and how many time each value is present
frequencyFixedAcidity
length(frequencyFixedAcidity) # number of values
names(frequencyFixedAcidity)[frequencyFixedAcidity == max(frequencyFixedAcidity)] # mode
sd(fixed.acidity)
labstatR::cv(fixed.acidity)   # coefficient of variation. Note that the formula to calculate it is one line below
cv = sd(fixed.acidity)/mean(fixed.acidity)*100
cv
hist(fixed.acidity, freq = F, breaks = 88, main = "Fixed Acidity")
lines(density(fixed.acidity), lwd = 2, col = "darkred")
skewness(fixed.acidity)
kurtosis(fixed.acidity)
kurtosis(fixed.acidity, excess=FALSE)


# VOLATILE ACIDITY:
summary(volatile.acidity)
frequencyVolatileAcidity <- table(volatile.acidity) 
frequencyVolatileAcidity
length(frequencyVolatileAcidity) 
names(frequencyVolatileAcidity)[frequencyVolatileAcidity == max(frequencyVolatileAcidity)] 
sd(volatile.acidity)
labstatR::cv(volatile.acidity) 
cv = sd(volatile.acidity)/mean(volatile.acidity)*100
cv
hist(volatile.acidity, freq = F, breaks = 88, main= "Volatile Acidity")
lines(density(volatile.acidity), lwd = 2, col = "darkred")
skewness(volatile.acidity)
kurtosis(volatile.acidity)
kurtosis(volatile.acidity, excess=FALSE)


# CITRIC ACID:
summary(citric.acid)
frequencyCitricAcid <- table(citric.acid) 
frequencyCitricAcid
length(frequencyCitricAcid) 
names(frequencyCitricAcid)[frequencyCitricAcid == max(frequencyCitricAcid)] 
sd(citric.acid)
labstatR::cv(citric.acid) 
cv = sd(citric.acid)/mean(citric.acid)*100
cv
hist(citric.acid, freq = F, breaks = 88, main="Citric Acid")
lines(density(citric.acid), lwd = 2, col = "darkred")
skewness(citric.acid)
kurtosis(citric.acid)
kurtosis(citric.acid, excess=FALSE)


# RESIDUAL SUGAR:
summary(residual.sugar)
frequencyResidualSugar <- table(residual.sugar) 
frequencyResidualSugar
length(frequencyResidualSugar) 
names(frequencyResidualSugar)[frequencyResidualSugar == max(frequencyResidualSugar)] 
sd(residual.sugar)
labstatR::cv(residual.sugar) 
cv = sd(residual.sugar)/mean(residual.sugar)*100
cv
hist(residual.sugar, freq = F, breaks = 78, main= "Residual Sugar")
lines(density(residual.sugar), lwd = 2, col = "darkred")
skewness(residual.sugar)
kurtosis(residual.sugar)
kurtosis(residual.sugar, excess=FALSE)


# CHLORIDES:
summary(chlorides)
frequencyChlorides <- table(chlorides) 
frequencyChlorides
length(frequencyChlorides) 
names(frequencyChlorides)[frequencyChlorides == max(frequencyChlorides)] 
sd(chlorides)
labstatR::cv(chlorides) 
cv = sd(chlorides)/mean(chlorides)*100
cv
hist(chlorides, freq = F, breaks = 78, main="Chlorides")
lines(density(chlorides), lwd = 2, col = "darkred")
skewness(chlorides)
kurtosis(chlorides)
kurtosis(chlorides, excess=FALSE)


# FREE SULFUR DIOXIDE:
summary(free.sulfur.dioxide)
frequencyFreeSulfurDioxide <- table(free.sulfur.dioxide) 
frequencyFreeSulfurDioxide
length(frequencyFreeSulfurDioxide) 
names(frequencyFreeSulfurDioxide)[frequencyFreeSulfurDioxide == max(frequencyFreeSulfurDioxide)] 
sd(free.sulfur.dioxide)
labstatR::cv(free.sulfur.dioxide) 
cv = sd(free.sulfur.dioxide)/mean(free.sulfur.dioxide)*100
cv
hist(free.sulfur.dioxide, freq = F, breaks = 52, main="Free Sulfur Dioxide")
lines(density(free.sulfur.dioxide), lwd = 2, col = "darkred")
skewness(free.sulfur.dioxide)
kurtosis(free.sulfur.dioxide)
kurtosis(free.sulfur.dioxide, excess=FALSE)


# TOTAL SULFUR DIOXIDE:
summary(total.sulfur.dioxide)
frequencyTotalSulfurDioxide <- table(total.sulfur.dioxide) 
frequencyTotalSulfurDioxide
length(frequencyTotalSulfurDioxide) 
names(frequencyTotalSulfurDioxide)[frequencyTotalSulfurDioxide == max(frequencyTotalSulfurDioxide)] 
sd(total.sulfur.dioxide)
labstatR::cv(total.sulfur.dioxide) 
cv = sd(total.sulfur.dioxide)/mean(total.sulfur.dioxide)*100
cv
hist(total.sulfur.dioxide, freq = F, breaks = 135, main="Total Sulfur Dioxide")
lines(density(total.sulfur.dioxide), lwd = 2, col = "darkred")
skewness(total.sulfur.dioxide)
kurtosis(total.sulfur.dioxide)
kurtosis(total.sulfur.dioxide, excess=FALSE)


# DENSITY:
summary(density)
frequencyDensity <- table(density) 
frequencyDensity
length(frequencyDensity) 
names(frequencyDensity)[frequencyDensity == max(frequencyDensity)] 
sd(density)
labstatR::cv(density) 
cv = sd(density)/mean(density)*100
cv
hist(density, freq = F, breaks =352, main="Density")
lines(density(density), lwd = 2, col = "darkred")
skewness(density)
kurtosis(density)
kurtosis(density, excess=FALSE)


# pH:
summary(pH)
frequencypH <- table(pH) 
frequencypH
length(frequencypH) 
names(frequencypH)[frequencypH == max(frequencypH)]
sd(pH)
labstatR::cv(pH) 
cv = sd(pH)/mean(pH)*100
cv
hist(pH, freq = F, breaks = 83, main="pH")
lines(density(pH), lwd = 2, col = "darkred")
skewness(pH)
kurtosis(pH)
kurtosis(pH, excess=FALSE)


# SULPHATES:
summary(sulphates)
frequencySulphates <- table(sulphates) 
frequencySulphates
length(frequencySulphates) 
names(frequencySulphates)[frequencySulphates == max(frequencySulphates)] 
sd(sulphates)
labstatR::cv(sulphates) 
cv = sd(sulphates)/mean(sulphates)*100
cv
hist(sulphates, freq = F, breaks = 90, main="Sulphates")
lines(density(sulphates), lwd = 2, col = "darkred")
skewness(sulphates)
kurtosis(sulphates)
kurtosis(sulphates, excess=FALSE)


# ALCOHOL:
summary(alcohol)
frequencyAlcohol <- table(alcohol) 
frequencyAlcohol
length(frequencyAlcohol) 
names(frequencyAlcohol)[frequencyAlcohol == max(frequencyAlcohol)] 
sd(alcohol)
labstatR::cv(alcohol) 
cv = sd(alcohol)/mean(alcohol)*100
cv
hist(alcohol, freq = F, breaks = 58, main="Alcohol")
lines(density(alcohol), lwd = 2, col = "darkred")
skewness(alcohol)
kurtosis(alcohol)
kurtosis(alcohol, excess=FALSE)


# QUALITY:
summary(quality)
frequencyQuality <- table(quality) 
frequencyQuality
length(frequencyQuality) 
names(frequencyQuality)[frequencyQuality == max(frequencyQuality)] 
sd(quality)
labstatR::cv(quality) 
cv = sd(quality)/mean(quality)*100
cv
skewness(quality)
kurtosis(quality)
kurtosis(quality, excess=FALSE)
table(quality)
round(table(quality)/length(quality)*100,2)
barplot(sort(table(quality)), col = "red", main = "Barplot of Quality", horiz = T, las = 1)
hist(quality, freq = F, main="Quality")
lines(density(quality), lwd = 2, col = "darkred")



#### B) Bivariate Analysis ####

cor(data_train)

ValueQuality <- ifelse(data_train$quality < 6, 0, 1)  # ValueQuality is 0 if quality = 4 o 5
data_train1 = data.frame(data_train, ValueQuality)    # create a dataframe with valuequality
str(data_train1)                                      # check if data are  correct
head(data_train1$ValueQuality, 957)                   # they are correct

grid.arrange(ggplot(data_train1, aes(fixed.acidity))       + geom_histogram(binwidth = 1, position = "fill", aes(fill=factor(ValueQuality))) + scale_fill_discrete(name = "ValueQuality") + ylab("proportion") + geom_hline(yintercept = 0.5), 
             ggplot(data_train1, aes(volatile.acidity))    + geom_histogram(binwidth = 0.5, position = "fill", aes(fill=factor(ValueQuality))) + scale_fill_discrete(name = "ValueQuality") + ylab("proportion") + geom_hline(yintercept = 0.5), 
             ggplot(data_train1, aes(citric.acid))         + geom_histogram(binwidth = 0.2, position = "fill", aes(fill=factor(ValueQuality))) + scale_fill_discrete(name = "ValueQuality") + ylab("proportion") + geom_hline(yintercept = 0.5), 
             ggplot(data_train1, aes(residual.sugar))      + geom_histogram(binwidth = 2, position = "fill", aes(fill=factor(ValueQuality))) + scale_fill_discrete(name = "ValueQuality") + ylab("proportion") + geom_hline(yintercept = 0.5), 
             ggplot(data_train1, aes(chlorides))           + geom_histogram(binwidth = 0.1, position = "fill", aes(fill=factor(ValueQuality))) + scale_fill_discrete(name = "ValueQuality") + ylab("proportion") + geom_hline(yintercept = 0.5), 
             ggplot(data_train1, aes(free.sulfur.dioxide)) + geom_histogram(binwidth = 15, position ="fill", aes(fill=factor(ValueQuality))) + scale_fill_discrete(name = "ValueQuality") + ylab("proportion") + geom_hline(yintercept = 0.5), 
             ncol = 2, nrow = 3, top = "Conditioned distributions given ValueQuality")  # "binwidth" adjusts the width of the bars; we choose binwidth to contain as many più values as possible.

grid.arrange(ggplot(data_train1, aes(total.sulfur.dioxide)) + geom_histogram(binwidth = 100, position = "fill", aes(fill=factor(ValueQuality))) + scale_fill_discrete(name = "ValueQuality") + ylab("proportion") + geom_hline(yintercept = 0.5), 
             ggplot(data_train1, aes(density)) + geom_histogram(binwidth = 0.001, position = "fill", aes(fill=factor(ValueQuality))) + scale_fill_discrete(name = "ValueQuality") + ylab("proportion") + geom_hline(yintercept = 0.5), 
             ggplot(data_train1, aes(pH)) + geom_histogram(binwidth = 0.1, position = "fill", aes(fill=factor(ValueQuality))) + scale_fill_discrete(name = "ValueQuality") + ylab("proportion") + geom_hline(yintercept = 0.5), 
             ggplot(data_train1, aes(sulphates)) + geom_histogram(binwidth = 0.5, position = "fill", aes(fill=factor(ValueQuality))) + scale_fill_discrete(name = "ValueQuality") + ylab("proportion") + geom_hline(yintercept = 0.5), 
             ggplot(data_train1, aes(alcohol)) + geom_histogram(binwidth = 1, position = "fill", aes(fill=factor(ValueQuality))) + scale_fill_discrete(name = "ValueQuality") + ylab("proportion") + geom_hline(yintercept = 0.5), 
             ncol = 2, nrow = 3, top = "Conditioned distributions given ValueQuality")






# ------------------------------- #
#### 2) MODELING TRAINING DATA ####
# ------------------------------- #


# ---------------------------- #
#### A) Logistic Regression ####
# ---------------------------- #

logR = step(glm(ValueQuality~.-quality, data = data_train1, family = binomial), 
            direction = "both") #  "~.-quality" means that we use all the variables,  except quality
summary(logR)

glm.probs1 <- predict.glm(logR, type = "response") # The predict() function can be used to predict probabilities, 
                                                   # given values of the predictors. The type="response" option tells R to output probabilities 
                                                   # of the form P(Y = 1|X), as opposed to other information such as the logit.
head(glm.probs1)  

N <- dim(data_train1)[1]              # dim(data_train1)[1] = 957 rows
glm.pred1 <- rep("0", N)            # assign "0" (Bad) for each unit with prob < 0.5
glm.pred1[glm.probs1 > 0.5] = "1"   # assign "1" (Excellent) for units with P(Y="1")>0.5
glm.pred1                           # look at the results

# Confusion matrix:
table(glm.pred1, data_train1$ValueQuality)

confMat1 <- addmargins(table(glm.pred1, data_train1$ValueQuality))
confMat1

delta1 <-(confMat1[1,2]+confMat1[2,1])/N*100 # misclassification error rate
delta1

tpr <- round(confMat1[2,2]/confMat1[3,2]*100, 2) 
tpr

fpr <- round(confMat1[2,1]/confMat1[3,1]*100, 2)
fpr

# ROC curve:
pred.t.LR_T=ROCR::prediction(glm.probs1, ValueQuality) 

# Need to specify ROCR:: because If you have library(neuralnet) open,
# it overrides the "prediction" function in ROCR and generates this error. 
# Double check that neuralnet, or any other package that may use a "prediction" function, 
# are detached.

perf.t.LR_T=performance(pred.t.LR_T, measure = "tpr", x.measure = "fpr") 
plot(perf.t.LR_T,colorize=TRUE,lwd=2, print.cutoffs.at=c(0.2,0.5,0.8)) 
abline(a=0,b=1, lty=2)
perf <- performance(pred.t.LR_T, measure = "auc", x.measure = "fpr")

#AUC:
AUC <- performance(pred.t.LR_T, measure = "auc", x.measure = "fpr")
AUC@y.values[[1]]


#DATA VALIDATION SET:
data_validation = read.csv("winequality_red_validation.csv")
str(data_validation)
data_validation = data_validation[,-1]  # remove the first column, with the index of the wine
sum(!complete.cases(data_validation))   # number of NA values = 0
plot_missing(data_validation)
attach(data_validation)

ValidQuality <- ifelse(data_validation$quality < 6, 0, 1) 
data_validation1 = data.frame(data_validation, ValidQuality) 
str(data_validation1) 
head(data_validation1$ValidQuality, 324) 

glm.probs1val <- predict.glm(logR, data_validation1, type = "response")
Nval<-dim(data_validation1)[1] 
glm.pred1val <- rep("0", Nval) 
glm.pred1val[glm.probs1val > 0.5] = "1" 
glm.pred1val 

#confusion matrix:
table(glm.pred1val, data_validation1$ValidQuality)
confMat1val <- addmargins(table(glm.pred1val, data_validation1$ValidQuality))
confMat1val
delta1val <-(confMat1val[1,2]+confMat1val[2,1])/N*100 
delta1val


# --------------------------------- #
#### B) Support Vector Machines #####
# --------------------------------- #

data_train_svm <- data_train1

# the main advantage of scaling is to avoid attributes in greater numeric ranges dominating
# those in smaller numeric ranges.  Another advantage is to avoid numerical difficulties
# during the calculation.  Because kernel values usually depend on the inner products of feature
# vectors, e.g.  the linear kernel and the polynomial kernel, large attribute values
# might cause numerical problems.
# We do normalization, because standardization assumes that data are normally distributed within each feature

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

data_train_svm <- normalize(data_train_svm)
str(data_train_svm)

data_validation_svm <- data_validation1
data_validation_svm <- normalize(data_validation_svm)
str(data_validation_svm)


# let's start with Linear Kernel:
set.seed(1)

# A cost argument allows us to specify the cost of
# a violation to the margin. When the cost argument is small, then the margins
# will be wide and many support vectors will be on the margin or will violate 
# the margin. 
# When the cost argument is large, then the margins will be narrow 
# and there will be few support vectors on the margin or violating the margin.

# The following command indicates that we want to compare SVMs with a linear
# kernel, using a range of values of the cost parameter C:

cost=c(0.001, 0.01, 0.1, 1,5,10,100)

# The e1071 library contains implementations for a number of statistical
# learning methods. In particular, the svm() function can be used to fit a
# support vector classifier when the argument kernel="linear" is used.

names_svc = c()
for(i in 1:7){
  names_svc = c(names_svc, paste("svc", i, sep = ""))
}
SVC = c()
error_svc = c()
for(i in 1:7){
  cat(paste("Computing the svc", i, "of 7 ..."), "\n" )
  name = names_svc[i]
  svmfit = svm(as.factor(data_train_svm$ValueQuality) ~ fixed.acidity + volatile.acidity + 
                 residual.sugar + chlorides + free.sulfur.dioxide + total.sulfur.dioxide + 
                 density + sulphates + alcohol, data_train_svm, kernel = "linear", cost = cost[i], scale=F)
  SVC[[name]] = svmfit
  ypred.train <- predict(svmfit, new_dataset <- data_validation_svm)
  tab = addmargins(table(ypred.train, data_validation_svm$ValidQuality))
  error_svc_i = (tab[1,2]+tab[2,1])/tab[3,3]
  error_svc = c(error_svc,error_svc_i)
}

dsvc = data.frame(model = names_svc, cost = format(cost, scientific=F), misc.error = error_svc)
dsvc

bmsvc = dsvc[which.min(dsvc$misc.error), ]
bmsvc$misc.error = bmsvc$misc.error*100
names(bmsvc)= c("Best SVC model", "cost", "Misclassification error on validation set (%)")
bmsvc

# now Polynomial Kernel:
set.seed(1)
PSVM = c()
names_psvm = c()
error_psvm = c()
degree = c(2,3,4,5,6,7,8,9,10)
for (i in 1:63){ #degree * cost = 9*7=63
  names_psvm = c(names_psvm, paste("psvm", i, sep = "")) 
}
j=1
for (d in degree){
  for (c in cost){
    cat(paste("Computing the polynomial svm", j, "of 63 ..."), "\n" )
    name = names_psvm[j]
    spvcm = svm(as.factor(data_train_svm$ValueQuality) ~ fixed.acidity + volatile.acidity + 
                  residual.sugar + chlorides + free.sulfur.dioxide + total.sulfur.dioxide + 
                  density + sulphates + alcohol, data = data_train_svm, kernel = "polynomial", cost = c, degree = d, scale=F)
    PSVM[[name]] = spvcm
    ypred_train.spvcm = predict(spvcm, new_dataset <- data_validation_svm)
    tab_svm = addmargins(table(ypred_train.spvcm, data_validation_svm$ValidQuality)) 
    error_psvm_i = (tab_svm[1,2]+tab_svm[2,1])/tab_svm[3,3]
    error_psvm = c(error_psvm,error_psvm_i)
    j=j+1
  }
}

dpsvm = data.frame(model = names_psvm, cost = rep(format(cost, scientific = F ), 9), degree = rep(degree, each = 7), misc.error = error_psvm)
dpsvm

bmpsvm = dpsvm[which.min(dpsvm$misc.error), ]
bmpsvm$misc.error = bmpsvm$misc.error*100
names(bmpsvm)= c("Best polynomial SVM model", "cost", "degree", "Misclassific ation error on validation set (%)")
bmpsvm


# now Radial Kernel:
set.seed(1)
RSVM = c()
names_rsvm = c()
error_rsvm = c()
for (i in 1:35){
  names_rsvm = c(names_rsvm, paste("rsvm", i, sep = ""))
}
gamma = c(0.5,1,2,3,4)
j=1
for (g in gamma){
  for (c in cost){
    cat(paste("Computing the radial svm", j, "of 35 ..."), "\n" )  #35 = 5*7
    name = names_rsvm[j]
    spvcm = svm(as.factor(data_train_svm$ValueQuality) ~ fixed.acidity + volatile.acidity + 
                  residual.sugar + chlorides + free.sulfur.dioxide + total.sulfur.dioxide + 
                  density + sulphates + alcohol, data = data_train_svm, kernel = "radial", cost = c, gamma = g, scale=F)
    RSVM[[name]] = spvcm
    pred.spvcm = predict(spvcm, new_dataset <- data_validation_svm)
    tab_rad = addmargins(table(pred.spvcm, data_validation_svm$ValidQuality)) 
    error_rsvm_i = (tab_rad[1,2]+tab_rad[2,1])/tab_rad[3,3]
    error_rsvm = c(error_rsvm,error_rsvm_i)
    j=j+1
  }
}

drsvm = data.frame(model = names_rsvm, cost = rep(format(cost, scientific = F), 5), gamma = rep(gamma, each = 7), misc.error = error_rsvm)
drsvm

bmrsvm = drsvm[which.min(drsvm$misc.error), ]
bmrsvm$misc.error = bmrsvm$misc.error*100
names(bmrsvm) = c("Best radial SVM model", "cost", "gamma", "Misclassificatio > n error on validation set (%)")
bmrsvm

# comparing the results:
bmsvc
bmpsvm
bmrsvm

#the best is RSVM 35:
summary(RSVM$rsvm35)

# comparing the results: 
data.frame(row.names = c("LinearK", "PolynomialK", "RadialK"), 
                          Best_model  = c(bmsvc$`Best SVC model`, bmpsvm$`Best polynomial SVM model`, bmrsvm$`Best radial SVM model`),
                          Misc_error =  c(bmsvc$`Misclassification error on validation set (%)`, bmpsvm$`Misclassific ation error on validation set (%)`, bmrsvm$`Misclassificatio > n error on validation set (%)`)
                          )






# -------------------------- #
#### C) Neural Networks  #####
# -------------------------- #

# To avoid this error: Error in if (ncol.matrix < rep) { : argument is of length zero
# or other similar errors, DO THIS:

# install.packages("devtools")
library(devtools)
# devtools::install_github("bips-hb/neuralnet")
library(neuralnet)

data_train_nn <- data_train1
str(data_train_nn)

data_validation_nn <- data_validation1
str(data_validation_nn)


set.seed(100)
NN = c()
names_nn = c()
error_nn = c()
for (i in 1:5){
  names_nn = c(names_nn, paste("nn", i, sep = ""))
}
i=1 

# 1 hidden layer with 1 to 5 number of neurons:

for (i in 1:5){
  cat(paste("Computing the nn", i, "of 5 ..."), "\n" )
  name = names_nn[i]
  nn_i = neuralnet(ValueQuality~fixed.acidity + volatile.acidity + 
                     residual.sugar + chlorides + free.sulfur.dioxide + total.sulfur.dioxide + 
                     density + sulphates + alcohol, data = data_train_nn, hidden = i, threshold = 0.1, stepmax = 1e7, linear.output = F, rep = 3)
  #i increased stepmax from the default value 1e5, to 1e7, because stepmax is the maximal count of
  #all gradient steps, and using value 1e5 gives the following warning:
  #Warning message:
  #Algorithm did not converge in 3 of 3 repetition(s) within the stepmax.
  NN[[name]] = nn_i
  i_best = which.min(nn_i$result.matrix[1,])
  yhat = round(nn_i$net.result[[i_best]])
  pred.nn = round(predict(nn_i, newdata = data_validation_nn, rep = i_best))
  tab_nn = addmargins(table(pred.nn, data_validation_nn$ValidQuality))
  error_nn_i = (tab_nn[1,2]+tab_nn[2,1])/tab_nn[3,3]
  error_nn = c(error_nn, error_nn_i)
}



# 2 hidden layers with 2 and 1 neuron:
set.seed(100)
name = "nn6"
nn_i = neuralnet(ValueQuality~fixed.acidity + volatile.acidity + 
                   residual.sugar + chlorides + free.sulfur.dioxide + total.sulfur.dioxide + 
                   density + sulphates + alcohol, data = data_train_nn, hidden = c(2,1), threshold = 0.1, stepmax = 1e7, linear.output = F, rep = 3)
NN[[name]] = nn_i
i_best = which.min(nn_i$result.matrix[1,])
yhat = round(nn_i$net.result[[i_best]])
pred.nn = round(predict(nn_i, newdata = data_validation_nn, rep = i_best))
tab_nn6 = addmargins(table(pred.nn, data_validation_nn$ValidQuality))
error_nn_i = (tab_nn6[1,2]+tab_nn6[2,1])/tab_nn6[3,3]
error_nn = c(error_nn,error_nn_i)



# 2 neural networks with 3 and 2 neurons:
set.seed(100)
name = "nn7"
nn_i = neuralnet(ValueQuality~fixed.acidity + volatile.acidity + 
                   residual.sugar + chlorides + free.sulfur.dioxide + total.sulfur.dioxide + 
                   density + sulphates + alcohol, data = data_train_nn, hidden = c(3,2), threshold = 0.1,stepmax = 1e7, linear.output = F, rep = 3)
NN[[name]] = nn_i
i_best = which.min(nn_i$result.matrix[1,])
yhat = round(nn_i$net.result[[i_best]])
pred.nn = round(predict(nn_i, newdata = data_validation_nn, rep = i_best))
tab_nn7 = addmargins(table(pred.nn, data_validation_nn$ValidQuality))
error_nn_i = (tab_nn7[1,2]+tab_nn7[2,1])/tab_nn7[3,3]
error_nn = c(error_nn,error_nn_i)



names_nn = c(names_nn ,"nn6", "nn7")
dnn = data.frame(model = names_nn, hidden = c(1:5,"(2,1)", "(3,2)"), misc.error = error_nn)
bmnn = dnn[which.min(dnn$misc.error), ]
bmnn$misc.error = bmnn$misc.error*100
names(bmnn) = c("Best nn model", "hidden", "Misclassification error on validation set (%)")
bmnn

plot(NN$nn4)




# ------------------------------------------------------- #
#### 3) COMPARE THE RESULTS AND CHOOSE THE BEST MODEL  ####
# ------------------------------------------------------- #

delta1
bmsvc
bmpsvm
bmrsvm
bmnn
best_model <- NN$nn4




# --------------------------------- #
#### 4) PREDICT TARGET VALUES   #####
# --------------------------------- #

data_test <- read.csv("winequality_red_test.csv")
str(data_test)
data_test = data_test[,-c(1,13)]
str(data_test)

predicted_prob <- predict(best_model, new_dataset <- data_test)

N <- dim(data_test)[1] 

predicted_values <- rep("0", N)
predicted_values[predicted_prob > 0.5] = 1

table(predicted_values)

data_test_predicted = data.frame(data_test, predicted_qualities=predicted_values)
write.csv(data_test_predicted, "data_test_predicted.csv")







# ----------------------------------------- #
#### 6) BUILD SVM WITH  MULTIPLE CLASSES ####
# ----------------------------------------- #

data_train = read.csv("winequality_red_train.csv")
data_train = data_train[,-1]      # remove the first column with the index of the wine
str(data_train)
sum(!complete.cases(data_train))  # number of NA values = 0
plot_missing(data_train)
attach(data_train)

data_validation = read.csv("winequality_red_validation.csv")
str(data_validation)
data_validation = data_validation[,-1]  # remove the first column, with the index of the wine
sum(!complete.cases(data_validation))   # number of NA values = 0
plot_missing(data_validation)
attach(data_validation)


data_train_multiple_svm <- data_train


normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

data_train_multiple_svm <- normalize(data_train_multiple_svm)
str(data_train_multiple_svm)

data_validation_multiple_svm <- data_validation
data_validation_multiple_svm <- normalize(data_validation_multiple_svm)
str(data_validation_multiple_svm)


# let's start with Linear Kernel:
set.seed(1)

cost=c(0.001, 0.01, 0.1, 1,5,10,100)

names_svc = c()
for(i in 1:7){
  names_svc = c(names_svc, paste("svc", i, sep = ""))
}
SVC = c()
error_svc = c()
for(i in 1:7){
  cat(paste("Computing the svc", i, "of 7 ..."), "\n" )
  name = names_svc[i]
  multiple_svmfit = svm(as.factor(data_train_multiple_svm$quality) ~ fixed.acidity + volatile.acidity + 
                 residual.sugar + chlorides + free.sulfur.dioxide + total.sulfur.dioxide + 
                 density + sulphates + alcohol, data_train_multiple_svm, kernel = "linear", cost = cost[i], scale=F)
  SVC[[name]] = multiple_svmfit
  ypred.train <- predict(multiple_svmfit, new_dataset <- data_validation_multiple_svm)
  tab = addmargins(table(ypred.train, data_validation_multiple_svm$quality))
  error_svc_i = (tab[1,2]+tab[2,1])/tab[3,3]
  error_svc = c(error_svc,error_svc_i)
}

dsvc = data.frame(model = names_svc, cost = format(cost, scientific=F), misc.error = error_svc)
dsvc

bmsvc = dsvc[which.min(dsvc$misc.error), ]
bmsvc$misc.error = bmsvc$misc.error*100
names(bmsvc)= c("Best SVC model", "cost", "Misclassification error on validation set (%)")
bmsvc

# now Polynomial Kernel:
set.seed(1)
PSVM = c()
names_psvm = c()
error_psvm = c()
degree = c(2,3,4,5,6,7,8,9,10)
for (i in 1:63){ #degree * cost = 9*7=63
  names_psvm = c(names_psvm, paste("psvm", i, sep = "")) 
}
j=1
for (d in degree){
  for (c in cost){
    cat(paste("Computing the polynomial svm", j, "of 63 ..."), "\n" )
    name = names_psvm[j]
    spvcm = svm(as.factor(data_train_multiple_svm$quality) ~ fixed.acidity + volatile.acidity + 
                  residual.sugar + chlorides + free.sulfur.dioxide + total.sulfur.dioxide + 
                  density + sulphates + alcohol, data = data_train_multiple_svm, kernel = "polynomial", cost = c, degree = d, scale=F)
    PSVM[[name]] = spvcm
    ypred_train.spvcm = predict(spvcm, new_dataset <- data_validation_multiple_svm)
    tab_svm = addmargins(table(ypred_train.spvcm, data_validation_multiple_svm$quality)) 
    error_psvm_i = (tab_svm[1,2]+tab_svm[2,1])/tab_svm[3,3]
    error_psvm = c(error_psvm,error_psvm_i)
    j=j+1
  }
}

dpsvm = data.frame(model = names_psvm, cost = rep(format(cost, scientific = F ), 9), degree = rep(degree, each = 7), misc.error = error_psvm)
dpsvm

bmpsvm = dpsvm[which.min(dpsvm$misc.error), ]
bmpsvm$misc.error = bmpsvm$misc.error*100
names(bmpsvm)= c("Best polynomial SVM model", "cost", "degree", "Misclassific ation error on validation set (%)")
bmpsvm


# now Radial Kernel:
set.seed(1)
RSVM = c()
names_rsvm = c()
error_rsvm = c()
for (i in 1:35){
  names_rsvm = c(names_rsvm, paste("rsvm", i, sep = ""))
}
gamma = c(0.5,1,2,3,4)
j=1
for (g in gamma){
  for (c in cost){
    cat(paste("Computing the radial svm", j, "of 35 ..."), "\n" )  #35 = 5*7
    name = names_rsvm[j]
    spvcm = svm(as.factor(data_train_multiple_svm$quality) ~ fixed.acidity + volatile.acidity + 
                  residual.sugar + chlorides + free.sulfur.dioxide + total.sulfur.dioxide + 
                  density + sulphates + alcohol, data = data_train_multiple_svm, kernel = "radial", cost = c, gamma = g, scale=F)
    RSVM[[name]] = spvcm
    pred.spvcm = predict(spvcm, new_dataset <- data_validation_multiple_svm)
    tab_rad = addmargins(table(pred.spvcm, data_validation_multiple_svm$quality)) 
    error_rsvm_i = (tab_rad[1,2]+tab_rad[2,1])/tab_rad[3,3]
    error_rsvm = c(error_rsvm,error_rsvm_i)
    j=j+1
  }
}

drsvm = data.frame(model = names_rsvm, cost = rep(format(cost, scientific = F), 5), gamma = rep(gamma, each = 7), misc.error = error_rsvm)
drsvm

bmrsvm = drsvm[which.min(drsvm$misc.error), ]
bmrsvm$misc.error = bmrsvm$misc.error*100
names(bmrsvm) = c("Best radial SVM model", "cost", "gamma", "Misclassificatio > n error on validation set (%)")
bmrsvm

# comparing the results:
bmsvc
bmpsvm
bmrsvm

#the best is RSVM 35:
summary(PSVM$psvm7)




