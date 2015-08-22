setwd("C:\\Users\\N\\Dropbox\\Coursera\\08 - Machine Learning\\MachineLearning")
library(caret)


#read data
df=read.csv("pml-training.csv")
summary(df)

#Remove NA columns
nadims=is.na(df) #Find na values in df
outvar=colSums(nadims)>19000 #find columsn those with larger than 19000 na entries
df=df[,!outvar] 
sum(is.na(df)) #verify absence of NA values

#Remove "" columns, since they have a low of empty values
outvar=colSums(df=="")>19000
df=df[,!outvar]


#Create the following partition: train:60%,test=20%, valid=20% 
inTrain=createDataPartition(y=df$classe, p=0.2, list=FALSE,times=2)
dftrain=df[-inTrain,]
dftest=df[inTrain[,1],]
dfvalid=df[inTrain[,2],]

#Remove highly correlated variables
correlationMatrix <- cor(dftrain[,7:59])
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.8)
dftrain=dftrain[,-(highlyCorrelated+6)]


#preProc=preProcess(dftrain[,-c(1:6,60)],method="pca",thresh=0.90)
#trainPC=predict(preProc,dftrain[,-c(1:6,60)])
#Try Training
modFit2=train(classe~., data=dftrain[,c(2,8:47)], method="treebag", B=5,
             train_control = trainControl(method="cv", number=10),
             prox=TRUE,allowParallel=TRUE)

#Predict the output
pred=predict(modFit,newdata=dfvalid)
confusionMatrix(data = pred, dfvalid$classe)


#Predict the output
pred=predict(modFit,newdata=dftest)
confusionMatrix(data = pred, dftest$classe)


#Make submission data
dft=read.csv("pml-testing.csv")
pred=predict(modFit2, newdata=dft)
pred
