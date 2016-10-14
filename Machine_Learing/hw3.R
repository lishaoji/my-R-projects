winequality.red = read.csv("~/STA 561/winequality-red.csv", sep=";")
class = as.numeric(winequality.red$quality > 5)
winequality.red$Class = as.factor(class)
levels(winequality.red$Class) <- make.names(levels(factor(winequality.red$Class)))
winequality.red <- subset(winequality.red, select = -quality)

library(caret)
library(pROC)

## CV
set.seed(1984)

# separate data into 10 folds
folds <- createFolds(winequality.red$Class, k = 10, list=TRUE)
colors = c("red", "green", "blue", "purple", "black", "navy", "yellow", "violet", "orange", "grey")

## For each feature
auccc = matrix(rep(NA, 22),nrow = 11, ncol = 2)
for (i in 1:11) {
  auc.f = NULL
  for (j in 1:10) {
    testData = winequality.red[folds[[j]],]
    col = names(winequality.red)[i]
    feature.roc = pROC::roc(testData$Class, testData[[col]])
    feature.auc = pROC::auc(feature.roc)
    if (j == 1) {
      plot(feature.roc, print.auc = T, print.auc.x=0.6, print.auc.y=0.5, print.auc.col=colors[j], type="l", xlab = names(winequality.red)[i], col=colors[j], lwd=1, lty=1)
    } else {
      plot(feature.roc, print.auc = T, print.auc.x=0.6, print.auc.y=0.55 - 0.05 * j, print.auc.col=colors[j], type="l", add=TRUE, col=colors[j], lwd=1, lty=1)
    }
    auc.f = c(auc.f, feature.auc)
  }
  legend("bottomright", legend=c("fold 1", "fold 2", "fold 3", "fold 4", "fold 5", "fold 6", "fold 7", "fold 8", "fold 9", "fold 10"), col=colors, lwd=1, cex = 0.5)
  auccc[i,1] = mean(auc.f)
  auccc[i,2] = sd(auc.f)
}
auccc = as.data.frame(auccc, row.names = names(winequality.red[1:11]),col.names = c("mean","sd"))


## Cross Validation on each algorithm
fitControl <- trainControl(method = "cv",
                           number = 10,
                           # Estimate class probabilities
                           classProbs = TRUE,
                           # Evaluate performance using
                           # the following function
                           summaryFunction = twoClassSummary)
models = c("logistic regression", "svm", "Boosted Trees", "C5.0", "rForest")
methods = c("glm", "svmRadial", "gbm", "C5.0", "rf")
auc.list = matrix(rep(NA,10), nrow = 5, ncol = 2)
for (k in 1:5) {
  auc.k = NULL
  for (i in 1:10) {
    train_idx = unlist(folds[c(1:10)[-i]])
    train = winequality.red[train_idx, ]
    test = winequality.red[-train_idx, ]
    if (k == 1) {
      model = glm(Class ~ . , data=train, family=binomial)
      pred.model = predict(model, newdata=test, type="response")
    } else {
      model = train(Class ~ ., data=train, method = methods[k], metric="ROC", trControl = fitControl, verbose=FALSE, tuneLength=5)
      pred.model = as.vector(predict(model, newdata=test, type="prob")[, "X1"])
    }
    # Plot ROC
    sort.pred.model = sort(pred.model)
    levels = c(0, sort.pred.model[-length(pred.model)] + diff(sort.pred.model)/2)
    TPR = sapply(levels, function (x) {
      rate = sum(pred.model > x & test$Class == "X1") / sum(test$Class == "X1")
      return(rate)
    })
    FPR = sapply(levels, function (x) {
      rate = sum(pred.model > x & test$Class == "X0") / sum(test$Class == "X0")
      return(rate)
    })
    if (i == 1) {
      plot(FPR, TPR, type = "l", col = colors[i], main = models[k])
    } else {
      par(new = TRUE)
      plot(FPR, TPR, type = "l", col = colors[i], axes = FALSE, xlab = "", ylab = "")
      par(new = FALSE)
    }
    auc.k = c(auc.k, pROC::auc(pROC::roc(test$Class,pred.model)))
  }
  auc.list[k,1] = mean(auc.k)
  auc.list[k,2] = sd(auc.k)
  legend("bottomright", legend=c("fold 1", "fold 2", "fold 3", "fold 4", "fold 5", "fold 6", "fold 7", "fold 8", "fold 9", "fold 10"), col=colors, lwd=1, cex = 0.5)
}
auc.list = as.data.frame(auc.list, row.names = models, col.names = c("mean", "sd"))

## Nested Cross Validation on SVM and Random Forest
K = c(1,5,10,50,100)

# SVM
library(kernlab)
auc.svm = NULL
for (m in 1:10) {
  testData = winequality.red[folds[[m]], ]
  error.k = NULL
  for (k in K) {
    error.inner = NULL
    for (n in c(1:10)[-m]) {
      train.index = unlist(folds[c(1:10)[-c(m, n)]])
      trainData = winequality.red[train.index, ]
      validationData = winequality.red[folds[[n]], ]
      model.fit = ksvm(Class ~ ., data=trainData, type="C-svc", kernel='vanilladot', C=k, scaled=c())
      pred.model.fit = as.vector(predict(model.fit, validationData, type = "response"))
      error.inner = c(error.inner, sum(pred.model.fit != validationData$Class)/nrow(validationData))
    }
    error.k = c(error.k, mean(error.inner))
  }
  # Choose the K that minimizes the misclassification error in the inner cross validation
  # and use it to fit in the test set
  K.chosen = K[which(error.k == min(error.k))]
  trainData = winequality.red[-folds[[m]], ]
  model.fit = ksvm(Class ~ ., data=trainData, type="C-svc", kernel='vanilladot', C=K.chosen, scaled=c(), prob.model = T)
  pred.model.fit = as.vector(predict(model.fit, testData, type = "prob")[, "X1"])
  roc.model.fit = pROC::roc(testData$Class, pred.model.fit)
  auc.model.fit = pROC::auc(roc.model.fit)
  if (m == 1) {
    plot(roc.model.fit, print.auc = T, print.auc.x=0.7, print.auc.y=0.5, print.auc.col=colors[m], type="l", main = "SVM ROC Curve", col=colors[m], lwd=1, lty=1)
  } else {
    plot(roc.model.fit, print.auc = T, print.auc.x=0.7, print.auc.y=0.55-0.05*m, print.auc.col=colors[m], type="l", add=TRUE, col=colors[m], lwd=1, lty=1)
  }
  auc.svm = c(auc.svm, auc.model.fit)
}
legend("bottomright", legend=c("fold 1", "fold 2", "fold 3", "fold 4", "fold 5", "fold 6", "fold 7", "fold 8", "fold 9", "fold 10"), col=colors, lwd=1, cex = 0.5)
mean(auc.svm)
sd(auc.svm)

# Random Forest
library(randomForest)
auc.rf = NULL
for (m in 1:10) {
  testData = winequality.red[folds[[m]], ]
  error.k = NULL
  for (k in K) {
    error.inner = NULL
    for (n in c(1:10)[-m]) {
      train.index = unlist(folds[c(1:10)[-c(m, n)]])
      trainData = winequality.red[train.index, ]
      validationData = winequality.red[folds[[n]], ]
      model.fit = randomForest(Class ~ ., data=trainData, ntree = k)
      pred.model.fit = as.vector(predict(model.fit, validationData, type = "response"))
      error.inner = c(error.inner, sum(pred.model.fit != validationData$Class)/nrow(validationData))
    }
    error.k = c(error.k, mean(error.inner))
  }
  # Choose the K that minimizes the misclassification error in the inner cross validation
  # and use it to fit in the test set
  K.chosen = K[which(error.k == min(error.k))]
  trainData = winequality.red[-folds[[m]], ]
  model.fit = randomForest(Class ~ ., data=trainData, ntree = K.chosen)
  pred.model.fit = as.vector(predict(model.fit, testData, type = "prob")[, "X1"])
  roc.model.fit = pROC::roc(testData$Class, pred.model.fit)
  auc.model.fit = pROC::auc(roc.model.fit)
  if (m == 1) {
    plot(roc.model.fit, print.auc = T, print.auc.x=0.7, print.auc.y=0.5, print.auc.col=colors[m], type="l", main = "Random Forest ROC Curve", col=colors[m], lwd=1, lty=1)
  } else {
    plot(roc.model.fit, print.auc = T, print.auc.x=0.7, print.auc.y=0.55-0.05*m, print.auc.col=colors[m], type="l", add=TRUE, col=colors[m], lwd=1, lty=1)
  }
  auc.rf = c(auc.rf, auc.model.fit)
}
legend("bottomright", legend=c("fold 1", "fold 2", "fold 3", "fold 4", "fold 5", "fold 6", "fold 7", "fold 8", "fold 9", "fold 10"), col=colors, lwd=1, cex = 0.5)
mean(auc.rf)
sd(auc.rf)



