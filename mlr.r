# Train a Multiple Linear Regression model given a dataset (input CSV file)
#
# background on MLR: https://www.investopedia.com/terms/m/mlr.asp
# R example: https://www.tutorialspoint.com/r/r_multiple_regression.htm

# read data in
train <- read.csv("data/moldescs.csv", header = T, sep = ",")
# only keep interesting columns
train <- train[,c("score","MolW","cLogP","RotB")]

# FBR: TODO
# center and scale all dependant variables, store the scaling parameters
# with the model

# train model
model <- lm(score ~ MolW + cLogP + RotB, data = train)

# Show the model.
a <- coef(model)[1]
b <- coef(model)[2]
c <- coef(model)[3]
d <- coef(model)[4]
print(c(a, b, c, d))
