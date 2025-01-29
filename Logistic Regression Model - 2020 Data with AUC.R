### --- 2020 Dataset Code --- ###

# Clear environment
rm(list = ls())


# Load necessary libraries
install.packages("caTools")
install.packages("pROC")
install.packages("caret")
install.packages("car")
install.packages("AICcmodavg") 

library(caTools)
library(pROC)
library(caret)
library(readr)
library(car)
library(tidyr)
library(AICcmodavg) 


# Set seed for reproducibility
set.seed(123)



### --- Load and Preprocess Data --- ###

# Load the data
healthdata_2020 <- read_csv("data path name")

# Encode Variables
factor_cols <- c("HeartDisease", "Smoking", "AlcoholDrinking", "Stroke", 
                 "DiffWalking", "Sex", "AgeCategory", "Race", "Diabetic", 
                 "PhysicalActivity", "GenHealth", "Asthma", "KidneyDisease", "SkinCancer")
healthdata_2020[factor_cols] <- lapply(healthdata_2020[factor_cols], as.factor)

# Handle missing values
healthdata_2020 <- na.omit(healthdata_2020)





### --- Define cross-validation settings --- ###

cv_control_2020 <- trainControl(method = "cv", # Cross-validation
                           number = 5,   # Number of folds
                           classProbs = TRUE, # For AUC
                           summaryFunction = twoClassSummary) # Use AUC for evaluation





### --- Full Model w/ Cross-Validation --- ###

# Train logistic regression with cross-validation
cv_model_2020_full <- train(HeartDisease ~ ., 
                  data = healthdata_2020, 
                  method = "glm", 
                  family = "binomial", 
                  trControl = cv_control_2020, 
                  metric = "ROC") # Optimize based on AUC

# Print results
print(cv_model_2020_full)

# Evaluate on full dataset
pred_probs_2020_full <- predict(cv_model_2020_full, healthdata_2020, type = "prob")[, "Yes"]
predictions_2020_full <- predict(cv_model_2020_full, healthdata_2020)

# Confusion Matrix
conf_matrix_2020_full <- confusionMatrix(predictions_2020_full, healthdata_2020$HeartDisease)
print(conf_matrix_2020_full)

# ROC Curve and AUC
roc_curve_2020_full <- roc(as.numeric(healthdata_2020$HeartDisease == "Yes"), pred_probs_2020_full)
print(auc(roc_curve_2020_full))

# AIC Value from Final Model
aic_value_2020_full <- AIC(cv_model_2020_full$finalModel)
print(paste("AIC from the Full 2020 cross-validation model:", aic_value_2020_full))






### --- Model C w/ Cross Validation --- ###

# Train logistic regression with cross-validation
cv_model_2020_C <- train(HeartDisease ~ . + PhysicalHealth:DiffWalking + Smoking:Asthma + AlcoholDrinking:MentalHealth + PhysicalActivity:SleepTime, 
                            data = healthdata_2020, 
                            method = "glm", 
                            family = "binomial", 
                            trControl = cv_control_2020, 
                            metric = "ROC") # Optimize based on AUC

# Print results
print(cv_model_2020_C)

# Evaluate on full dataset
pred_probs_2020_C <- predict(cv_model_2020_C, healthdata_2020, type = "prob")[, "Yes"]
predictions_2020_C <- predict(cv_model_2020_C, healthdata_2020)

# Confusion Matrix
conf_matrix_2020_C <- confusionMatrix(predictions_2020_C, healthdata_2020$HeartDisease)
print(conf_matrix_2020_C)

# ROC Curve and AUC
roc_curve_2020_C <- roc(as.numeric(healthdata_2020$HeartDisease == "Yes"), pred_probs_2020_C)
print(auc(roc_curve_2020_C))

# AIC Value from Final Model
aic_value_2020_C <- AIC(cv_model_2020_C$finalModel)
print(paste("AIC from Model C 2020 cross-validation model:", aic_value_2020_C))








### --- Extracting Most Influential and Significant Predictors --- ###

# Extract coefficients from the model summary
coefficients_2020 <- summary(model_2020)$coefficients

# Convert to a data frame
coeff_df_2020 <- as.data.frame(coefficients_2020)
colnames(coeff_df_2020) <- c("Estimate", "StdError", "zValue", "pValue")

# Add absolute z-values for ranking
coeff_df_2020$absZ <- abs(coeff_df_2020$zValue)

# Filter significant variables (p-value < 0.05)
significant_coeffs_2020 <- coeff_df_2020[coeff_df_2020$pValue < 0.05, ]

# Rank by absolute z-value and select top 15 most significant predictors
ranked_coeffs_2020 <- significant_coeffs_2020[order(-significant_coeffs_2020$absZ), ]
top_30_2020 <- head(ranked_coeffs_2020, 30)

# Rank top predictors by highest Estimate value (descending)
top_30_estimate_2020 <- top_30_2020[order(-top_30_2020$Estimate), ]

# View the results
print(top_30_estimate_2020)

# Set working directory
setwd("working directory path name")

# Export the top 30 predictors to CSV
write.csv(top_30_estimate_2020, "top_30_2020.csv", row.names = TRUE)


### --- Calculate VIF --- ###

vif_values_2020 <- vif(model_2020)

# Print VIF values
print(vif_values_2020)

# Export VIF as CSV
write.csv(vif_values_2020, "vif_values_2020.csv", row.names = TRUE)










# 
# 
# 
# 
# 
# ### --- Split Data into Training and Testing Sets --- ###
# 
# split_2020 <- sample.split(healthdata_2020$HeartDisease, SplitRatio = 0.7)
# train_data_2020 <- subset(healthdata_2020, split_2020 == TRUE)
# test_data_2020 <- subset(healthdata_2020, split_2020 == FALSE)
# 
# # Ensure factor levels match between train and test sets
# for (col in factor_cols) {
#   test_data_2020[[col]] <- factor(test_data_2020[[col]], levels = levels(train_data_2020[[col]]))
# }
# 
# 
# ### --- Fit and Evaluate Logistic Regression Model --- ###
# 
# # Fit the logistic regression model
# model_2020 <- glm(HeartDisease ~ ., data = train_data_2020, family = binomial(link = "logit"))
# summary(model_2020)
# 
# # Predict probabilities on the test set
# pred_probs_2020 <- predict(model_2020, test_data_2020, type = "response")
# 
# # Convert probabilities to binary outcomes
# predictions_2020 <- ifelse(pred_probs_2020 > 0.5, "Yes", "No")
# predictions_2020 <- factor(predictions_2020, levels = c("No", "Yes"))
# 
# # Evaluate the model: Create Confusion Matrix
# conf_matrix_2020 <- confusionMatrix(predictions_2020, test_data_2020$HeartDisease)
# print(conf_matrix_2020)
# 
# # Create ROC curve and calculate AUC
# roc_curve_2020 <- roc(as.numeric(test_data_2020$HeartDisease == "Yes"), pred_probs_2020)
# print(auc(roc_curve_2020))
# 
# 
# 
# 
# 
# 
# ### --- Separate Subset Model Creation --- ###
# 
# 
# ### --- Model C With Interaction Variables --- ###
# 
# # Fit the logistic regression model
# modelC_2020 <- glm(HeartDisease ~ . + PhysicalHealth:DiffWalking + Smoking:Asthma + AlcoholDrinking:MentalHealth + PhysicalActivity:SleepTime, data = train_data_2020, family = binomial(link = "logit"))
# summary(modelC_2020)
# 
# # Predict probabilities on the test set
# pred_probs_C_2020 <- predict(modelC_2020, test_data_2020, type = "response")
# 
# # Convert probabilities to binary outcomes
# predictions_C_2020 <- ifelse(pred_probs_C_2020 > 0.5, "Yes", "No")
# predictions_C_2020 <- factor(predictions_C_2020, levels = c("No", "Yes"))
# 
# # Evaluate the model: Create Confusion Matrix
# conf_matrix_C_2020 <- confusionMatrix(predictions_C_2020, test_data_2020$HeartDisease)
# print(conf_matrix_C_2020)
# 
# # Create ROC curve and calculate AUC
# roc_curve_C_2020 <- roc(as.numeric(test_data_2020$HeartDisease == "Yes"), pred_probs_C_2020)
# print(auc(roc_curve_C_2020))
# 
# 
# 




# ### --- Model A: Using 'Standard' Predictors --- ###
# 
# # Fit logistic regression model for Model A
# modelA_2020 <- glm(HeartDisease ~ AgeCategory + GenHealth + Smoking + Diabetic + BMI, data = train_data_2020, family = binomial(link = "logit"))
# 
# # Print summary of Model A
# print("Model A Summary:")
# print(summary(modelA_2020))
# 
# # Predict probabilities on the test set for Model A
# pred_probs_A_2020 <- predict(modelA_2020, test_data_2020, type = "response")
# 
# # Convert probabilities to binary predictions for Model A
# predictions_A_2020 <- ifelse(pred_probs_A_2020 > 0.5, "Yes", "No")
# predictions_A_2020 <- factor(predictions_A_2020, levels = levels(test_data_2020$HeartDisease))
# 
# # Create Confusion Matrix for Model A
# conf_matrix_A_2020 <- confusionMatrix(predictions_A_2020, test_data_2020$HeartDisease)
# print("Model A Confusion Matrix:")
# print(conf_matrix_A_2020)
# 
# # Create ROC curve and calculate AUC for Model A
# roc_curve_A_2020 <- roc(test_data_2020$HeartDisease, pred_probs_A_2020)
# auc_value_A_2020 <- auc(roc_curve_A_2020)
# 
# # Print AUC and ROC curve for Model A
# print(paste("Model A AUC:", auc_value_A_2020))
# plot(roc_curve_A_2020, col = "blue", main = "Model A ROC Curve")




# ### --- Model B: Using 'Non-Standard' Predictors --- ###
# 
# # Fit logistic regression model for Model B
# modelB_2020 <- glm(HeartDisease ~ Stroke + KidneyDisease + Asthma + DiffWalking + SleepTime, data = train_data_2020, family = binomial(link = "logit"))
# 
# # Print summary of Model B
# print("Model B Summary:")
# print(summary(modelB_2020))
# 
# # Predict probabilities on the test set for Model B
# pred_probs_B_2020 <- predict(modelB_2020, test_data_2020, type = "response")
# 
# # Convert probabilities to binary predictions for Model B
# predictions_B_2020 <- ifelse(pred_probs_B_2020 > 0.5, "Yes", "No")
# predictions_B_2020 <- factor(predictions_B_2020, levels = levels(test_data_2020$HeartDisease))
# 
# # Create Confusion Matrix for Model B
# conf_matrix_B_2020 <- confusionMatrix(predictions_B_2020, test_data_2020$HeartDisease)
# print("Model B Confusion Matrix:")
# print(conf_matrix_B_2020)
# 
# # Create ROC curve and calculate AUC for Model B
# roc_curve_B_2020 <- roc(test_data_2020$HeartDisease, pred_probs_B_2020)
# auc_value_B_2020 <- auc(roc_curve_B_2020)
# 
# # Print AUC and ROC curve for Model B
# print(paste("Model B AUC:", auc_value_B_2020))
# plot(roc_curve_B_2020, col = "red", main = "Model B ROC Curve")




# 
# ### --- Compare Models --- ###
# models_2020 <- list(cv_model_2020_full, cv_model_2020_C) 
# models_2020.names <- c('cv_model_2020_full', 'cv_model_2020_C') 
# aictab(cand.set = models_2020, modnames = models_2020.names)
