### --- 2022 Dataset Code --- ###

# Clear environment
rm(list = ls())

# Load necessary libraries
library(dplyr)
library(caret)
library(pROC)
library(glmnet)
library(tidyr)
library(car)
library(AICcmodavg) 

# Load the 2022 dataset
healthdata_2022 <- read_csv("data path name")

# Set seed for reproducibility
set.seed(123)


### --- Data Manipulation, Preprocessing --- ###

# Remove rows with NA in HadAngina
healthdata_2022 <- healthdata_2022[!is.na(healthdata_2022$HadAngina), ]

# Remove 'State' from dataset
healthdata_2022 <- healthdata_2022 %>% select(-State)

# Replace NA's with "Missing" for categorical variables
healthdata_2022 <- healthdata_2022 %>%
  mutate(across(c("TetanusLast10Tdap", "PneumoVaxEver", "HIVTesting", "ChestScan", 
                  "CovidPos", "HighRiskLastYear", "FluVaxLast12", "AlcoholDrinkers", 
                  "ECigaretteUsage", "SmokerStatus", "DifficultyErrands", 
                  "DifficultyConcentrating", "DifficultyWalking", 
                  "DifficultyDressingBathing", "BlindOrVisionDifficulty", 
                  "DeafOrHardOfHearing", "RaceEthnicityCategory", "RemovedTeeth", 
                  "PhysicalActivities", "AgeCategory", "LastCheckupTime", 
                  "HadSkinCancer", "HadDepressiveDisorder", "HadHeartAttack", 
                  "HadArthritis", "HadCOPD", "HadKidneyDisease", "HadAsthma", 
                  "GeneralHealth", "HadStroke", "HadDiabetes"), 
                ~ replace_na(., "Missing")))

# Impute missing numeric values with column mean
healthdata_2022 <- healthdata_2022 %>%
  mutate(across(where(is.numeric), ~ ifelse(is.na(.), mean(., na.rm = TRUE), .)))

# Convert categorical variables to factors
healthdata_2022 <- healthdata_2022 %>%
  mutate(across(where(is.character), as.factor))


### --- Data Check --- ###

# Check for remaining NA values
cols_to_check_2022 <- c("TetanusLast10Tdap", "PneumoVaxEver", "HIVTesting", "ChestScan", 
                        "CovidPos", "HighRiskLastYear", "FluVaxLast12", "AlcoholDrinkers", 
                        "ECigaretteUsage", "SmokerStatus", "DifficultyErrands", 
                        "DifficultyConcentrating", "DifficultyWalking", 
                        "DifficultyDressingBathing", "BlindOrVisionDifficulty", 
                        "DeafOrHardOfHearing", "RaceEthnicityCategory", "RemovedTeeth", 
                        "PhysicalActivities", "AgeCategory", "LastCheckupTime", 
                        "HadHeartAttack", "HadSkinCancer", "HadDepressiveDisorder", 
                        "HadArthritis", "HadCOPD", "HadKidneyDisease", "HadAsthma", 
                        "GeneralHealth", "HadStroke", "HadDiabetes")

sapply(healthdata_2022[cols_to_check_2022], function(col) sum(is.na(col)))

# Count of each response type
sapply(healthdata_2022[cols_to_check_2022], function(col) table(col, useNA = "ifany"))



### --- Define cross-validation settings --- ###

cv_control_2022 <- trainControl(method = "cv", # Cross-validation
                           number = 5,   # Number of folds
                           classProbs = TRUE, # For AUC
                           summaryFunction = twoClassSummary) # Use AUC for evaluation




### --- Full Model w/ Cross-Validation --- ###

# Train logistic regression with cross-validation
cv_model_2022_full <- train(HadAngina ~ ., 
                            data = healthdata_2022, 
                            method = "glm", 
                            family = "binomial", 
                            trControl = cv_control_2022, 
                            metric = "ROC") # Optimize based on AUC

# Print results
print(cv_model_2022_full)

# Evaluate on full dataset
pred_probs_2022_full <- predict(cv_model_2022_full, healthdata_2022, type = "prob")[, "Yes"]
predictions_2022_full <- predict(cv_model_2022_full, healthdata_2022)

# Confusion Matrix
conf_matrix_2022_full <- confusionMatrix(predictions_2022_full, healthdata_2022$HadAngina)
print(conf_matrix_2022_full)

# ROC Curve and AUC
roc_curve_2022_full <- roc(as.numeric(healthdata_2022$HadAngina == "Yes"), pred_probs_2022_full)
print(auc(roc_curve_2022_full))

# AIC Value from Final Model
aic_value_2022_full <- AIC(cv_model_2022_full$finalModel)
print(paste("AIC from the Full 2022 cross-validation model:", aic_value_2022_full))





### --- Full Model w/ Cross-Validation --- ###

# Train logistic regression with cross-validation
cv_model_2022_C <- train(HadAngina ~ . + WeightInKilograms:BMI, 
                            data = healthdata_2022, 
                            method = "glm", 
                            family = "binomial", 
                            trControl = cv_control_2022, 
                            metric = "ROC") # Optimize based on AUC

# Print results
print(cv_model_2022_C)

# Evaluate on full dataset
pred_probs_2022_C <- predict(cv_model_2022_C, healthdata_2022, type = "prob")[, "Yes"]
predictions_2022_C <- predict(cv_model_2022_C, healthdata_2022)

# Confusion Matrix
conf_matrix_2022_C <- confusionMatrix(predictions_2022_C, healthdata_2022$HadAngina)
print(conf_matrix_2022_C)

# ROC Curve and AUC
roc_curve_2022_C <- roc(as.numeric(healthdata_2022$HadAngina == "Yes"), pred_probs_2022_C)
print(auc(roc_curve_2022_C))

# AIC Value from Final Model
aic_value_2022_C <- AIC(cv_model_2022_C$finalModel)
print(paste("AIC from Model C 2022 cross-validation model:", aic_value_2022_C))





### --- Determine Most Influential Predictors --- ###

model_summary_2022 <- summary(model_2022)

# Extract coefficients, standard errors, and p-values from the summary
coefficients_data_2022 <- as.data.frame(model_summary_2022$coefficients)
colnames(coefficients_data_2022) <- c("Estimate", "Std.Error", "z.value", "p.value")

# Filter for significant predictors at the 0.001 level
significant_predictors_2022 <- coefficients_data_2022[coefficients_data_2022$p.value < 0.001, ]

# Rank predictors by their coefficients (highest to lowest)
ranked_predictors_2022 <- significant_predictors_2022[order(significant_predictors_2022$Estimate, decreasing = TRUE), ]

# Select the top 30 predictors
top_30_predictors_2022 <- head(ranked_predictors_2022, 30)

# View the results
print(top_30_predictors_2022)

# Set working directory
setwd("working directory pathname")

# Export the top 30 predictors to CSV
write.csv(top_30_predictors_2022, "top_30_predictors_2022.csv", row.names = TRUE)



### --- Calculate VIF --- ###

vif_values_2022 <- vif(model_2022)

# Print VIF values
print(vif_values_2022)

# Export VIF as CSV
write.csv(vif_values_2022, "vif_values_2022.csv", row.names = TRUE)































# ### --- Split Dataset into Training and Testing Sets --- ###
# 
# trainIndex_2022 <- createDataPartition(healthdata_2022$HadAngina, p = 0.7, list = FALSE)
# train_data_2022 <- healthdata_2022[trainIndex_2022, ]
# test_data_2022 <- healthdata_2022[-trainIndex_2022, ]
# 
# 
# ### --- Build Logistic Regression Model --- ###
# 
# model_2022 <- glm(HadAngina ~ ., data = train_data_2022, family = binomial)
# 
# # Summarize the model
# summary(model_2022)





# 
# ### --- Full Model Evaluation --- ###
# 
# # Predict using test set
# test_data_2022$predicted_prob <- predict(model_2022, test_data_2022, type = "response")
# test_data_2022$predicted_class <- ifelse(test_data_2022$predicted_prob > 0.5, "Yes", "No")
# 
# test_data_2022$predicted_class <- factor(test_data_2022$predicted_class, levels = c("No", "Yes"))
# test_data_2022$HadAngina <- factor(test_data_2022$HadAngina, levels = c("No", "Yes"))
# 
# # Confusion Matrix
# conf_matrix_2022 <- confusionMatrix(test_data_2022$predicted_class, test_data_2022$HadAngina)
# print(conf_matrix_2022)
# 
# # Calculate AUC-ROC
# roc_curve_2022 <- roc(as.numeric(test_data_2022$HadAngina), test_data_2022$predicted_prob)
# auc_value_2022 <- auc(roc_curve_2022)
# 
# print(paste("AUC:", auc_value_2022))
# plot(roc_curve_2022, col = "blue", main = "ROC Curve")
# 

# 
# 
# ### --- Model A: Using Standard Predictors --- ###
# 
# # Fit logistic regression model for Model A
# modelA_2022 <- glm(HadAngina ~ AgeCategory + GeneralHealth + HadDiabetes + Sex + RaceEthnicityCategory, data = train_data_2022, family = binomial(link = "logit"))
# 
# # Print summary of Model A
# print("Model A Summary:")
# print(summary(modelA_2022))
# 
# # Predict probabilities on the test set for Model A
# pred_probs_A_2022 <- predict(modelA_2022, test_data_2022, type = "response")
# 
# # Convert probabilities to binary predictions for Model A
# predictions_A_2022 <- ifelse(pred_probs_A_2022 > 0.5, "Yes", "No")
# predictions_A_2022 <- factor(predictions_A_2022, levels = levels(test_data_2022$HadAngina))
# 
# # Confusion Matrix for Model A
# conf_matrix_A_2022 <- confusionMatrix(predictions_A_2022, test_data_2022$HadAngina)
# print("Model A Confusion Matrix:")
# print(conf_matrix_A_2022)
# 
# # Create ROC curve and calculate AUC for Model A
# roc_curve_A_2022 <- roc(as.numeric(test_data_2022$HadAngina), pred_probs_A_2022)
# auc_value_A_2022 <- auc(roc_curve_A_2022)
# 
# print(paste("Model A AUC:", auc_value_A_2022))
# 
# 
# ### --- Model B: Using Non-Standard Predictors --- ###
# 
# # Fit logistic regression model for Model B
# modelB_2022 <- glm(HadAngina ~ HadStroke + RemovedTeeth + PneumoVaxEver + SleepHours + BlindOrVisionDifficulty, data = train_data_2022, family = binomial(link = "logit"))
# 
# # Print summary of Model B
# print("Model B Summary:")
# print(summary(modelB_2022))
# 
# # Predict probabilities on the test set for Model B
# pred_probs_B_2022 <- predict(modelB_2022, test_data_2022, type = "response")
# 
# # Convert probabilities to binary predictions for Model B
# predictions_B_2022 <- ifelse(pred_probs_B_2022 > 0.5, "Yes", "No")
# predictions_B_2022 <- factor(predictions_B_2022, levels = levels(test_data_2022$HadAngina))
# 
# # Confusion Matrix for Model B
# conf_matrix_B_2022 <- confusionMatrix(predictions_B_2022, test_data_2022$HadAngina)
# print("Model B Confusion Matrix:")
# print(conf_matrix_B_2022)
# 
# # Create ROC curve and calculate AUC for Model B
# roc_curve_B_2022 <- roc(as.numeric(test_data_2022$HadAngina), pred_probs_B_2022)
# auc_value_B_2022 <- auc(roc_curve_B_2022)
# 
# print(paste("Model B AUC:", auc_value_B_2022))
# 
# 
# 
# ### --- Model C: Using Interaction Terms --- ###
# 
# # Fit logistic regression model for Model C
# modelC_2022 <- glm(HadAngina ~ . + WeightInKilograms:BMI, 
#                    data = train_data_2022, 
#                    family = binomial(link = "logit"))
# 
# # Print summary of Model C
# print("Model C Summary:")
# print(summary(modelC_2022))
# 
# # Predict probabilities on the test set for Model C
# pred_probs_C_2022 <- predict(modelC_2022, test_data_2022, type = "response")
# 
# # Convert probabilities to binary predictions for Model C
# predictions_C_2022 <- ifelse(pred_probs_C_2022 > 0.5, "Yes", "No")
# predictions_C_2022 <- factor(predictions_C_2022, levels = levels(test_data_2022$HadAngina))
# 
# # Confusion Matrix for Model C
# conf_matrix_C_2022 <- confusionMatrix(predictions_C_2022, test_data_2022$HadAngina)
# print("Model C Confusion Matrix:")
# print(conf_matrix_C_2022)
# 
# # Create ROC curve and calculate AUC for Model C
# roc_curve_C_2022 <- roc(as.numeric(test_data_2022$HadAngina), pred_probs_C_2022)
# auc_value_C_2022 <- auc(roc_curve_C_2022)
# 
# print(paste("Model C AUC:", auc_value_C_2022))
# 
# 
# 
# ### --- Model D: Using Interaction Terms --- ###
# 
# # Fit logistic regression model for Model D
# modelD_2022 <- glm(HadAngina ~ . + WeightInKilograms:BMI + SmokerStatus:ECigaretteUsage + DifficultyWalking:DifficultyDressingBathing:DifficultyErrands, 
#                    data = train_data_2022, 
#                    family = binomial(link = "logit"))
# 
# # Print summary of Model D
# print("Model D Summary:")
# print(summary(modelD_2022))
# 
# # Predict probabilities on the test set for Model D
# pred_probs_D_2022 <- predict(modelD_2022, test_data_2022, type = "response")
# 
# # Convert probabilities to binary predictions for Model D
# predictions_D_2022 <- ifelse(pred_probs_D_2022 > 0.5, "Yes", "No")
# predictions_D_2022 <- factor(predictions_D_2022, levels = levels(test_data_2022$HadAngina))
# 
# # Confusion Matrix for Model D
# conf_matrix_D_2022 <- confusionMatrix(predictions_D_2022, test_data_2022$HadAngina)
# print("Model D Confusion Matrix:")
# print(conf_matrix_D_2022)
# 
# # Create ROC curve and calculate AUC for Model D
# roc_curve_D_2022 <- roc(as.numeric(test_data_2022$HadAngina), pred_probs_D_2022)
# auc_value_D_2022 <- auc(roc_curve_D_2022)
# 
# print(paste("Model D AUC:", auc_value_D_2022))
# 
# 
# 
# 
# 
# 
# ### --- Compare Models --- ###
# models_2022 <- list(model_2022, modelA_2022, modelB_2022, modelC_2022, modelD_2022) 
# models_2022.names <- c('model_2022', 'modelA_2022', 'modelB_2022', 'modelC_2022', 'modelD_2022') 
# aictab(cand.set = models_2022, modnames = models_2022.names)
# 
# 

