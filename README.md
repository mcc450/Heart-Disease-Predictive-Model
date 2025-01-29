# Heart-Disease-Predictive-Model

Heart Disease Predictive Model - R
This project was completed as a final assignment in my Masters of Analytics coursework. This analysis is titled 'Predicting Risk of Heart Disease Using Non-Typical Risk Factors'. This project was completed with 2 additional teammates. I was responsible for developing the scope and direction of the project, as well as coding, analysis, and implementation plan.

As part of this project, you will find the source code I developed for the analysis along with the PDF presentation used to present the analysis findings.

Project Introduction
Heart disease has been the leading cause of death for individuals over 20 years old since 1921, according to the American Heart Association. Despite ongoing research, rates remain high. The American Heart Association’s Life’s Essential Eight outlines key factors to reduce heart disease risk, but there may be lesser-known risk factors with similar predictive power. Identifying these factors could offer deeper insights into individual health and improve early intervention efforts. As heart disease progresses, treatment becomes more expensive and difficult. A well-rounded predictive model can help detect risks early, allowing individuals to take preventive action before it is too late.

Project Methodology & Skills Used
Data Preprocessing: Handling missing values, encoding categorical variables, and removing irrelevant features
Feature Engineering: Incorporating interaction terms to capture complex relationships
Predictive Modeling: Training logistic regression models with 5-fold cross-validation
Model Evaluation: Using AUC-ROC curves, confusion matrices, and AIC values to assess performance
Multicollinearity Analysis: Calculating Variance Inflation Factors (VIF) to ensure model stability
Variable Selection: Ranking predictors based on significance (p-values, z-scores) to identify key risk factors
Results & Reporting: Exporting significant findings for further analysis and visualization
Key Findings
Best Model - "The "Full" logistic regression model, using all variables, performs best with 5-fold cross-validation


Minimal Gains: No significant performance improvement from splitting variables or using "Non-Standard" predictors
Feature Selection Impact: Interaction variables and feature exclusions did not improve model performance
Comprehensive Prediction: Using all variables ensures the most comprehensive and simple risk prediction
