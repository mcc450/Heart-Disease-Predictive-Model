# Heart Disease Predictive Model - R
<p>This project was completed as a final assignment in my Masters of Analytics coursework. This analysis is titled 'Predicting Risk of Heart Disease Using Non-Typical Risk Factors'. This project was completed with 2 additional teammates. I was responsible for developing the scope and direction of the project, as well as coding, analysis, and implementation plan.</p>

<p>As part of this project, you will find the source code I developed for the analysis along with the PDF presentation used to present the analysis findings.</p>

## Project Introduction
<p>Heart disease has been the leading cause of death for individuals over 20 years old since 1921, according to the American Heart Association. Despite ongoing research, rates remain high. The American Heart Association’s Life’s Essential Eight outlines key factors to reduce heart disease risk, but there may be lesser-known risk factors with similar predictive power. Identifying these factors could offer deeper insights into individual health and improve early intervention efforts. As heart disease progresses, treatment becomes more expensive and difficult. A well-rounded predictive model can help detect risks early, allowing individuals to take preventive action before it is too late.</p>


## Project Methodology & Skills Used

<ul>
<li> <b>Data Preprocessing:</b> Handling missing values, encoding categorical variables, and removing irrelevant features </li>
<li> <b>Feature Engineering:</b> Incorporating interaction terms to capture complex relationships </li>
<li> <b>Predictive Modeling:</b> Training logistic regression models with 5-fold cross-validation </li>
<li> <b>Model Evaluation:</b> Using AUC-ROC curves, confusion matrices, and AIC values to assess performance </li>
<li> <b>Multicollinearity Analysis:</b> Calculating Variance Inflation Factors (VIF) to ensure model stability </li>
<li> <b>Variable Selection:</b> Ranking predictors based on significance (p-values, z-scores) to identify key risk factors </li>
<li> <b>Results & Reporting:</b> Exporting significant findings for further analysis and visualization </li>
</ul>


## Key Findings

<ul>
<li>Best Model - "The "Full" logistic regression model, using all variables, performs best with 5-fold cross-validation</li>
<li><b>Minimal Gains:</b> No significant performance improvement from splitting variables or using "Non-Standard" predictors</li>
<li><b>Feature Selection Impact:</b> Interaction variables and feature exclusions did not improve model performance</li>
<li><b>Comprehensive Prediction:</b> Using all variables ensures the most comprehensive and simple risk prediction</li>
</ul>
