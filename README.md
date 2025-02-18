# OpticalDiseases
**Introduction**
With the thorough research on multiple aspects and sources, our group decided to opt for a medical related subject i.e. ocular disease dataset from Kaggle containing patient diagnosis, demographic, and medical image data from different hospitals/medical centres in China (Larxel, 2020). This is to analyze the research questions related to age patterns in disease prevalence, image-based disease prediction models, and factors influencing diagnosis likelihood in the near future. If time permits, a comparison of supervised and unsupervised deep learning will also be performed for image recognition. Through these techniques, our group hopes to uncover insights from this multifaceted medical data.
# Ocular Disease Prediction

## **Project Description**
This project focuses on predicting ocular diseases using machine learning and natural language processing (NLP). By analyzing structured and unstructured medical data, the goal is to provide **early detection** and **personalized treatment recommendations**.

## **Group Members**
- Abdul Moiez
- Fuad Ali Khowaja
- Sanover Tasneem
- Ramish Fatima

## **Dataset Overview**

**Dimension of the Dataset**
The dataset which represents the “real life” patient information contains about 6,400 records of ocular disease data from 5,000 patients, with 15 attributes including patient demographic information, diagnoses, and fundus images of patients' eyes. Although this doesn’t satisfy the minimum requirements of 10,000 records, however, most medical records dataset found online were between the brackets of 3000 to 6000 records only, hence, our group went with the dataset having maximum numbers and approved by the instructor. 

## Rationale for Dataset 
Our group chose this medical dataset because it presents an opportunity to work with image analysis and processing techniques, an area members were intrigued to explore. The images coupled with demographic and diagnosis data and the keywords given by doctors also enables us to investigate research questions around age-related disease trends and factors influencing human vs eye diseases likelihood that can occur due to the current state.

### **Data Includes:**
- **Patient demographics** (age, gender, medical history)
- **Ophthalmic data** (disease type, severity, treatment history)
- **Medical notes** (textual descriptions from patient records)

## **Objectives**
- **Early disease detection** using machine learning models.
- **Patient segmentation** based on disease type and risk factors.
- **Insights extraction** from medical notes through NLP.
- **Personalized treatment recommendations** using clustering techniques.

## **Machine Learning Models Used**
1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Random Forest Classifier** *(Best performing model)*
4. **XGBoost Classifier** *(Advanced feature analysis)*
5. **K-means Clustering** *(For patient stratification)*
6. **NLP-based Text Classification** *(Medical notes analysis)*

## **Data Processing & Feature Engineering**
- **Exploratory Data Analysis (EDA):**
  - Distribution analysis, feature correlations, and missing value handling.
- **Oversampling using SMOTE:**
  - Addressed class imbalance in disease prediction.
- **Feature selection using GridSearchCV:**
  - Optimized model parameters.
- **Disease grouping and classification:**
  - Categorized into **MONR, MINR, DAMD, SNR, and DR**.

## **Findings & Insights**
### **Patient Stratification & Personalized Treatment**
- Identified key patient clusters with **distinct disease patterns**.
- Tailored treatment recommendations for each subgroup.
- Highlighted **age-related** susceptibilities to specific diseases.

### **Text Mining & NLP Analysis**
- Extracted disease patterns using **Latent Dirichlet Allocation (LDA)**.
- **Word cloud visualization** for frequently diagnosed conditions.
- Classified medical notes with **84.97% accuracy**.

## **Deployment & Integration**
- **Flask-based Web Application** integrated with the trained models.
- **MLOps Pipeline** for automated model deployment.


## **Key Takeaways**
- **Diabetes and hypertension significantly impact ocular health.**
- **Early screening and stratification improve treatment planning.**
- **NLP and text mining enable better medical documentation analysis.**
- **Machine learning models, particularly Random Forest, offer strong predictive accuracy.**

## **Future Enhancements**
- **Deep Learning Approaches** for better image-based disease classification.
- **Enhancing NLP models** for improved medical notes categorization.
- **Real-time integration** with hospital databases for live predictions.

 
