## Project Description:
The project involves performing sentiment analysis on Twitter data to classify tweets into three categories: negative, neutral, or positive. By utilizing natural language processing (NLP) techniques, the goal is to build a machine learning model that can accurately predict the sentiment of text data. The model employs a Naive Bayes classifier, a popular choice for text classification tasks due to its simplicity and effectiveness.

## Project Objective:
The objective of this project is to:
Analyze the sentiment of tweets to classify them as negative, neutral, or positive.
Build a robust machine learning model using the Naive Bayes classifier.
Evaluate the model’s performance using accuracy, classification report, and confusion matrix.
Provide visual insights into the sentiment distribution and model performance.
## Dataset Used:

- <a href="https://github.com/Paschal-lee/Wine-Quality-Prediction/blob/main/WineQT.csv">Dataset</a>

## Key Questions:
How well does the Naive Bayes classifier perform for sentiment analysis of Twitter data?
What is the distribution of sentiments in the dataset (positive, neutral, negative)?
Which features (words) are most indicative of sentiment in the tweets?
How can the model be improved to handle imbalances in sentiment categories?

## Process:
Data Loading and Cleaning:
The dataset is loaded and cleaned by dropping rows with missing values.
Columns are renamed for clarity (Clean Text and Category).

Exploratory Data Analysis (EDA):
Exploratory visualizations like pairplots help understand the relationships between variables.
A check for missing values ensures no data gaps.

Data Preprocessing:
The text data is transformed into a format suitable for machine learning using CountVectorizer, which converts the text into a matrix of token counts.

Model Training:
The dataset is split into training and testing sets (80-20 split).
A Naive Bayes classifier is initialized and trained on the training data.

Model Evaluation:
Predictions are made on the test set, and the model’s performance is evaluated using accuracy score, classification report, and confusion matrix.
Visual representations, such as confusion matrices, provide further insights into model performance.

## Insights:
Accuracy: The model achieved an accuracy of 74.7%, indicating that it performs reasonably well, but there is room for improvement.
Class Imbalance: The model struggles slightly with negative sentiment (lower recall for negative sentiment), which could be due to imbalances in the dataset.
Feature Importance: The words in the "Clean Text" column that are highly indicative of positive, neutral, or negative sentiments could be extracted further using techniques like word clouds or feature importance analysis.

## Conclusion:
The sentiment analysis model developed using the Naive Bayes classifier is a strong first step for classifying Twitter sentiments. Despite its good overall performance, the model could be further enhanced by addressing data imbalances and experimenting with other advanced text preprocessing or feature engineering techniques, such as using TF-IDF instead of simple counts, or exploring deep learning models for better accuracy. Visualization tools like confusion matrices also provide valuable insights into where the model's predictions are going wrong, helping to target specific areas for improvement.
