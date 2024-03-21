# Airbnb-Price-Analysis-and-Prediction

_**Techniques**_: Regression Models, Data Analysis, Feature Engineering/ Selection, Hyperparameter Tuning, Cross-Validation <br />
_**Tools**_: Python, Plotly <br />

<img width="489" alt="Screen Shot 2024-03-21 at 5 08 47 PM" src="https://github.com/SeungPang11/Airbnb-Price-Analysis-and-Prediction/assets/67944800/489dc539-1c84-43fd-8340-3a2bc77149b4">
<br />


## Datasets 
**[Seattle Airbnb Open Data](https://www.kaggle.com/datasets/airbnb/seattle?select=listings.csv)** <br />
* 1.4 million listing activity data and 96 variables

## Exploratory Data Analysis
* _**Word Frequency of the Dataset**_<br />

**Gossip Cop Dataset** <br />
<img width="542" alt="Screen Shot 2024-01-04 at 6 39 46 PM" src="https://github.com/SeungPang11/Fake-News-Detection-with-Maching-Learning/assets/67944800/b8ec6a70-ad7b-4c35-859e-df88a003bb93"><br />

**Politifact Dataset** <br />
<img width="543" alt="Screen Shot 2024-01-04 at 6 40 08 PM" src="https://github.com/SeungPang11/Fake-News-Detection-with-Maching-Learning/assets/67944800/cfcba6d8-e34e-4454-8eda-e4124ebc0a94"> <br />

* _**Word Count Distribution**_<br />
<img width="400" alt="Screen Shot 2024-01-06 at 12 52 19 AM" src="https://github.com/SeungPang11/Fake-News-Detection-with-Maching-Learning-Updated-Jan-24-/assets/67944800/a6b52cf3-c048-485b-abe5-f74a71488984">

![Fake News Detection EDA Analysis](https://github.com/SeungPang11/Fake-News-Detection-with-Maching-Learning-Updated-Jan-24-/assets/67944800/9db71001-8b29-46ea-ba38-2533ad505011)



## Methods
_**Feature Engineering**_<br />
* **Word Count**: The number of words in news titles. <br>
* **Retweet Count**: The number of times in news has been tweeted/ retweeted. <br>
* **TF-IDF**: Transform text into a representation of numbers while removing stopwords. <br>
* **Sentiment**: Emotional tone of news (Polarity [-1,1], Sensitivity [0,1]). <br>
* **Scaling**: [MaxAbsScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MaxAbsScaler.html) is applied for sparse CSR matrix. 
[Compare Different Scalers](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#plot-all-scaling-max-abs-scaler-section)
<br>

_**Synthetic Minority Over-Sampling Technique (SMOTE)**_<br />
* Oversample minority class to tweak the model <br />
(reduce False Negatives, at the cost of increasing False Positives).<br /> 
The result is generally an increase in recall, at the cost of lower precision.

* Before SMOTE, the model was good at detecting real news (high True Positives),<br /> 
while performing poorly at detecting fake news (low True Negatives).<br /> 

_**Naive Bayes**_<br />
* Based on conditional probability (Bayes Theorem), probability of an event <br />
occurring given another event already happened, and assumes all features equally <br />
affect the outcome.

_**Logistic Regression**_<br />
* Models the probability of a discrete outcome given input variables.

_**Support Vector Machine**_<br />
* Finds a hyperplane in an N-dimensional space (# of features)  <br />
that distinctly classifies the data points.<br /> 
Computationally intensive and works better on small data with large features.

_**XGBoost**_<br />
* Based on the gradient-boosted trees algorithm which predicts <br />
a target variable by combining the estimates of a set of simpler models.

____________________________________________________
## Result - Updated Jan 2024
* Updated text pre-processing to better remove emojis, URLs, and special characters <br>
* Cross Validation to evaluate model performance <br>
* Tested oversampling minority class & undersampling majority class <br>
* **Improved performance**  
<img width="700" alt="Screen Shot 2024-01-06 at 7 09 30 PM" src="https://github.com/SeungPang11/Fake-News-Detection-with-Maching-Learning-Updated-Jan-24-/assets/67944800/0dc09df2-45a4-4429-afbb-6c92c013e94d"><br>
 -Achieved a **significantly improved F1** score from previous performance <br>
 -**Recall is more important** than Precision (Classifying Fake News as Real News is worse)<br>
 -**Best Performance:** Bernoulli Naive Bayes (Higher Recall and F1) <br>
 -**Worst Performance:** SVM (works best with high dimensional small data) <br>
  and computationally intensive
  
