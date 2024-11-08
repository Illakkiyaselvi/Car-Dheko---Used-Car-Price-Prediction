<div align="center">

 # Car Dheko - Used Car Price Prediction

</div>
The "Car Dheko - Used Car Price Prediction" project aims to develop an accurate machine learning model that predicts the prices of used cars. The project also involves deploying this model as an interactive web application using Streamlit. This tool will be user-friendly and accessible to both customers and sales representatives, allowing them to input car features and obtain real-time price predictions.

##  Project Scope

The project leverages historical data on used car prices from CarDekho, which includes various features such as make, model, year, fuel type, transmission type, and more. The objective is to create a machine learning model that can accurately predict used car prices based on these features. The model will be integrated into a Streamlit-based web application, which will allow users to input car details and receive price estimates instantly.

##  Problem Statement

Enhancing customer experience and streamlining the pricing process at Car Dheko by leveraging machine learning. The goal is to create a user-friendly tool that predicts the prices of used cars based on various features and deploy it as an interactive web application.

##  Data Source

* We would be working with quite a large data which contains about __8369__ data points where again we would be dividing that into the training set and the test set.
* Having a look at some of the cars that we are always excited to use in our daily lives, it is better to understand how these cars are being sold and their average prices.
* Feel free to take a look at the dataset that was used in the process of predicting the prices of cars. Below is the link.

 __Source:__ https://www.cardekho.com/usedCars
 ##  Approach
 
### 1. Data Processing

* Import and concatenate the datasets for various cities, convert them into a structured format, and concatenate them into a single dataset.

**Data before structuring (unstructured data)**
 


* Address missing data using imputation techniques—mean, median, or mode for numerical columns, and mode or a new category for categorical columns.

* Next is standardizing the Data. Ensure that the data is in the correct format, e.g., removing units from numerical columns and converting them to integers.

* Encoding categorical variables using one-hot encoding for nominal categorical variables and label or ordinal encoding for ordinal categorical variables.

* Normalizing numerical features using techniques like Min-Max Scaling or Standard Scaling.

* Identify and remove or cap outliers using methods such as the IQR (Interquartile Range) method or Z-score analysis.

### 2. Exploratory Data Analysis (EDA)

In this section, the dataset is thoroughly explored to uncover patterns, trends, and insights that can guide further analysis and model development. EDA helps in understanding the data better and in identifying any anomalies, relationships, or key features that may influence the model's performance.

#### 2.1 Statistics Calculated:

Provide a quantitative overview of the dataset by calculating key summary statistics.

**Mean:** The average value for each numerical feature, providing insight into the central tendency.
 
**Median:** The middle value when the data is ordered, useful for understanding the distribution, especially in the presence of outliers.
 
**Mode:** The most frequently occurring value, particularly relevant for categorical variables.
 
**Standard Deviation:** Measures the dispersion of the data points from the mean, indicating how spread out the values are.

#### 2.2 Data Visualization

Visualizing data is crucial for identifying patterns, relationships, and outliers that may not be immediately apparent from summary statistics alone. The following

visualizations were created:

**(i) Scatter Plots :**

Visualize relationships between two numerical variables. For example, a scatter plot of price vs. mileage can reveal how the price decreases as mileage increases.



**(ii) Histograms :**

Show the distribution of a single numerical variable. For instance, a histogram of engine_size can help identify the most common engine sizes in the dataset.


**(iii) Box plot :**

The plot_boxplots function generates box plots for specified columns in a DataFrame to help identify outliers and understand the distribution of the data. Box plots are useful for visualizing data distributions and detecting any anomalies or extreme values.



**(iv) Correlation heatmap :**

The plot_correlation_heatmap_fix function generates a correlation heatmap for numeric features in a DataFrame, providing a visual representation of feature relationships. This helps in identifying correlations and understanding how features relate to each other.



#### 2.3 Insights and Observations

From the summary statistics and visualizations, several interesting patterns and trends were observed:

**Insight 1:** A negative correlation between mileage and price, indicating that cars with higher mileage tend to have lower prices.

**Insight 2:** Certain engine sizes are more common, suggesting they may be more desirable or widely available in the market.

**Insight 3:** Outliers in the price distribution indicate the presence of luxury or high-end vehicles that could skew the analysis.

#### 2.4 Feature Selection

Feature selection is a critical step in the modeling process that involves identifying the most important features that significantly impact car prices. By selecting the most relevant features, the model becomes more efficient, interpretable, and performs better.

**2.4.1 Techniques Used for Feature Selection**

To identify the important features that have a significant impact on car prices, the following techniques were employed:

**(i) Correlation Analysis:**

* Objective: Assess the linear relationship between numerical features and the target variable (car prices).
 
* Method: Calculate the correlation coefficients between the features and the target variable. Features with high positive or negative correlation are considered important.

**(ii) Feature Importance from Models:**

* Objective: Leverage machine learning models to determine which features contribute most to the model’s predictions.
 
* Method: Use algorithms like Random Forest or Gradient Boosting, which can provide a measure of feature importance as part of their training process.

**(iii) Domain Knowledge:**

* Objective: Use industry-specific insights and expert knowledge to identify features that are logically expected to influence car prices.

* Method: Incorporate knowledge of factors like brand reputation, engine specifications, and market trends that are known to affect car valuation.

* Example: High-end brands or cars with advanced safety features are generally expected to have higher prices. These insights guide the inclusion or exclusion of certain features in the final model.



By focusing on these selected features, the model is likely to achieve better performance, with improved accuracy and interpretability, while reducing the risk of overfitting by removing irrelevant or redundant features.

### 3. Model Development

* Divide the dataset into training and testing sets to evaluate the model's performance on unseen data.

* Commonly, the dataset is split into 70% training data and 30% testing data, or 80% training and 20% testing. This ensures that the model is trained on a substantial portion of the data while retaining enough data for a reliable test.

* Choose appropriate machine learning algorithms.

**(i) Algorithms:**

   **- Linear Regression:** Useful for modeling the relationship between the dependent and independent variables when the relationship is linear.
 
   **- Decision Trees:** Helpful for capturing complex, non-linear relationships in the data.
 
   **- Random Forests:** An ensemble method that combines multiple decision trees to improve predictive performance and control overfitting.
 
   **- Gradient Boosting Machines:** Another ensemble method that builds models sequentially to correct errors made by previous models.

* Ensure the model's performance is consistent and not dependent on a particular training subset.Use cross-validation techniques such as k-fold cross-validation to evaluate the model on different subsets of the data.

* Enhance the model's performance by tuning its hyperparameters.

 **(ii) Methods:**
 
  **- Grid Search :** Exhaustively searches through a specified parameter grid to find the best combination.
   
  **- Random Search :** Randomly samples parameters from a distribution to find a good combination more efficiently

### 4. Model Evaluation

**4.1 Performance Metrics:**

To assess how well your machine learning models are performing, you'll use several key metrics:

**(i) Mean Absolute Error (MAE):**

  Measures the average magnitude of errors in a set of predictions, without considering their direction. It’s the average over the test sample of the absolute differences between prediction and actual observation where all individual differences have equal weight.

**(ii) Mean Squared Error (MSE):**

  Measures the average of the squares of the errors—that is, the average squared difference between the estimated values and the actual value. This metric penalizes larger errors more than MAE.

**(iii) R-squared (R²):**

  Represents the proportion of the variance in the dependent variable that is predictable from the independent variables. It ranges from 0 to 1, where a higher value indicates a better fit.

**(iv) Root Mean Squared Error (MSE):**

  Root Mean Squared Error (RMSE) is a commonly used metric to evaluate the accuracy of a regression model. It measures the square root of the average of the squared differences between the predicted values and the actual values. RMSE gives more weight to larger errors because the errors are squared before averaging, making it sensitive to outliers.

#### 4.2 Model Comparison

To choose the best model, you should compare the models you have trained based on the performance metrics mentioned above. Here’s how you can approach this:

* Calculate Metrics: Compute MAE, MSE, and R² for each model you have trained.
 
* Compare Scores: Evaluate which model has the lowest MAE and MSE and the highest R².
 
* Select the Best Model: Choose the model that best balances these metrics according to your specific needs and objectives.


* Random Forest has the lowest MSE, the lowest RMSE , and the highest R² .This suggests that it has the best overall performance, providing the most accurate predictions and the best fit to the data.
 
### 5. Optimization

#### 5.1 Feature Engineering

Feature engineering involves creating new features or modifying existing ones to enhance the model’s performance. This process often involves:

  **(i) Creating New Features:**
 
  * Derive new features from existing data that might better capture the underlying patterns.

  **(ii) Modifying Existing Features:**

   * Transform existing features to improve their contribution to the model. Like Normalizing or scaling features to bring them onto a similar scale,Encoding categorical variables appropriately, and Handling missing values through imputation or feature engineering.

 #### 5.2 Regularization

Regularization techniques are used to prevent overfitting by adding a penalty to the model's complexity. Key techniques include:

  **5.2.1 Lasso Regression (L1 Regularization):**

  Adds a penalty equal to the absolute value of the magnitude of coefficients. This can lead to sparse models where some feature weights are exactly zero, effectively performing feature selection.  

  **5.2.2 Ridge Regression (L2 Regularization):**

  Adds a penalty equal to the square of the magnitude of coefficients. This helps in reducing the complexity of the model and can handle multicollinearity.

  **5.2.3 Choosing the Right Regularization Technique:**

  Depending on the problem and the dataset, you might choose Lasso if feature selection is important, or Ridge if you need to handle multicollinearity and reduce overfitting.

By applying these optimization techniques, you can improve the performance and generalization of your model, leading to better predictive accuracy and robustness.


### 6. Streamlit Application:

* This section describes how the final model was deployed as an interactive web application using Streamlit. The application allows users to input various car features and receive real-time predictions for the car's price.
 
#### 6.1 Streamlit Setup

* Installation: Ensure Streamlit is installed in your environment.

```
 pip install streamlit

```
* Application Structure: The Streamlit app is structured to have an intuitive user interface where users can input car details such as make, model, year, mileage, engine size, and more.

#### 6.2 Building the User Interface

* Input Fields: Streamlit provides various input widgets like sliders, text boxes, and dropdowns. These allow users to specify the car features they want to input for the price prediction.

* Prediction Button: A button that, when clicked, triggers the model to make a prediction based on the input features

#### 6.3 Integrating the Model

* The trained machine learning model is loaded and used to make predictions based on the user inputs.

* The model processes the input data and returns a price prediction, which is then displayed in the app.

#### 6.4 Running the Application

* Running Locally: You can run the application locally using the following command:

```
streamlit run app.py
```

* Deploying the Application: You can deploy the Streamlit app using platforms like Streamlit Cloud, Heroku, or any other hosting service that supports Python web applications


##  Conclusion

**1.Accurate Predictive Model for Used Car Prices**

Developed a robust machine learning model capable of predicting used car prices with high accuracy. The model demonstrates strong performance metrics, including low Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE), ensuring reliable price predictions.

**2.In-Depth Data Analysis and Visualizations**

Conducted a thorough exploratory data analysis (EDA) that included generating summary statistics and creating various visualizations such as scatter plots, histograms, box plots, and correlation heatmaps. These analyses provided valuable insights into the dataset, revealing key patterns and relationships that influenced the model’s performance.

**3.Comprehensive Documentation**

Provided detailed documentation that outlines the methodology used in the project, including data preprocessing, feature engineering, model development, and evaluation. The documentation also includes explanations of the different machine learning models tested and their performance results, ensuring clarity and transparency.

**4.Interactive Streamlit Application**

Developed an interactive Streamlit application that allows users to input car features and obtain real-time price predictions. This user-friendly application makes the model accessible to non-technical users and facilitates easy and immediate use of the predictive tool.
