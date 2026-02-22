# 🏆 Kaggle Challenge Projects

A collection of machine learning projects solving real-world predictive challenges from Kaggle competitions. This repository showcases advanced data science techniques including exploratory analysis, feature engineering, and model optimization.

## 📂 Projects

### 1. 🏡 Ames House Prices: Advanced Regression Techniques
**Challenge:** Predict the final sale price of homes in Ames, Iowa  
**Model Performance:** MAE ~$14,000 (< 8% error on $180k homes) | 13% Kaggle test error
**Best Algorithm:** XGBoost Regressor

#### Key Highlights:
- **80+ features** analyzed using correlation heatmaps to identify price drivers
- **Data cleaning:** Handled categorical vs. missing data distinctions (e.g., "None" vs. NaN)
- **Feature engineering:** One-hot encoded 81 columns into 200+ features
- **Model comparison:** Random Forest, Linear Regression, and XGBoost tested
- **Insight:** Minor features matter—over-pruning weak columns actually increased error

**Notebooks:** [house_price.ipynb](house_price.ipynb) | [house_price.md](house_price.md)

---

### 2. 🚢 Titanic Survival Prediction
**Challenge:** Predict passenger survival based on demographics and ticket information  
**Model Performance:** 84.5% validation accuracy | 77% Kaggle test accuracy  
**Best Algorithm:** Random Forest Classifier (with hyperparameter tuning)

#### Key Highlights:
- **Strong predictors identified:** Gender and passenger class were dominant survival factors
- **Feature engineering:** Extracted titles (Mr, Mrs, Master) from names to capture historical context
- **Handling missing data:** Median imputation for age, dropped unreliable columns
- **Hyperparameter tuning:** GridSearchCV optimized n_estimators, max_depth, and min_samples_split
- **Model comparison:** Logistic Regression (70%), XGBoost (83%), Random Forest (84.5%)

**Notebooks:** [titanic.ipynb](titanic.ipynb) | [titanic.md](titanic.md)

---

### 3. 🔢 Digit Recognizer: Handwritten Digit Classification
**Challenge:** Classify handwritten digits (0-9) from pixel data  
**Architecture:** Convolutional Neural Network (CNN)
**Model Performance:** 94% Kaggle test accuracy
**Dataset:** MNIST - 42,000 training images (28×28 pixels each)

#### Key Highlights:
- **Data preprocessing:** Normalized pixel values [0,255] → [0,1], reshaped flat vectors to 28×28×1 images
- **CNN Architecture:** 3 convolutional layers with max pooling for feature extraction, dense layers for classification
- **Model layers:** Conv2D (32 filters) → MaxPool → Conv2D (64 filters) → MaxPool → Conv2D (64 filters) → Dense layers
- **Training:** 5 epochs with 80/20 train-validation split, Adam optimizer, sparse categorical crossentropy loss
- **Performance tracking:** Real-time monitoring of training/validation accuracy and loss convergence

**Notebooks:** [digit_recognizer.ipynb](digit_recognizer.ipynb) | [digit_recognizer.md](digit_recognizer.md)

---

## 🛠️ Tech Stack
- **Language:** Python 3
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-Learn, XGBoost
- **Deep Learning:** TensorFlow, Keras
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Jupyter Notebooks


## 📊 Skills Demonstrated

- **Exploratory Data Analysis (EDA):** Correlation analysis, outlier detection, visualization
- **Data Cleaning:** Handling missing values, distinguishing between NaN and categorical "None"
- **Feature Engineering:** One-hot encoding, feature extraction, family size aggregation
- **Model Selection:** Testing multiple algorithms and selecting based on performance
- **Hyperparameter Tuning:** GridSearchCV for optimal model parameters
- **Error Analysis:** Understanding validation vs. test accuracy gaps
- **Regression & Classification:** Both supervised learning paradigms covered
- **Deep Learning & CNNs:** Convolutional neural networks for image classification
- **Data Normalization & Reshaping:** Preparing raw pixel data for neural networks

## 📈 Results Summary

| Project | Task | Best Model | Score |
|---------|------|-----------|-------|
| House Prices | Regression | XGBoost | 13% Error|
| Titanic | Classification | Random Forest | 77% Accuracy |
| Digit Recognizer | Image Classification | CNN (Deep Learning) | 94% Accuracy |

## 📝 Notes

- Each project includes detailed markdown documentation explaining the methodology
- Notebooks are designed for educational purposes with step-by-step explanations
- Model performance reflects real Kaggle competition submissions

## 🚀 Quick Start

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   ```

2. **Install dependencnumpy scikit-learn xgboost matplotlib seaborn tensorflow
   ```bash
   pip install pandas scikit-learn xgboost matplotlib seaborn notebook
   ```

3. **Download datasets from Kaggle:**
   - [House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
   - [Titanic](https://www.kaggle.com/c/titanic)
   - [Digit Recognizer](https://www.kaggle.com/c/digit-recognizer)

4. **Run the notebooks:**
   ```bash
   jupyter notebook
   ```
