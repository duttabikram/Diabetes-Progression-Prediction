# Diabetes Progression Prediction

A Machine Learning project that predicts **diabetes disease progression** using a regression-based approach.

The project uses a **Decision Tree Regressor** to learn relationships between patient-related features and a quantitative diabetes progression measure. A trained model is saved using Pickle and integrated into a **Streamlit application** for interactive predictions.

---

## 📌 Overview

Diabetes progression can be influenced by multiple patient-related factors. Machine Learning can be used to identify patterns in these factors and estimate the progression of the disease.

This project implements an end-to-end Machine Learning workflow:

- Data preprocessing
- Exploratory data analysis
- Feature analysis
- Regression model training
- Model evaluation
- Model serialization
- Interactive prediction using Streamlit

---

## 🎯 Objectives

- Analyze factors associated with diabetes progression.
- Build a regression model for predicting disease progression.
- Train a Decision Tree Regressor using the available dataset.
- Evaluate the trained model using regression metrics.
- Deploy the trained model through an interactive Streamlit application.

---

## 🧠 Machine Learning Approach

The project uses a **Decision Tree Regressor**.

A Decision Tree Regressor works by recursively splitting the dataset based on feature values and creating regions where the target values are relatively similar.

For example:

```text
Patient Data
     │
     ▼
Feature Analysis
     │
     ▼
Decision Tree
     │
 ┌───┴────┐
 │        │
Condition Condition
 │        │
 ▼        ▼
Prediction Prediction
     │
     ▼
Diabetes Progression
```

The trained model is saved as `model.pkl` and loaded by the application for making predictions.

---

## 🔄 Machine Learning Workflow

```text
Dataset
   │
   ▼
Data Loading
   │
   ▼
Data Preprocessing
   │
   ▼
Exploratory Data Analysis
   │
   ▼
Feature Selection
   │
   ▼
Train / Test Split
   │
   ▼
Decision Tree Regressor
   │
   ▼
Model Evaluation
   │
   ▼
Save Trained Model
   │
   ▼
Streamlit Application
   │
   ▼
Diabetes Progression Prediction
```

---

## 🛠️ Tech Stack

### Programming Language

- Python

### Machine Learning

- Scikit-learn
- Decision Tree Regressor

### Data Processing

- Pandas
- NumPy

### Data Visualization

- Matplotlib
- Seaborn

### Application

- Streamlit

### Model Serialization

- Pickle

### Development

- Jupyter Notebook
- Git
- GitHub

---

## 📂 Project Structure

```text
Diabetes-Progression-Prediction/
│
├── Diabetes Prediction DT Regressor.ipynb
│   └── Data analysis and model development
│
├── app.py
│   └── Streamlit prediction application
│
├── model.pkl
│   └── Trained Decision Tree Regression model
│
├── requirements.txt
│   └── Project dependencies
│
├── Screenshot 2025-07-24 013739.png
├── Screenshot 2025-07-24 013802.png
│   └── Application screenshots
│
├── LICENSE
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/duttabikram/Diabetes-Progression-Prediction.git
```

### 2. Navigate to the project

```bash
cd Diabetes-Progression-Prediction
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Run the Streamlit application using:

```bash
streamlit run app.py
```

The application will open in your browser.

You can enter the required patient-related parameters and the application will use the trained Machine Learning model to generate a diabetes progression prediction.

---

## 🖥️ Streamlit Application

The project includes a simple interactive interface built with Streamlit.

The application:

1. Accepts user input.
2. Processes the input in the required format.
3. Loads the trained `model.pkl`.
4. Passes the input to the Decision Tree Regressor.
5. Displays the predicted diabetes progression value.

### Application Screenshot

![Diabetes Prediction Application](Screenshot%202025-07-24%20013739.png)

### Prediction Result

![Prediction Result](Screenshot%202025-07-24%20013802.png)

---

## 📊 Model

### Decision Tree Regressor

The Decision Tree Regressor was selected as the regression model for this project.

**Advantages:**

- Easy to understand and interpret
- Can model nonlinear relationships
- Requires relatively little data preprocessing
- Can capture interactions between features

The trained model is stored in:

```text
model.pkl
```

and loaded by `app.py` during prediction.

---

## 📓 Jupyter Notebook

The complete model development process is available in:

```text
Diabetes Prediction DT Regressor.ipynb
```

The notebook contains the data analysis and Machine Learning experimentation used to develop the prediction model.

---

## 🚀 Future Improvements

Possible improvements include:

- Experimenting with Random Forest Regression
- Gradient Boosting and other ensemble models
- Hyperparameter tuning
- Cross-validation
- Feature importance analysis
- Comparing multiple regression algorithms
- Improving the Streamlit UI
- Deploying the application online
- Adding prediction explanations to help users understand model output

---

## ⚠️ Disclaimer

This project is intended for **educational and demonstration purposes only**.

The predictions generated by this Machine Learning model should **not be considered medical advice, diagnosis, or treatment recommendations**.

---

## 📜 License

This project is licensed under the **GNU General Public License v3.0 (GPL-3.0)**.

---

## 👨‍💻 Author

**Bikram Dutta**

GitHub: [duttabikram](https://github.com/duttabikram)
