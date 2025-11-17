# Fire prediction App (Custom Logistic Regression + Streamlit + Docker)

This project demonstrates a fully modular Logistic Regression implementation using Sklearn applied to the Fire prediction dataset from Algerian_forest_fires_cleaned_dataset.csv file
It includes:

- A complete machine learning pipeline
- A Streamlit web UI for training and visualization
- A fully modular ML codebase
- Optional Docker container for deployment

---

## Features

### Machine Learning
- Explanatory Data Analysis for the forest fire dataset
- Custom Logistic Regression sklearn model
- Optional L2 (Ridge) regularization
- Automatic feature normalization
- Training metrics and RMSE visualization

### Streamlit App
- Visualize predictions vs actual
- ROC-AUC curve
- Confusion Matrix
- Precision and Recall 
- Display learned weights
- Interactive model parameters
- Works locally or in Docker

## Project Structure
```bash
Logistic_Regression/
│
├── app.py # Streamlit UI
│
├── Logistic_Regression/ # Modular ML package
│ ├── init.py
│ ├── data_utils.py
│ ├── helpers.py
│ ├── optimizers.py
│ ├── model.py
│
├── requirements.txt # Python dependencies
└── Dockerfile # Docker container
└── run.sh # Optional to run the script
```

### 🧠 How the App Works
```bash
1. Loads the wine dataset from sklearn
2. Lets you choose:
  - regularization
  - Max number of iterations
3. Trains the custom regression model
4. Displays:
  - learned weights
  - RMSE
  - actual vs predicted values
  - ROC-AUC curve (RMSE over time)
  - Confusion Matrix
  - Precision and Recall
```
## Installation

## 🛠 Installation (without Docker)

### 1. Clone the repo
```bash
git clone https://github.com/AmreetNanda/Logistic_Regression.git
cd Logistic_Regression
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Streamlit
```bash
streamlit run app.py
```
Open in your browser:
👉 http://localhost:8501

## 🐳 Running with Docker (optional)
### Build the image
```bash
docker build -t logistic-regression .
```

### Run the container
```bash
docker run -p 8501:8501 logistic-regression
```
Open: 👉 http://localhost:8501
## Screenshots

![App Screenshot](https://github.com/AmreetNanda/Logistic_Regression/blob/main/Logistic_Regression.png)


## License

[MIT](https://choosealicense.com/licenses/mit/)

