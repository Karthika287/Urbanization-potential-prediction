# 🏙️ Indian Rural Areas Urbanization Potential Predictor

An intelligent, data-driven web application designed to predict the urbanization potential of rural areas and identify emerging census towns across Indian states.

Instead of relying on rigid, human-assigned mathematical weights, this system utilizes an advanced Machine Learning engine to automatically discover infrastructure correlation patterns directly from regional data.

---

## 🚀 Key Features

* **📊 Regional Insight Dashboard:** Filter data by state to evaluate village ranks, average development scores, and spot high-potential growth clusters using interactive visual scatter matrices.
* **🎛️ AI Policy Simulator:** Adjust infrastructure vectors (like road density, internet penetration, and school counts) via real-time sliders to forecast how policy changes impact a village's urban classification tier.
* **🧠 Explainable AI (XAI) Panel:** Transparently review the Relative Importance Scores calculated by the Machine Learning model, showing exactly which assets drive regional transformation.

---

## 🛠️ The Tech Stack

* **Core Language:** Python
* **User Interface:** Streamlit (Dynamic Web Framework)
* **Data Processing:** Pandas & OpenPyXL
* **Interactive Visualization:** Plotly Express
* **Machine Learning Engine:** Scikit-Learn (Random Forest Regressor & MinMaxScaler)

---

## 💻 How to Run This Project Locally

### 1. Clone or Download the Repository
Download the code files (`app.py`) into a dedicated project folder on your machine.

### 2. Install Dependencies
Open your terminal/command prompt inside your project folder and install the required Python libraries:
```bash
pip install streamlit pandas plotly openpyxl scikit-learn
