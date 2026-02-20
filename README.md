# ✈️ Advanced Airfare Price Prediction

## 📌 Project Overview

This project predicts flight ticket prices using Machine Learning and an interactive Streamlit web application.

The system estimates airfare based on airline, number of stops, flight duration, route, travel date, class type, and seasonal factors.  
It simulates real-world pricing dynamics such as peak season multipliers and operational adjustments.

---

## 🎯 Objective

Build an intelligent airfare estimation system that:

- Learns pricing patterns from data
- Captures nonlinear cost influences
- Provides real-time predictions via UI
- Demonstrates end-to-end ML workflow

---

## 🤖 Machine Learning Problem Type

- **Type:** Supervised Learning
- **Category:** Regression
- **Target Variable:** `Price`

---

## 📊 Dataset Strategy

Since real airline pricing data is proprietary, a synthetic dataset was generated programmatically.

The dataset models realistic factors:

- Airline selection
- Stops & duration effects
- Route influence
- Travel class multipliers
- Seasonal demand variation

---

## 🧹 Data Processing Steps

- Categorical encoding using LabelEncoder
- Feature selection
- Train-test split
- Linear Regression modeling

---

## 🧠 Feature Engineering

The system incorporates domain-inspired features:

- Seasonal pricing (Peak months)
- Stops adjustment
- Duration-based pricing
- Class-based multipliers
- Random variability for realism

---

## 🏆 Model Used

- **Linear Regression**

Chosen for:

- Interpretability
- Fast training
- Baseline regression modeling
- Suitability for structured features

---

## 📈 Prediction Logic

The model learns relationships of form:

y = β₀ + β₁x₁ + β₂x₂ + ... + ε

Where airfare price is influenced by operational and contextual variables.

---

## 🌐 Deployment

An interactive web application was built using **Streamlit**.

Workflow:

User Input → Feature Encoding → Model Prediction → Price Output

Includes:

- Dynamic UI controls
- Input validation
- Price breakdown visualization
- Prediction summary table

---

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- Scikit-Learn
- Streamlit
- Matplotlib

---

## 🚀 How to Run

### Install Dependencies

```bash
pip install -r requirements.txt



---

# ✅ **requirements.txt (Very Important for GitHub)**

Create file:

```txt id="as9d12"
numpy
pandas
scikit-learn
streamlit
matplotlib
