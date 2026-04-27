# Artificial Neural Network — Power Output Prediction

A regression-based ANN built with TensorFlow/Keras to predict the **net hourly electrical energy output** of a Combined Cycle Power Plant based on ambient environmental variables.

---

## About the Project

This project builds and trains an Artificial Neural Network from scratch to perform regression on real-world power plant data. The model learns the relationship between environmental conditions (temperature, pressure, humidity, exhaust vacuum) and power output.

**Dataset:** Combined Cycle Power Plant Dataset (`Folds5x2_pp.xlsx`)  
**Task:** Regression — predict electrical power output (MW)  
**Framework:** TensorFlow / Keras

---

## Model Architecture

```
Input Layer     →  4 features
Hidden Layer 1  →  6 neurons, ReLU activation
Hidden Layer 2  →  6 neurons, ReLU activation
Output Layer    →  1 neuron (regression output)
```

**Training Config:**
- Optimizer: Adam
- Loss Function: Mean Squared Error (MSE)
- Epochs: 100
- Batch Size: 32
- Train/Test Split: 80% / 20%

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| TensorFlow / Keras | ANN model building & training |
| NumPy | Numerical operations |
| Pandas | Data loading & preprocessing |
| Scikit-learn | Train/test split |

---

## How to Run

**1. Clone the repo**
```bash
git clone https://github.com/Shashankroblox/Ml-model.git
cd Ml-model
```

**2. Install dependencies**
```bash
pip install tensorflow numpy pandas scikit-learn openpyxl
```

**3. Add the dataset**  
Place `Folds5x2_pp.xlsx` in the project root directory.

**4. Run the notebook**  
Open `ANN_Power_Prediction.ipynb` in Jupyter Notebook or Google Colab and run all cells.

---

## Results

The model was trained for 100 epochs and converged successfully, producing predictions closely aligned with actual power output values.

Sample output (predicted vs actual):
```
[[431.ghost  431.23]
 [462.12  461.88]
 [473.34  474.01]
 ...]
```

---

## Project Structure

```
Ml-model/
├── ANN_Power_Prediction.ipynb   # Main notebook
├── Folds5x2_pp.xlsx             # Dataset (add manually)
├── README.md                    # Project documentation
├── app.py                       # XGBoost Streamlit app
└── requirements.txt             # Dependencies
```

---

## Author

**Shashank Chakravarti**   
[GitHub](https://github.com/Shashankroblox)
