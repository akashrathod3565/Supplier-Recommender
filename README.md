# Supplier Recommender POC

##  Project Description
This is a Proof-of-Concept application that recommends the top 3 eligible suppliers for a given item and quantity using Machine Learning (XGBoost) and a simple Tkinter GUI.  
It simulates supplier and historical RFQ data to train the model and recommends based on normalized features like Price, Reliability, Quality, and Compliance.

---

##  Features
- Supplier simulation (50 suppliers).
- Historical RFQ data generation (5000 samples).
- Machine learning model (XGBoost) to predict supplier suitability.
- Simple GUI using Tkinter for interactive supplier recommendations.
- Top 3 supplier suggestions displayed in a table.

---

##  Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn xgboost
   ```

---

##  Usage

1. Run the Python script:
   ```bash
   python supplier_recommender.py
   ```

2. Enter:
   - Item ID (e.g., `Item_01`).
   - Quantity (e.g., `800`).
   - Preferred Plant (Plant_A / Plant_B / Plant_C).

3. Click **Recommend Top 3 Suppliers** to get recommendations.

---

## 🛠 Future Improvements
- Integrate with a real database.
- Add advanced filtering options.
- Improve the GUI with additional supplier details.

---

