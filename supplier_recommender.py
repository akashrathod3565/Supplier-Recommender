
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

np.random.seed(42)

# -------------------------------
# Simulate Suppliers Data (50 suppliers)
# -------------------------------
num_suppliers = 50
suppliers = pd.DataFrame({
    'Supplier_ID': [f"S{i+1}" for i in range(num_suppliers)],
    'Location': np.random.choice(['Plant_A', 'Plant_B', 'Plant_C'], num_suppliers),
    'Capacity': np.random.randint(500, 5000, num_suppliers),
    'Avg_Price': np.random.randint(80, 150, num_suppliers),
    'Delivery_Reliability': np.random.uniform(0.6, 1.0, num_suppliers),
    'Quality_Score': np.random.uniform(0.6, 1.0, num_suppliers),
    'Compliance_Score': np.random.uniform(0.7, 1.0, num_suppliers)
})

# Normalize numeric features for ML model training
for col in ['Avg_Price','Delivery_Reliability','Quality_Score','Compliance_Score','Capacity']:
    suppliers[col+'_norm'] = (suppliers[col]-suppliers[col].min()) / (suppliers[col].max()-suppliers[col].min())

# -------------------------------
# Simulate Historical RFQs Data (5000 rows)
# -------------------------------
num_rfq = 5000
items = [f"Item_{i}" for i in range(1,51)]  # 50 unique items
historical_rfq = []

for _ in range(num_rfq):
    rfq_qty = np.random.randint(50, 3000)
    rfq_item = np.random.choice(items)
    winner = suppliers.sample(1, weights=(suppliers['Delivery_Reliability']*0.4 +
                                           suppliers['Quality_Score']*0.3 +
                                           suppliers['Compliance_Score']*0.3)).iloc[0]
    historical_rfq.append({'Item_ID': rfq_item, 'Quantity': rfq_qty, 'Supplier_ID': winner['Supplier_ID']})

historical_rfq = pd.DataFrame(historical_rfq)

# -------------------------------
# Prepare Data for ML Model
# -------------------------------
train_data = historical_rfq.merge(suppliers, on='Supplier_ID', how='left')
train_data['Selected'] = 1

# Generate negative examples for balanced dataset
negatives = train_data.copy()
negatives = negatives.loc[:, ['Item_ID','Quantity','Supplier_ID','Selected']].copy()
negatives['Selected'] = 0
train_data = pd.concat([train_data, negatives], ignore_index=True)

# Label encode Item_ID for model input
all_items = list(train_data['Item_ID'].unique())
le = LabelEncoder()
le.fit(all_items)
train_data['Item_ID_enc'] = le.transform(train_data['Item_ID'])

# Features and target
feature_cols = ['Avg_Price_norm','Delivery_Reliability_norm','Quality_Score_norm','Compliance_Score_norm','Quantity','Item_ID_enc']
X_train = train_data[feature_cols]
y_train = train_data['Selected']

# Train XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
model.fit(X_train, y_train)

# -------------------------------
# Build Tkinter GUI for User Interaction
# -------------------------------
root = tk.Tk()
root.title("Supplier Recommender POC")

# Input fields for Item ID, Quantity, Preferred Plant
tk.Label(root, text="Item ID:").grid(row=0, column=0, padx=5, pady=5, sticky='e')
item_entry = tk.Entry(root)
item_entry.insert(0, "Item_01")
item_entry.grid(row=0, column=1, padx=5, pady=5)

tk.Label(root, text="Quantity:").grid(row=1, column=0, padx=5, pady=5, sticky='e')
quantity_entry = tk.Entry(root)
quantity_entry.insert(0, "800")
quantity_entry.grid(row=1, column=1, padx=5, pady=5)

tk.Label(root, text="Preferred Plant:").grid(row=2, column=0, padx=5, pady=5, sticky='e')
plant_var = tk.StringVar(value='Plant_A')
plant_menu = ttk.Combobox(root, textvariable=plant_var, values=['Plant_A','Plant_B','Plant_C'])
plant_menu.grid(row=2, column=1, padx=5, pady=5)

# Treeview table to show Top 3 supplier recommendations
cols = ['Supplier_ID', 'Predicted_Prob', 'Avg_Price', 'Delivery_Reliability', 'Quality_Score', 'Compliance_Score']
tree = ttk.Treeview(root, columns=cols, show='headings')
for col in cols:
    tree.heading(col, text=col)
    tree.column(col, width=120)
tree.grid(row=4, column=0, columnspan=2, padx=5, pady=5)

# Recommendation logic
def recommend():
    item_id = item_entry.get()
    try:
        quantity = int(quantity_entry.get())
    except:
        messagebox.showerror("Error", "Quantity must be an integer")
        return
    plant = plant_var.get()

    # Filter eligible suppliers
    eligible = suppliers[(suppliers['Location']==plant) & (suppliers['Capacity']>=quantity) & (suppliers['Compliance_Score']>=0.85)].copy()
    if eligible.empty:
        messagebox.showinfo("Result","No eligible suppliers found!")
        return

    eligible['Quantity'] = quantity
    if item_id not in le.classes_:
        le.classes_ = np.append(le.classes_, item_id)
    eligible['Item_ID_enc'] = le.transform([item_id]*len(eligible))

    X_pred = eligible[feature_cols]
    eligible['Predicted_Prob'] = model.predict_proba(X_pred)[:,1]
    top3 = eligible.sort_values('Predicted_Prob',ascending=False).head(3)

    # Populate table with top 3 recommended suppliers
    for row in tree.get_children():
        tree.delete(row)
    for _, row in top3.iterrows():
        tree.insert('',tk.END,values=(row['Supplier_ID'], round(row['Predicted_Prob'],2),
                                      row['Avg_Price'], round(row['Delivery_Reliability'],2),
                                      round(row['Quality_Score'],2), round(row['Compliance_Score'],2)))

# Button to trigger recommendation
tk.Button(root, text="Recommend Top 3 Suppliers", command=recommend).grid(row=3, column=0, columnspan=2, pady=10)

# Start GUI event loop
root.mainloop()
