from pathlib import Path
import numpy as np
import pandas as pd


RANDOM_STATE = 42
N_CUSTOMERS = 500

def generate_dataset(n_customers=N_CUSTOMERS, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)
    classes = rng.choice(["LOW_VALUE", "REGULAR", "HIGH_VALUE"], size=n_customers, p=[0.34, 0.40, 0.26])
    profiles = {
        "LOW_VALUE": (30, 300_000, 18, 1_400, 2.0, 25, 8),
        "REGULAR": (39, 550_000, 42, 2_300, 4.0, 18, 13),
        "HIGH_VALUE": (45, 900_000, 75, 3_600, 6.0, 10, 20),
    }
    rows = []
    for customer_id, label in enumerate(classes, start=1001):
        age, income, purchases, order_value, frequency, discount, online_share = profiles[label]
        total_purchases = max(3, int(round(rng.normal(purchases, purchases * 0.22))))
        average_order_value = max(300, round(rng.normal(order_value, order_value * 0.18), 2))
        purchase_frequency = max(0.5, round(rng.normal(frequency, 0.7), 2))
        total_spending = round(max(average_order_value * total_purchases * rng.normal(1, 0.10), 500), 2)
        online_purchases = int(np.clip(round(total_purchases * rng.normal(online_share / 100, 0.12)), 0, total_purchases))
        rows.append({
            "Customer_ID": customer_id,
            "Age": int(np.clip(round(rng.normal(age, 7)), 18, 75)),
            "Annual_Income": round(max(100_000, rng.normal(income, income * 0.20)), 2),
            "Total_Purchases": total_purchases,
            "Average_Order_Value": average_order_value,
            "Purchase_Frequency": purchase_frequency,
            "Total_Spending": total_spending,
            "Discount_Usage": int(np.clip(round(rng.normal(discount, 5)), 0, 40)),
            "Online_Purchases": online_purchases,
            "Store_Purchases": total_purchases - online_purchases,
            "Customer_Class": label,
        })
    return pd.DataFrame(rows)


output_path = Path("data/customer_data.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
df = generate_dataset()
df.to_csv(output_path, index=False)
print(f"Created {len(df)} customers at {output_path}")
print(df["Customer_Class"].value_counts().sort_index())
