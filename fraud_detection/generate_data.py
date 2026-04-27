import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 50000

data = {
    'step': np.random.randint(1, 745, n_samples),
    'type': np.random.choice(['CASH_OUT', 'PAYMENT', 'CASH_IN', 'TRANSFER', 'DEBIT'], n_samples, p=[0.2, 0.3, 0.2, 0.2, 0.1]),
    'amount': np.random.exponential(5000, n_samples),
    'nameOrig': [f'O{i}' for i in range(n_samples)],
    'oldbalanceOrg': np.random.exponential(50000, n_samples),
    'newbalanceOrig': np.random.exponential(50000, n_samples),
    'nameDest': [f'D{i}' for i in range(n_samples)],
    'oldbalanceDest': np.random.exponential(50000, n_samples),
    'newbalanceDest': np.random.exponential(50000, n_samples),
    'isFraud': np.random.binomial(1, 0.015, n_samples)
}

df = pd.DataFrame(data)
df.to_csv('data/paysim.csv', index=False)
print(f'Created synthetic dataset: {len(df)} rows')
