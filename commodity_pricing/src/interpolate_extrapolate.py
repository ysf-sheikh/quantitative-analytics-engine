import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# 1. Raw Data Input
data = {
    'Dates': [
        '10/31/20', '11/30/20', '12/31/20', '1/31/21', '2/28/21', '3/31/21', '4/30/21', '5/31/21', '6/30/21', '7/31/21', '8/31/21', '9/30/21',
        '10/31/21', '11/30/21', '12/31/21', '1/31/22', '2/28/22', '3/31/22', '4/30/22', '5/31/22', '6/30/22', '7/31/22', '8/31/22', '9/30/22',
        '10/31/22', '11/30/22', '12/31/22', '1/31/23', '2/28/23', '3/31/23', '4/30/23', '5/31/23', '6/30/23', '7/31/23', '8/31/23', '9/30/23',
        '10/31/23', '11/30/23', '12/31/23', '1/31/24', '2/29/24', '3/31/24', '4/30/24', '5/31/24', '6/30/24', '7/31/24', '8/31/24', '9/30/24'
    ],
    'Prices': [
        10.1, 10.3, 11.0, 10.9, 10.9, 10.9, 10.4, 9.84, 10.0, 10.1, 10.3, 10.2,
        10.1, 11.2, 11.4, 11.5, 11.8, 11.5, 10.7, 10.7, 10.4, 10.5, 10.4, 10.8,
        11.0, 11.6, 11.6, 12.1, 11.7, 12.0, 11.5, 11.2, 10.9, 11.4, 11.1, 11.5,
        11.8, 12.2, 12.8, 12.6, 12.4, 12.7, 12.1, 11.4, 11.5, 11.6, 11.5, 11.8
    ]
}

df = pd.DataFrame(data)
df['Dates'] = pd.to_datetime(df['Dates'], format='%m/%d/%y')
df['Days'] = (df['Dates'] - df['Dates'].min()).dt.days

# 2. Fit Cubic Spline for the historical/existing period
cs = CubicSpline(
    df['Days'],
    df['Prices'],
    bc_type='natural',
    extrapolate=False
)

# 3. Model Extrapolation (Linear trend + Annual seasonality)
# Calculate baseline drift and seasonal amplitude from history to project 1 year out
X = np.column_stack([
    np.ones(len(df)), 
    df['Days'], 
    np.sin(2 * np.pi * df['Days'] / 365.25), 
    np.cos(2 * np.pi * df['Days'] / 365.25)
])
beta, _, _, _ = np.linalg.lstsq(X, df['Prices'], rcond=None)

def extrapolate_price(days):
    return beta[0] + beta[1] * days + beta[2] * np.sin(2 * np.pi * days / 365.25) + beta[3] * np.cos(2 * np.pi * days / 365.25)

# 4. Master Pricing Function
def get_gas_price(date_str):
    """Takes a date string (YYYY-MM-DD) and returns the estimated natural gas price."""
    target_date = pd.to_datetime(date_str)
    days_diff = (target_date - df['Dates'].min()).days
    
    # If within historical data range
    if 0 <= days_diff <= df['Days'].max():
        return float(cs(days_diff))
    # If in the 1-year extrapolation zone
    elif df['Days'].max() < days_diff <= (df['Days'].max() + 365):
        return float(extrapolate_price(days_diff))
    else:
        max_date = df['Dates'].max() + pd.Timedelta(days=365)
        return f"Date falls out of bounds (Max extrapolation: {max_date.date()})"

# 5. Visualization Generation
all_days = np.arange(0, df['Days'].max() + 366)
all_dates = [df['Dates'].min() + pd.Timedelta(days=int(d)) for d in all_days]
fitted_prices = [get_gas_price(d.strftime('%Y-%m-%d')) for d in all_dates]

plt.figure(figsize=(12, 6))
plt.scatter(df['Dates'], df['Prices'], color='red', zorder=5, label='Monthly Market Snapshots')
plt.plot(all_dates, fitted_prices, color='blue', label='Continuous Pricing Curve (Spline + Forecast)')
plt.axvline(df['Dates'].max(), color='grey', linestyle='--', label='Start of Extrapolation')
plt.title('Natural Gas Forward Curve & Extrapolation Engine')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

# Quick test print
print(f"Price on 2022-06-15 (Historical Interpolation): ${get_gas_price('2022-06-15'):.2f}")
print(f"Price on 2025-01-15 (Future Seasonal Extrapolation): ${get_gas_price('2025-01-15'):.2f}")
