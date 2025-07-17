import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib 

df = pd.read_csv('EV_DataSet.csv')

numeric_cols = [
    'Battery Electric Vehicles (BEVs)',
    'Plug-In Hybrid Electric Vehicles (PHEVs)',
    'Electric Vehicle (EV) Total',
    'Non-Electric Vehicle Total',
    'Total Vehicles'
]
for col in numeric_cols:
    df[col] = df[col].astype(str).str.replace(',', '', regex=False).astype(float)

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df[df['Date'].notnull()]

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

df = df[df['Electric Vehicle (EV) Total'].notnull()]

df['County'] = df['County'].fillna('Unknown')
df['State'] = df['State'].fillna('Unknown')

Q1 = df['Percent Electric Vehicles'].quantile(0.25)
Q3 = df['Percent Electric Vehicles'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df['Percent Electric Vehicles'] = np.where(df['Percent Electric Vehicles'] > upper_bound, upper_bound, np.where(df['Percent Electric Vehicles'] < lower_bound, lower_bound, df['Percent Electric Vehicles']))


le_county = LabelEncoder()
le_state = LabelEncoder()
le_vehicle_use = LabelEncoder()

df['County_Encoded'] = le_county.fit_transform(df['County'])
df['State_Encoded'] = le_state.fit_transform(df['State'])
df['Vehicle_Primary_Use_Encoded'] = le_vehicle_use.fit_transform(df['Vehicle Primary Use'])

features = [
    'Year', 'Month',
    'County_Encoded', 'State_Encoded', 'Vehicle_Primary_Use_Encoded',
    'Battery Electric Vehicles (BEVs)',
    'Plug-In Hybrid Electric Vehicles (PHEVs)',
    'Non-Electric Vehicle Total',
    'Total Vehicles'
]
X = df[features]
y = df['Electric Vehicle (EV) Total']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"--- Model Evaluation (RandomForestRegressor) ---")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

latest_year = df['Year'].max()

most_frequent_county = df['County'].mode()[0]
most_frequent_state = df['State'].mode()[0]
most_frequent_vehicle_use = df['Vehicle Primary Use'].mode()[0]

encoded_mf_county = le_county.transform([most_frequent_county])[0]
encoded_mf_state = le_state.transform([most_frequent_state])[0]
encoded_mf_vehicle_use = le_vehicle_use.transform([most_frequent_vehicle_use])[0]

latest_year_data = df[df['Year'] == latest_year]
avg_bevs = latest_year_data['Battery Electric Vehicles (BEVs)'].mean()
avg_phevs = latest_year_data['Plug-In Hybrid Electric Vehicles (PHEVs)'].mean()
avg_non_ev_total = latest_year_data['Non-Electric Vehicle Total'].mean()
avg_total_vehicles = latest_year_data['Total Vehicles'].mean()

future_years_months = []
for year in range(latest_year + 1, latest_year + 4):
    for month in range(1, 13):
        future_years_months.append({'Year': year, 'Month': month})

future_df = pd.DataFrame(future_years_months)

future_df['County_Encoded'] = encoded_mf_county
future_df['State_Encoded'] = encoded_mf_state
future_df['Vehicle_Primary_Use_Encoded'] = encoded_mf_vehicle_use
future_df['Battery Electric Vehicles (BEVs)'] = avg_bevs
future_df['Plug-In Hybrid Electric Vehicles (PHEVs)'] = avg_phevs
future_df['Non-Electric Vehicle Total'] = avg_non_ev_total
future_df['Total Vehicles'] = avg_total_vehicles

future_predictions = model.predict(future_df[features])
future_df['Forecasted Electric Vehicle (EV) Total'] = future_predictions

print("\n--- Forecasted EV Adoption for a Representative Case (Most Frequent County/State/Vehicle Use) ---")

annual_forecast = future_df.groupby('Year')['Forecasted Electric Vehicle (EV) Total'].sum().reset_index()
print(annual_forecast)

plt.figure(figsize=(12, 7))

df_agg_year = df.groupby('Year')['Electric Vehicle (EV) Total'].sum().reset_index()
plt.plot(df_agg_year['Year'], df_agg_year['Electric Vehicle (EV) Total'], label='Historical EV Total (Aggregated)', marker='o', color='blue')

plt.plot(annual_forecast['Year'], annual_forecast['Forecasted Electric Vehicle (EV) Total'], label='Forecasted EV Total (Aggregated)', linestyle='--', marker='x', color='red')

plt.title('Historical and Forecasted Electric Vehicle (EV) Adoption')
plt.xlabel('Year')
plt.ylabel('Total Electric Vehicles')
plt.grid(True)
plt.legend()
plt.xticks(np.arange(df['Year'].min(), future_df['Year'].max() + 1, 1))
plt.tight_layout()
plt.savefig('ev_adoption_forecast_rf.png')
plt.show()