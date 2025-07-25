# app.py
from flask import Flask, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import os

app = Flask(__name__)

# --- Data Loading and Preprocessing Functions (from your original script) ---
def load_and_preprocess_data(file_path):
    """
    Loads the EV dataset, preprocesses it by cleaning numeric columns,
    converting dates, handling missing values, and encoding categorical features.

    Args:
        file_path (str): The path to the CSV dataset.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
        tuple: A tuple containing the fitted LabelEncoders for county, state, and vehicle use.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        print("Please ensure the 'EV_DataSet.csv' file is in the same directory as app.py.")
        return None, (None, None, None)

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
    df['Percent Electric Vehicles'] = np.where(
        df['Percent Electric Vehicles'] > upper_bound,
        upper_bound,
        np.where(df['Percent Electric Vehicles'] < lower_bound, lower_bound, df['Percent Electric Vehicles'])
    )

    le_county = LabelEncoder()
    le_state = LabelEncoder()
    le_vehicle_use = LabelEncoder()

    df['County_Encoded'] = le_county.fit_transform(df['County'])
    df['State_Encoded'] = le_state.fit_transform(df['State'])
    df['Vehicle_Primary_Use_Encoded'] = le_vehicle_use.fit_transform(df['Vehicle Primary Use'])

    return df, (le_county, le_state, le_vehicle_use)

def train_model(X_train, y_train):
    """
    Trains a RandomForestRegressor model.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model and returns performance metrics.
    """
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "MAE": round(mae, 2),
        "MSE": round(mse, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 2)
    }
    return metrics

def get_feature_importances(model, features):
    """
    Returns feature importances from the trained model.
    """
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    return importances.to_dict()

def prepare_future_data(df, le_county, le_state, le_vehicle_use,
                        target_county, target_state, target_vehicle_use):
    """
    Prepares a DataFrame for future predictions based on specified county, state,
    and vehicle use, using average values for other features from the latest year.
    """
    latest_year = df['Year'].max()

    try:
        encoded_target_county = le_county.transform([target_county])[0]
    except ValueError:
        encoded_target_county = le_county.transform([df['County'].mode()[0]])[0]

    try:
        encoded_target_state = le_state.transform([target_state])[0]
    except ValueError:
        encoded_target_state = le_state.transform([df['State'].mode()[0]])[0]

    try:
        encoded_target_vehicle_use = le_vehicle_use.transform([target_vehicle_use])[0]
    except ValueError:
        encoded_target_vehicle_use = le_vehicle_use.transform([df['Vehicle Primary Use'].mode()[0]])[0]

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

    future_df['County_Encoded'] = encoded_target_county
    future_df['State_Encoded'] = encoded_target_state
    future_df['Vehicle_Primary_Use_Encoded'] = encoded_target_vehicle_use
    future_df['Battery Electric Vehicles (BEVs)'] = avg_bevs
    future_df['Plug-In Hybrid Electric Vehicles (PHEVs)'] = avg_phevs
    future_df['Non-Electric Vehicle Total'] = avg_non_ev_total
    future_df['Total Vehicles'] = avg_total_vehicles

    return future_df

@app.route('/')
def dashboard():
    file_path = 'EV_DataSet.csv'
    df, (le_county, le_state, le_vehicle_use) = load_and_preprocess_data(file_path)

    if df is None:
        return "Error: Data file not found or could not be processed. Please check server logs.", 500

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

    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    importances = get_feature_importances(model, features)

    # --- Automatic Forecasting Scenario (Most Frequent Case) ---
    target_county = df['County'].mode()[0]
    target_state = df['State'].mode()[0]
    target_vehicle_use = df['Vehicle Primary Use'].mode()[0]

    future_df = prepare_future_data(df, le_county, le_state, le_vehicle_use,
                                    target_county, target_state, target_vehicle_use)

    future_predictions = model.predict(future_df[features])
    future_df['Forecasted Electric Vehicle (EV) Total'] = future_predictions

    # Prepare data for Chart.js
    historical_chart_data = df.groupby('Year')['Electric Vehicle (EV) Total'].sum().reset_index()
    historical_chart_data = historical_chart_data.to_dict(orient='records')

    forecasted_chart_data = future_df.groupby('Year')['Forecasted Electric Vehicle (EV) Total'].sum().reset_index()
    forecasted_chart_data = forecasted_chart_data.to_dict(orient='records')

    return render_template('index.html',
                           metrics=metrics,
                           importances=importances,
                           historical_data=json.dumps(historical_chart_data),
                           forecasted_data=json.dumps(forecasted_chart_data),
                           forecasted_data_list=forecasted_chart_data, # Pass the list for Jinja loop
                           forecast_scenario=f"{target_county}, {target_state}, {target_vehicle_use}")

if __name__ == '__main__':
    # Ensure 'templates' directory exists
    if not os.path.exists('templates'):
        os.makedirs('templates')

    # Create the index.html file inside the templates directory
    html_template_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EV Adoption Forecast Dashboard</title>
    <!-- Tailwind CSS CDN -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Chart.js CDN for plotting -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Inter', sans-serif;
            background-color: #f0f2f5;
            display: flex;
            justify-content: center;
            align-items: flex-start;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            background-color: #ffffff;
            border-radius: 12px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
            padding: 32px;
            max-width: 1200px;
            width: 100%;
            display: flex;
            flex-direction: column;
            gap: 24px;
        }
        h1, h2 {
            color: #2c3e50;
            font-weight: 700;
        }
        .section-title {
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 8px;
            margin-bottom: 16px;
        }
        .metrics-grid, .importances-list {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
        }
        .metric-item, .importance-item {
            background-color: #ecf0f1;
            padding: 12px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        .metric-item strong, .importance-item strong {
            display: block;
            font-size: 1.2em;
            color: #2c3e50;
        }
        .metric-item span, .importance-item span {
            font-size: 0.9em;
            color: #7f8c8d;
        }
        #forecastOutput ul {
            list-style-type: none;
            padding: 0;
        }
        #forecastOutput li {
            background-color: #f7f9fb;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 8px 12px;
            margin-bottom: 5px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        #forecastOutput li:last-child {
            margin-bottom: 0;
        }
    </style>
</head>
<body class="bg-gray-100 p-8">
    <div class="container">
        <h1 class="text-3xl text-center mb-6">Electric Vehicle Adoption Forecast Dashboard</h1>

        <!-- Model Evaluation Section -->
        <div class="bg-white p-6 rounded-lg shadow-md">
            <h2 class="text-xl section-title mb-4">Model Evaluation (RandomForestRegressor)</h2>
            <div class="metrics-grid">
                <div class="metric-item">
                    <strong>{{ metrics['MAE'] | round(2) }}</strong>
                    <span>Mean Absolute Error (MAE)</span>
                </div>
                <div class="metric-item">
                    <strong>{{ metrics['MSE'] | round(2) }}</strong>
                    <span>Mean Squared Error (MSE)</span>
                </div>
                <div class="metric-item">
                    <strong>{{ metrics['RMSE'] | round(2) }}</strong>
                    <span>Root Mean Squared Error (RMSE)</span>
                </div>
                <div class="metric-item">
                    <strong>{{ metrics['R2'] | round(2) }}</strong>
                    <span>R-squared (R2)</span>
                </div>
            </div>
        </div>

        <!-- Feature Importances Section -->
        <div class="bg-white p-6 rounded-lg shadow-md">
            <h2 class="text-xl section-title mb-4">Feature Importances</h2>
            <div class="importances-list">
                {% for feature, importance in importances.items() %}
                <div class="importance-item"><strong>{{ feature }}</strong> <span>{{ "%.6f" | format(importance) }}</span></div>
                {% endfor %}
            </div>
        </div>

        <!-- Forecast Chart Section -->
        <div class="bg-white p-6 rounded-lg shadow-md">
            <h2 class="text-xl section-title mb-4">Historical and Forecasted EV Adoption for {{ forecast_scenario }}</h2>
            <canvas id="evAdoptionChart"></canvas>
            <div id="forecastOutput" class="mt-4 text-gray-700">
                <h3 class="text-lg font-semibold mb-2">Annual Forecasted EV Totals:</h3>
                <ul>
                    {% for data in forecasted_data_list %}
                    <li><span>Year {{ data['Year'] }}:</span> <strong>{{ "{:,.0f}".format(data['Forecasted Electric Vehicle (EV) Total']) }}</strong></li>
                    {% endfor %}
                </ul>
            </div>
        </div>
    </div>

    <script>
        // Parse JSON data passed from Flask
        const historicalData = JSON.parse('{{ historical_data | tojson }}');
        const forecastedData = JSON.parse('{{ forecasted_data | tojson }}');

        const evAdoptionChartCtx = document.getElementById('evAdoptionChart').getContext('2d');
        let evChart;

        function renderChart() {
            const allYears = [...historicalData.map(d => d.year), ...forecastedData.map(d => d.Year)];
            const minYear = Math.min(...allYears);
            const maxYear = Math.max(...allYears);
            const labels = Array.from({ length: maxYear - minYear + 1 }, (_, i) => minYear + i);

            const historicalEvTotals = labels.map(year => {
                const data = historicalData.find(d => d.year === year);
                return data ? data.evTotal : null;
            });

            const forecastedEvTotals = labels.map(year => {
                const data = forecastedData.find(d => d.Year === year);
                return data ? data['Forecasted Electric Vehicle (EV) Total'] : null;
            });

            if (evChart) {
                evChart.destroy();
            }

            evChart = new Chart(evAdoptionChartCtx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'Historical EV Total (Aggregated)',
                            data: historicalEvTotals,
                            borderColor: '#3498db', /* Blue */
                            backgroundColor: 'rgba(52, 152, 219, 0.2)',
                            fill: false,
                            tension: 0.1,
                            pointRadius: 5,
                            pointBackgroundColor: '#3498db',
                            spanGaps: true
                        },
                        {
                            label: `Forecasted EV Total (Most Frequent Scenario)`,
                            data: forecastedEvTotals,
                            borderColor: '#e74c3c', /* Red */
                            backgroundColor: 'rgba(231, 76, 60, 0.2)',
                            borderDash: [5, 5],
                            fill: false,
                            tension: 0.1,
                            pointRadius: 5,
                            pointBackgroundColor: '#e74c3c',
                            spanGaps: true
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Historical and Forecasted Electric Vehicle (EV) Adoption',
                            font: { size: 18, weight: 'bold' }
                        },
                        tooltip: {
                            mode: 'index',
                            intersect: false,
                            callbacks: {
                                label: function(context) {
                                    let label = context.dataset.label || '';
                                    if (label) {
                                        label += ': ';
                                    }
                                    if (context.parsed.y !== null) {
                                        label += new Intl.NumberFormat('en-US').format(context.parsed.y);
                                    }
                                    return label;
                                }
                            }
                        },
                        legend: {
                            position: 'bottom',
                        }
                    },
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Year',
                                font: { size: 14 }
                            },
                            ticks: {
                                autoSkip: true,
                                maxRotation: 0,
                                minRotation: 0,
                                callback: function(value) {
                                    return labels[value];
                                }
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Total Electric Vehicles',
                                font: { size: 14 }
                            },
                            beginAtZero: true,
                            ticks: {
                                callback: function(value) {
                                    return value.toLocaleString();
                                }
                            }
                        }
                    }
                }
            });
        }

        document.addEventListener('DOMContentLoaded', renderChart);
    </script>
</body>
</html>
    """
    with open(os.path.join('templates', 'index.html'), 'w') as f:
        f.write(html_template_content)

    app.run(debug=True)
