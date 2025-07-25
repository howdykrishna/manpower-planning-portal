# Manpower Planning Portal
# Built with Streamlit - Supports Ratio Trend, Markov, and Monte Carlo Simulation

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="Workforce Forecast Portal", layout="wide")
st.title("📊 Manpower Planning Portal")
st.markdown("""
This portal helps HR and business leaders perform workforce forecasting using:
- **Ratio Trend Analysis**
- **Markov Analysis**
- **Monte Carlo Simulation**

✅ Choose your model below and enter relevant inputs. 
📥 Download results in Excel format.
""")

model = st.selectbox("🔍 Select Forecasting Model", 
                     ["Ratio Trend Analysis", "Markov Analysis", "Monte Carlo Simulation"])

# Helper to download DataFrame as Excel
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
        writer.close()
    processed_data = output.getvalue()
    return processed_data

# --- RATIO TREND ANALYSIS ---
def ratio_trend():
    st.subheader("📈 Ratio Trend Analysis Inputs")
    st.markdown("""Used for **small to medium businesses** with consistent growth metrics. 
    Industries: **Retail, Manufacturing, Services**.
    """)

    rows = st.number_input("Number of Historical Years", 2, 10, 3)
    data = []
    for i in range(rows):
        col1, col2, col3 = st.columns(3)
        with col1:
            year = st.text_input(f"Year {i+1}", value=str(2020 + i), key=f"yr_{i}")
        with col2:
            employees = st.number_input(f"Employees in {year}", key=f"emp_{i}")
        with col3:
            output = st.number_input(f"Output (e.g., Revenue) in {year}", key=f"out_{i}")
        data.append({'Year': year, 'Employees': employees, 'Output': output})

    df = pd.DataFrame(data)
    try:
        df['Ratio'] = df['Employees'] / df['Output']
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        avg_ratio = df['Ratio'].mean()
        forecast_output = st.number_input("Forecasted Output for Next Year (Revenue/Units)", value=1000.0)
        forecast_year = str(int(df['Year'].iloc[-1]) + 1)
        forecast_employees = forecast_output * avg_ratio
        forecast_df = pd.DataFrame([{"Year": forecast_year, "Forecasted Output": forecast_output, "Estimated Employees": int(round(forecast_employees))}])

        st.success(f"🔮 Estimated Employees Required for {forecast_year}: {int(round(forecast_employees))}")
        st.dataframe(df)
        st.dataframe(forecast_df)

        if st.download_button("📥 Download Forecast as Excel", data=to_excel(forecast_df), file_name="ratio_trend_forecast.xlsx"):
            st.toast("Downloaded successfully!")
    except Exception as e:
        st.error(f"❌ Error calculating ratio trend: {e}")

# --- MARKOV ANALYSIS ---
def markov_model():
    st.subheader("📊 Markov Analysis Inputs")
    st.markdown("""Used for **medium to large companies** with structured career progression. 
    Industries: **IT Services, BPO, Consulting, Banking**.
    """)

    state_names = st.text_area("Enter Role/Level States (comma-separated)", "Junior,Mid,Senior,Exit")
    states = [x.strip() for x in state_names.split(",")]
    n = len(states)

    current_headcount = []
    st.markdown("### Current Headcount by State")
    for state in states:
        current_headcount.append(st.number_input(f"{state} Count", key=f"curr_{state}", min_value=0))

    st.markdown("### Transition Probability Matrix")
    st.markdown("Each row should sum to 1.0 (100%).")
    matrix = []
    try:
        for i in range(n):
            row = st.text_input(f"Transition probabilities from {states[i]} (comma-separated)",
                                value=','.join([str(round(1.0/n, 2))]*n), key=f"row_{i}")
            matrix.append([float(x.strip()) for x in row.split(",")])

        years = st.slider("Number of Years to Forecast", 1, 10, 3)
        current = np.array(current_headcount)
        mat = np.array(matrix)
        history = [current]
        for _ in range(years):
            current = np.dot(current, mat)
            history.append(current)

        forecast_df = pd.DataFrame(history, columns=states)
        forecast_df.insert(0, "Year", [f"Year {i}" for i in range(years+1)])
        st.dataframe(forecast_df)
        st.line_chart(forecast_df.set_index("Year"))

        if st.download_button("📥 Download Markov Forecast", data=to_excel(forecast_df), file_name="markov_forecast.xlsx"):
            st.toast("Downloaded successfully!")
    except Exception as e:
        st.error(f"❌ Error in Markov calculation: {e}")

# --- MONTE CARLO SIMULATION ---
def monte_carlo():
    st.subheader("🌀 Monte Carlo Simulation")
    st.markdown("""Used for **large enterprises** managing uncertainty and volatility. 
    Industries: **FMCG, Pharma, Telecom, BFSI, Manufacturing**.
    """)

    init = st.number_input("Initial Headcount", 0, 10000, 500)
    years = st.slider("Forecast Period (Years)", 1, 10, 5)
    sims = st.slider("Number of Simulations", 100, 5000, 1000)
    attrition = st.slider("Annual Attrition Rate (%)", 0, 100, 10) / 100
    hiring = st.slider("Annual Hiring Rate (%)", 0, 200, 50) / 100
    promotion = st.slider("Annual Promotion Rate (%)", 0, 100, 5) / 100

    try:
        results = []
        for _ in range(sims):
            count = init
            yearly = []
            for _ in range(years):
                exits = np.random.binomial(count, attrition)
                joins = np.random.poisson(hiring * count)
                moves = np.random.binomial(count, promotion)
                count = max(0, count - exits + joins)
                yearly.append(count)
            results.append(yearly)

        avg_counts = np.mean(results, axis=0)
        monte_df = pd.DataFrame({"Year": [f"Year {i+1}" for i in range(years)], "Average Headcount": avg_counts})
        st.dataframe(monte_df)
        st.line_chart(monte_df.set_index("Year"))

        if st.download_button("📥 Download Monte Carlo Forecast", data=to_excel(monte_df), file_name="montecarlo_forecast.xlsx"):
            st.toast("Downloaded successfully!")
    except Exception as e:
        st.error(f"❌ Error running simulation: {e}")

# --- Dispatcher ---
if model == "Ratio Trend Analysis":
    ratio_trend()
elif model == "Markov Analysis":
    markov_model()
elif model == "Monte Carlo Simulation":
    monte_carlo()

st.markdown("---")
st.markdown("Built with ❤️ by your HR Analytics Assistant | [Deploy to Streamlit Cloud or Render.com]")
