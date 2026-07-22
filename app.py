import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

# =========================================================
# 0. AUTOMATIC DATASET INITIALIZER (For missing files)
# =========================================================
raw_path = "Urbanization.xlsx"

def generate_synthetic_data():
    """Generates realistic dummy data if the original Excel file is missing."""
    np.random.seed(42)
    n_villages = 100
    
    states = ["Maharashtra", "Tamil Nadu", "Uttar Pradesh", "Karnataka", "Kerala"]
    villages = [f"Village_{i+1}" for i in range(n_villages)]
    
    data = {
        'state_name': np.random.choice(states, n_villages),
        'village_name': villages,
        'Population_Density': np.random.randint(100, 1200, n_villages),
        'Distance_From_City': np.random.randint(5, 80, n_villages),
        'Number_of_Industries': np.random.randint(0, 15, n_villages),
        'Population_Growth_Rate': np.random.uniform(0.5, 4.5, n_villages),
        'Road_Density': np.random.uniform(1.0, 15.0, n_villages),
        'Literacy_Rate': np.random.uniform(50.0, 95.0, n_villages),
        'Employment_Rate': np.random.uniform(40.0, 85.0, n_villages),
        'Internet_Penetration': np.random.randint(10, 95, n_villages),
        'Electricity_Access': np.random.randint(60, 100, n_villages),
        'Water_Access': np.random.randint(50, 100, n_villages),
        'Infrastructure_Index': np.random.randint(20, 90, n_villages),
        'schools': np.random.randint(1, 8, n_villages),
        'Hospitals_Count': np.random.randint(0, 4, n_villages)
    }
    return pd.DataFrame(data)

# Load real file if it exists, otherwise build fallback dataset quietly
if os.path.exists(raw_path):
    raw_df = pd.read_excel(raw_path)
else:
    raw_df = generate_synthetic_data()

# =========================================================
# 1. ANALYTICS & PREDICTION ENGINE
# =========================================================
FEATURE_COLS = [
    'Population_Density', 'Distance_From_City', 'Number_of_Industries',
    'Population_Growth_Rate', 'Road_Density', 'Literacy_Rate', 'Employment_Rate',
    'Internet_Penetration', 'Electricity_Access', 'Water_Access', 'Infrastructure_Index',
    'schools', 'Hospitals_Count'
]

def train_ml_model(df):
    X = df[FEATURE_COLS].copy()
    X['Distance_From_City'] = X['Distance_From_City'].max() - X['Distance_From_City']
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    y = X_scaled.mean(axis=1) 
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    importances = model.feature_importances_
    ml_weights = dict(zip(FEATURE_COLS, importances))
    
    df['URI_Score'] = model.predict(X_scaled)
    df['URI_Score'] = (df['URI_Score'] - df['URI_Score'].min()) / (df['URI_Score'].max() - df['URI_Score'].min())
    
    return df, ml_weights, scaler, model

def classify_tier(score):
    if score >= 0.70: return 'Tier 1: High Urbanization Potential'
    elif score >= 0.45: return 'Tier 2: Emerging/Moderate'
    else: return 'Tier 3: Stable Rural'

# Execute Calculation Pipelines
processed_df, AI_WEIGHTS, dataset_scaler, trained_model = train_ml_model(raw_df)
processed_df['Urbanization_Tier'] = processed_df['URI_Score'].apply(classify_tier)
processed_df['Rank'] = processed_df['URI_Score'].rank(ascending=False, method='min').astype(int)

# =========================================================
# 2. STREAMLIT APPLICATION LAYOUT
# =========================================================
st.set_page_config(page_title="Rurban Predictor", layout="wide")
st.title("🏙️ Indian Rural Areas Urbanization Potential Predictor")

tab1, tab2, tab3 = st.tabs(["📊 Regional Insight Dashboard", "🎛️ Policy Simulator", "🔍 Factor Weight Analysis"])

# --- TAB 1: REGIONAL INSIGHT DASHBOARD ---
with tab1:
    st.subheader("Regional Performance Filter")
    selected_state = st.selectbox("Choose Target State Context:", sorted(processed_df['state_name'].unique()))
    state_df = processed_df[processed_df['state_name'] == selected_state]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Villages Evaluated", len(state_df))
    col2.metric("Average State Score", round(state_df['URI_Score'].mean(), 2))
    col3.metric("High Potential Nodes", len(state_df[state_df['Urbanization_Tier'].str.contains("Tier 1")]))
    
    fig = px.scatter(
        state_df, x="Population_Density", y="Infrastructure_Index", 
        color="Urbanization_Tier", size="Population_Growth_Rate", hover_name="village_name",
        labels={"Population_Density": "Population Density", "Infrastructure_Index": "Infrastructure Index"}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(state_df[['Rank', 'village_name', 'URI_Score', 'Urbanization_Tier']].sort_values(by="Rank"), use_container_width=True)

# --- TAB 2: POLICY SIMULATOR ---
with tab2:
    st.subheader("Real-Time Variable Adjustment Matrix")
    target_village = st.selectbox("Select Village Node:", sorted(processed_df['village_name'].unique()))
    v_data = processed_df[processed_df['village_name'] == target_village].iloc[0]
    
    st.warning(f"Current Baseline Score: {round(v_data['URI_Score'], 3)} ({v_data['Urbanization_Tier']})")
    
    sc1, sc2 = st.columns(2)
    with sc1:
        sim_dist = st.slider("Distance From City (km)", int(processed_df['Distance_From_City'].min()), int(processed_df['Distance_From_City'].max()), int(v_data['Distance_From_City']))
        sim_roads = st.slider("Local Road Density", float(processed_df['Road_Density'].min()), float(processed_df['Road_Density'].max()), float(v_data['Road_Density']))
        sim_ind = st.slider("Industrial Counter", int(processed_df['Number_of_Industries'].min()), int(processed_df['Number_of_Industries'].max()), int(v_data['Number_of_Industries']))
        sim_net = st.slider("Internet Penetration (%)", int(processed_df['Internet_Penetration'].min()), int(processed_df['Internet_Penetration'].max()), int(v_data['Internet_Penetration']))
    with sc2:
        sim_elec = st.slider("Electricity Access (%)", int(processed_df['Electricity_Access'].min()), int(processed_df['Electricity_Access'].max()), int(v_data['Electricity_Access']))
        sim_sch = st.slider("Schools Count", int(processed_df['schools'].min()), int(processed_df['schools'].max()), int(v_data['schools']))
        sim_hosp = st.slider("Hospitals Count", int(processed_df['Hospitals_Count'].min()), int(processed_df['Hospitals_Count'].max()), int(v_data['Hospitals_Count']))
        sim_infra = st.slider("Overall Infrastructure Index", int(processed_df['Infrastructure_Index'].min()), int(processed_df['Infrastructure_Index'].max()), int(v_data['Infrastructure_Index']))

    input_data = {
        'Population_Density': v_data['Population_Density'],
        'Distance_From_City': processed_df['Distance_From_City'].max() - sim_dist, 
        'Number_of_Industries': sim_ind,
        'Population_Growth_Rate': v_data['Population_Growth_Rate'],
        'Road_Density': sim_roads,
        'Literacy_Rate': v_data['Literacy_Rate'],
        'Employment_Rate': v_data['Employment_Rate'],
        'Internet_Penetration': sim_net,
        'Electricity_Access': sim_elec,
        'Water_Access': v_data['Water_Access'],
        'Infrastructure_Index': sim_infra,
        'schools': sim_sch,
        'Hospitals_Count': sim_hosp
    }
    
    input_df = pd.DataFrame([input_data])[FEATURE_COLS]
    scaled_input = dataset_scaler.transform(input_df)
    raw_pred = trained_model.predict(scaled_input)[0]
    
    min_score = trained_model.predict(dataset_scaler.transform(processed_df[FEATURE_COLS])).min()
    max_score = trained_model.predict(dataset_scaler.transform(processed_df[FEATURE_COLS])).max()
    sim_score = (raw_pred - min_score) / (max_score - min_score)

    st.markdown("---")
    rc1, rc2 = st.columns(2)
    rc1.metric("Simulated Potential Score", round(sim_score, 3), delta=round(sim_score - v_data['URI_Score'], 3))
    rc2.subheader(f"Predicted Status: **{classify_tier(sim_score)}**")

# --- TAB 3: FACTOR WEIGHT ANALYSIS ---
with tab3:
    st.subheader("📊 Key Factors Driving Urban Growth")
    st.write("This chart shows which infrastructure features have the biggest impact on turning a village into an urban center.")
    
    weights_df = pd.DataFrame(list(AI_WEIGHTS.items()), columns=['Development Factor', 'Impact Level']).sort_values(by='Impact Level', ascending=False)
    
    fig_weights = px.bar(
        weights_df, x='Impact Level', y='Development Factor', 
        orientation='h', title='Infrastructure Impact Breakdown',
        labels={'Impact Level': 'Impact Level', 'Development Factor': 'Development Factor'}
    )
    fig_weights.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_weights, use_container_width=True)
