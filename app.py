import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go

# =====================
# Helper function to load pickle files
# =====================
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# =====================
# Load transformers and model
# =====================
numeric_imputer = load_pickle("Files/numeric_imputer.pkl")
categorical_imputer = load_pickle("Files/categorical_imputer.pkl")

cat_cols = ['RainToday', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm']
encoders = {col: load_pickle(f"Files/{col}_label_encoder.pkl") for col in cat_cols}

scaler_std = load_pickle("Files/scaler_std.pkl")
scaler_mm = load_pickle("Files/scaler_mm.pkl")
pt_yeo = load_pickle("Files/pt_yeo_johnson.pkl")

model = load_pickle("Files/DecisionTreeClassifier.pkl")

# =====================
# Streamlit Config & CSS
# =====================
st.set_page_config(page_title="Weather Prediction AUS", layout="wide")

st.markdown("""
<style>
h1, h2, h3 {
    color: #1E3D59;
    font-weight: bold;
}
.stButton>button {
    background-color: #2b2a33; 
    color: white;
    border-radius: 12px;
    font-size: 18px;
    padding: 0.5em 1.5em;
    transition: 0.3s;
}
.stButton>button:hover {
    background-color: #36343f;
    transform: scale(1.05);
}
.rain-card {
    padding: 1em;
    border-radius: 15px;
    background: rgba(55, 53, 62, 0.7);
    text-align: center;
    font-size: 22px;
    color: white;
    font-weight: bold;
    margin-top: 1em;
}
</style>
""", unsafe_allow_html=True)

st.title("🌦️ Weather AUS Prediction App")

# =====================
# Sidebar navigation
# =====================
page = st.sidebar.radio("Home", ["Prediction", "Rain analysis by Date"])

# =====================
# Page 1 - Prediction
# =====================
if page == "Prediction":
    st.header("Will it rain tomorrow?")

    col1, col2 = st.columns(2)

    with col1:
        location = st.selectbox("Location", encoders['Location'].classes_)
        rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 0.0)
        windgustdir = st.selectbox("Wind Gust Direction", encoders['WindGustDir'].classes_)
        windgustspeed = st.slider("Wind Gust Speed", 0, 150, 20)
        winddir9am = st.selectbox("Wind Direction 9am", encoders['WindDir9am'].classes_)
        winddir3pm = st.selectbox("Wind Direction 3pm", encoders['WindDir3pm'].classes_)

    with col2:
        humidity9am = st.slider("Humidity 9am (%)", 0, 100, 50)
        humidity3pm = st.slider("Humidity 3pm (%)", 0, 100, 50)
        cloud9am = st.slider("Cloud 9am (oktas)", 0, 9, 4)
        cloud3pm = st.slider("Cloud 3pm (oktas)", 0, 9, 4)
        raintoday = st.selectbox("Rain Today", encoders['RainToday'].classes_)
        risk_mm = st.number_input("Risk (mm)", 0.0, 500.0, 0.0)

    if st.button("Predict"):
        input_data = pd.DataFrame({
            "Location": [location],
            "Rainfall": [rainfall],
            "WindGustDir": [windgustdir],
            "WindGustSpeed": [windgustspeed],
            "WindDir9am": [winddir9am],
            "WindDir3pm": [winddir3pm],
            "Humidity9am": [humidity9am],
            "Humidity3pm": [humidity3pm],
            "Cloud9am": [cloud9am],
            "Cloud3pm": [cloud3pm],
            "RainToday": [raintoday],
            "RISK_MM": [risk_mm]
        })

        for col in cat_cols:
            input_data[col] = encoders[col].transform(input_data[col])

        input_data[numeric_imputer.feature_names_in_] = numeric_imputer.transform(input_data[numeric_imputer.feature_names_in_])
        input_data[categorical_imputer.feature_names_in_] = categorical_imputer.transform(input_data[categorical_imputer.feature_names_in_])
        input_data[scaler_std.feature_names_in_] = scaler_std.transform(input_data[scaler_std.feature_names_in_])
        input_data[scaler_mm.feature_names_in_] = scaler_mm.transform(input_data[scaler_mm.feature_names_in_])
        input_data[pt_yeo.feature_names_in_] = pt_yeo.transform(input_data[pt_yeo.feature_names_in_])

        prediction = model.predict(input_data)[0]
        
        if prediction == 1:
            st.markdown('<div class="rain-card">🌧️ It will RAIN Tomorrow!</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="rain-card">☀️ No Rain Tomorrow!</div>', unsafe_allow_html=True)


# =====================
# Page 2 - Rain Analysis
# =====================

elif page == "Rain analysis by Date":
    st.header("Rain Analysis")

    df = pd.read_csv("weatherAUS.csv", parse_dates=['Date'])
    # ===== Columns for filters =====
    col_main, col_optional = st.columns(2)

    with col_main:
        location = st.selectbox("Choose Location", df['Location'].unique())
        month = st.selectbox("Choose Month", sorted(df['Date'].dt.month.unique()))

    with col_optional:
        year = st.selectbox("Choose Year (optional)", [None]+list(sorted(df['Date'].dt.year.unique())))
        day = st.selectbox("Choose Day (optional)", [None]+list(sorted(df['Date'].dt.day.unique())))

    df_filtered = df[(df['Location']==location) & (df['Date'].dt.month==month)]
    if year is not None:
        df_filtered = df_filtered[df_filtered['Date'].dt.year==year]
    if day is not None:
        df_filtered = df_filtered[df_filtered['Date'].dt.day==day]

    if df_filtered.empty:
        st.warning("No data for selected filters.")
    else:
        rain_days = df_filtered[df_filtered["RainTomorrow"]=="Yes"]
        if len(rain_days) > 0:
            st.success(f"🌧️ It rained {len(rain_days)} day(s) in the selection!")
        else:
            st.info("☀️ No rain in the selection!")

        df_sorted = df_filtered.sort_values("Date")

        # RainTomorrow Bar
        st.subheader("Rain Tomorrow Distribution")
        fig1 = px.histogram(df_filtered, x="RainTomorrow", color="RainTomorrow",
                            color_discrete_map={"Yes":"skyblue","No":"lightgray"},
                            title="Rain vs No Rain")
        st.plotly_chart(fig1, use_container_width=True)

        # Rainfall Scatter + Line
        st.subheader("Rainfall Over Time (mm)")
        fig2 = px.scatter(df_sorted, x="Date", y="Rainfall", trendline="lowess",
                          title="Rainfall Trend")
        st.plotly_chart(fig2, use_container_width=True)

        # Humidity Area
        st.subheader("Humidity (%) Over Time")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df_sorted["Date"], y=df_sorted["Humidity9am"],
                                  mode='lines', name='Humidity 9am', fill='tozeroy',
                                  line=dict(color='skyblue')))
        fig3.add_trace(go.Scatter(x=df_sorted["Date"], y=df_sorted["Humidity3pm"],
                                  mode='lines', name='Humidity 3pm', fill='tozeroy',
                                  line=dict(color='lightgreen')))
        fig3.update_layout(title="Humidity Trend")
        st.plotly_chart(fig3, use_container_width=True)

        # Cloud Line
        st.subheader("Cloud Coverage (oktas)")
        fig4 = px.line(df_sorted, x="Date", y=["Cloud9am","Cloud3pm"],
                       labels={"value":"Cloud (oktas)","variable":"Time"},
                       title="Cloud Coverage Over Time")
        st.plotly_chart(fig4, use_container_width=True)

        # Pressure Scatter
        st.subheader("Pressure (hPa) Over Time")
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=df_sorted["Date"], y=df_sorted["Pressure9am"],
                                  mode='markers', name='Pressure 9am', marker_color='skyblue'))
        fig5.add_trace(go.Scatter(x=df_sorted["Date"], y=df_sorted["Pressure3pm"],
                                  mode='markers', name='Pressure 3pm', marker_color='lightgreen'))
        fig5.update_layout(title="Pressure Trend")
        st.plotly_chart(fig5, use_container_width=True)

        # Wind Speed Bar
        st.subheader("Wind Speed (km/h) Over Time")
        fig6 = go.Figure()
        fig6.add_trace(go.Bar(x=df_sorted["Date"], y=df_sorted["WindSpeed9am"],
                              name="Wind 9am", marker_color='skyblue'))
        fig6.add_trace(go.Bar(x=df_sorted["Date"], y=df_sorted["WindSpeed3pm"],
                              name="Wind 3pm", marker_color='lightgreen'))
        fig6.update_layout(title="Wind Speed", barmode='group')
        st.plotly_chart(fig6, use_container_width=True)

        # Temperature Box Plot
        st.subheader("Temperature (°C) Distribution")
        fig7 = go.Figure()
        fig7.add_trace(go.Box(y=df_sorted["MaxTemp"], name="Max Temp", marker_color='skyblue'))
        fig7.add_trace(go.Box(y=df_sorted["MinTemp"], name="Min Temp", marker_color='lightgreen'))
        fig7.update_layout(title="Max & Min Temperature Distribution")
        st.plotly_chart(fig7, use_container_width=True)
