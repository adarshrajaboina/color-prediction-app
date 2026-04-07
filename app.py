import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier



st.set_page_config(
    page_title="Paint Color Prediction",
    page_icon="🎨",
    layout="wide"
)

# -------------------------------
# Exact Color Mapping
# -------------------------------
color_map = {
    "forest_green": "#228B22",
    "ivory_white": "#FFFFF0",
    "beige_brown": "#D2B48C",
    "brick_red": "#CB4154",
    "plum_purple": "#8E4585",
    "magenta": "#FF00FF",
    "tangerine_orange": "#F28500",
    "turquoise": "#40E0D0",
    "navy_blue": "#000080",
    "royal_blue": "#4169E1",
    "pure_white": "#FFFFFF",
    "golden_yellow": "#FFD700",
    "mustard_yellow": "#FFDB58",
    "teal_blue": "#008080",
    "tan_brown": "#D2B48C",
    "sea_green": "#2E8B57",
    "scarlet_red": "#FF2400",
    "burnt_orange": "#CC5500",
    "sunflower_yellow": "#FFC512",
    "lavender_purple": "#E6E6FA",
    "chocolate_brown": "#7B3F00",
    "azure_blue": "#007FFF",
    "lime_green": "#32CD32",
    "coral_orange": "#FF7F50",
    "coffee_brown": "#6F4E37",
    "charcoal_grey": "#36454F",
    "silver_grey": "#C0C0C0",
    "amber_yellow": "#FFBF00",
    "deep_black": "#000000",
    "lemon_yellow": "#FFF44F",
    "crimson_red": "#DC143C",
    "violet_purple": "#8F00FF",
    "aqua_cyan": "#00FFFF",
    "emerald_green": "#50C878",
    "off_white": "#FAF9F6",
    "rust_orange": "#B7410E",
    "cobalt_blue": "#0047AB",
    "sky_blue": "#87CEEB",
    "olive_green": "#808000",
    "peach_orange": "#FFE5B4",
    "rose_pink": "#FF66CC",
    "maroon_red": "#800000",
    "walnut_brown": "#5C4033",
    "ruby_red": "#9B111E",
    "ash_grey": "#B2BEB5",
    "mint_green": "#98FF98"
}

# -------------------------------
# Helper function for text color
# -------------------------------
def get_text_color(hex_color):
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    brightness = (r * 299 + g * 587 + b * 114) / 1000
    return "black" if brightness > 150 else "white"


# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to:",
    ["📄 Description", "📈 Prediction"]
)

# -------------------------------
# DESCRIPTION PAGE
# -------------------------------
if page == "📄 Description":
    st.title("🎨 Paint Color Prediction System")

    st.markdown("""
    ## Welcome

### 🔹 Overview
This application predicts the **final paint color** based on raw material composition, formulation parameters, and process conditions using Machine Learning.

---

### 🔹 Problem Statement
-In paint manufacturing, predicting the final color before production is difficult and time-consuming.  
-Traditional methods require trial-and-error, leading to increased cost and production delay.

👉 This system solves that problem by **predicting color instantly using ML models**.

---

### 🔹 Objective
-Predict paint color using formulation inputs  
-Reduce manual experimentation  
-Improve production efficiency  
-Provide instant color preview for decision making  

---

### 🔹 Features
- ✅ Predicts color name using ML model  
- ✅ Displays exact color preview (HEX-based)  
- ✅ User-friendly interface using Streamlit  
- ✅ Handles real-world formulation inputs  
- ✅ Fast and interactive prediction  

---

### 🔹 Machine Learning Approach
- Data preprocessing (encoding + scaling)  
- Feature alignment using trained columns  
- Model used: **Random Forest Classifier**  

👉 Why Random Forest?
- Handles complex relationships  
- Works well with mixed data types  
- Gives high accuracy and stability  

---

### 🔹 Workflow
1. User enters input values  
2. Data is preprocessed & encoded  
3. Features are scaled  
4. Model predicts color name  
5. Color name is mapped to HEX code  
6. Color preview is displayed  

---

### 🔹 Output
- Predicted color name  
- Exact color visualization  
- HEX color code  
- Input summary  

---
                
### 🔹 Recommendations 
Based on the predicted color and input parameters, the system can provide smart suggestions:

- 🎯 Adjust pigment percentage to achieve brighter or darker shades  
- 🎨 Modify reflectance values to fine-tune color accuracy  
- ⚙️ Optimize mixing speed and time for better consistency  
- 🌡️ Maintain optimal temperature to avoid color variation  
- 🧪 Suggest suitable binder and solvent combinations  

👉 These recommendations help improve **product quality and consistency**.
                
---

### 🔹 Advantages
- Reduces trial-and-error in manufacturing  
- Saves time and cost  
- Easy to use for non-technical users  
- Real-time prediction  

---

### 🔹 Future Scope
- Predict exact RGB values instead of labels  
- Add viscosity prediction in same app  
- Improve accuracy using deep learning  
- Deploy as industrial tool  


    ---
    Go to **Prediction** page to test the model.
    """)

# -------------------------------
# PREDICTION PAGE
# -------------------------------
elif page == "📈 Prediction":
    st.title("🎯 Paint Color Prediction")

    @st.cache_resource
    def load_files():
        model = joblib.load("color_model.pkl")
        scaler = joblib.load("color_scaler.pkl")
        columns = joblib.load("color_columns.pkl")
        return model, scaler, columns

    model, scaler, columns = load_files()

    st.write("Enter values and click **Predict Color**.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            pigment_type = st.selectbox(
                "Pigment Type",
                ["phthalocyanine blue", "titanium dioxide", "iron oxide", "carbon black"]
            )

            pigment_percent = st.number_input("Pigment Percent(range: 10-40)", min_value=10.0, max_value=40.0, value=20.0)
            pigment_particle_size = st.number_input("Pigment Particle Size(range: 0.1-10.0)", min_value=0.1, max_value=10.0, value=1.0)
            pigment_mixture_ratio = st.number_input("Pigment Mixture Ratio(range: 0.0-1.0)", min_value=0.0, max_value=1.0, value=0.5)
            brightness = st.number_input("Brightness(range: 0-100)", min_value=0.0, max_value=100.0, value=50.0)
            opacity = st.number_input("Opacity(range: 0-100)", min_value=0.0, max_value=100.0, value=50.0)
            tone_strength = st.number_input("Tone Strength(range: 0-100)", min_value=0.0, max_value=100.0, value=50.0)
            gloss_level = st.number_input("Gloss Level(range: 0-100)", min_value=0.0, max_value=100.0, value=50.0)
            reflectance_450nm = st.number_input("Reflectance 450nm(range: 0.0-1.0)", min_value=0.0, max_value=1.0, value=0.5)
            reflectance_550nm = st.number_input("Reflectance 550nm(range: 0.0-1.0)", min_value=0.0, max_value=1.0, value=0.5)

        with col2:
            reflectance_650nm = st.number_input("Reflectance 650nm(range: 0.0-1.0)", min_value=0.0, max_value=1.0, value=0.5)

            binder_type = st.selectbox(
                "Binder Type",
                ["acrylic", "alkyd", "epoxy", "polyurethane"]
            )

            binder_percent = st.number_input("Binder Percent(range: 10-50)", min_value=10.0, max_value=50.0, value=25.0)

            solvent_type = st.selectbox(
                "Solvent Type",
                ["water", "xylene", "toluene", "mineral spirit"]
            )

            solvent_percent = st.number_input("Solvent Percent(range: 5-40)", min_value=5.0, max_value=40.0, value=15.0)

            base_type = st.selectbox(
                "Base Type",
                ["water-based", "solvent-based"]
            )

            mixing_speed_rpm = st.number_input("Mixing Speed RPM(range: 500-2000)", min_value=500.0, max_value=2000.0, value=1000.0)
            mixing_time_min = st.number_input("Mixing Time (min)(range: 10-60)", min_value=10.0, max_value=60.0, value=30.0)
            temperature_c = st.number_input("Temperature (°C)(range: 15-40)", min_value=15.0, max_value=40.0, value=25.0)
            drying_time_hr = st.number_input("Drying Time (hr)(range: 1-10)", min_value=1.0, max_value=10.0, value=4.0)

        submitted = st.form_submit_button("Predict Color")

    if submitted:
        try:
            sample = {
                'pigment_type': pigment_type,
                'pigment_percent': pigment_percent,
                'pigment_particle_size': pigment_particle_size,
                'pigment_mixture_ratio': pigment_mixture_ratio,
                'brightness': brightness,
                'opacity': opacity,
                'tone_strength': tone_strength,
                'gloss_level': gloss_level,
                'reflectance_450nm': reflectance_450nm,
                'reflectance_550nm': reflectance_550nm,
                'reflectance_650nm': reflectance_650nm,
                'binder_type': binder_type,
                'binder_percent': binder_percent,
                'solvent_type': solvent_type,
                'solvent_percent': solvent_percent,
                'base_type': base_type,
                'mixing_speed_rpm': mixing_speed_rpm,
                'mixing_time_min': mixing_time_min,
                'temperature_c': temperature_c,
                'drying_time_hr': drying_time_hr
            }

            sample_df = pd.DataFrame([sample])

            # One-hot encoding
            sample_df = pd.get_dummies(sample_df)

            # Match training columns
            sample_df = sample_df.reindex(columns=columns, fill_value=0)

            # Scale
            sample_scaled = scaler.transform(sample_df)

            # Predict
            prediction = model.predict(sample_scaled)
            predicted_color = str(prediction[0]).strip().lower()

            # Exact mapped hex color
            hex_color = color_map.get(predicted_color, "#808080")

            # Text color based on background
            text_color = get_text_color(hex_color)

            # Show result
            st.success(f"✅ Predicted Color: **{predicted_color}**")

            # Color preview box
            st.markdown(f"""
            <div style="
                width: 400px;
                height: 160px;
                background-color: {hex_color};
                border-radius: 15px;
                border: 3px solid black;
                margin-top: 15px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: {text_color};
                font-size: 24px;
                font-weight: bold;
            ">
                {predicted_color}
            </div>
            """, unsafe_allow_html=True)

            st.write(f"🎨 HEX Code: {hex_color}")

            st.subheader("Input Summary")
            st.dataframe(pd.DataFrame([sample]), use_container_width=True)

        except Exception as e:
            st.error(f"Error during prediction: {e}")