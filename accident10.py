import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier # CART implementation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import os 
import joblib
import os

# Load models safely
cart_clf, casualty_regressor, accident_regressor = None, None, None

try:
    cart_clf = joblib.load("models/cart_clf.pkl")
except FileNotFoundError:
    st.error("❌ Severity prediction model not found (cart_clf.pkl)")

try:
    casualty_regressor = joblib.load("models/casualty_regressor.pkl")
except FileNotFoundError:
    st.warning("⚠️ Casualty prediction model not found (casualty_regressor.pkl)")

try:
    accident_regressor = joblib.load("models/accident_regressor.pkl")
except FileNotFoundError:
    st.warning("⚠️ Accident count prediction model not found (accident_regressor.pkl)")


weather_map = {
    1: 'Fine', 2: 'Raining', 3: 'Snowing',
    4: 'Fine + high winds',
    5: 'Raining + high winds',
    6: 'Snowing + high winds',
    7: 'Fog or mist'
}
road_type_map = {
    1: 'Roundabout', 2: 'One way street', 3: 'Dual carriageway',
    6: 'Single carriageway', 7: 'Slip road',
    12: 'One way street/Slip road'
   
}
light_conditions_map = {
    1: 'Daylight',
    4: 'Darkness - lights lit',
    5: 'Darkness - lights unlit',
    6: 'Darkness - no lighting',
    7: 'Darkness - lighting unknown'
}
road_surface_map = {
    1: 'Dry', 2: 'Wet or damp', 3: 'Snow',
    4: 'Frost or ice', 5: 'Flood over 3cm. deep'
}
day_map = { 
    1: 'Sunday', 2: 'Monday', 3: 'Tuesday', 4: 'Wednesday', 5: 'Thursday', 6: 'Friday', 7: 'Saturday'
    
}
urban_rural_map = { 
    1: 'Urban', 2: 'Rural'
}

CODES_TO_EXCLUDE = {-1, 8, 9, 3} 

severity_map = {
    0: 'Predicted: Slight Severity',
    1: 'Predicted: Serious/Fatal Severity'
}

# --- FIXED FILE PATHS ---
DATA_FILE = r"C:\Users\nkart\OneDrive\Documents\DA-mini\archive (2)\Integrated_Accident_Data.csv"
BACKGROUND_IMAGE = r"C:\Users\nkart\Downloads\dramatic-car-accident-scene-dark-highway-safety-awareness-moody-setting-heavily-damaged-emphasizing-road-342680698.jpg"

# --- File Existence Checks (Keep as before) ---
if not os.path.exists(DATA_FILE):
    st.error(f"Error: Data file not found at the specified path: '{DATA_FILE}'")
    st.error("Please ensure the file exists and the path is correct.")
    st.stop()

if not os.path.exists(BACKGROUND_IMAGE):
    st.warning(f"Warning: Background image not found at path: '{BACKGROUND_IMAGE}'. Using default background.")
    BACKGROUND_IMAGE = None

# --- UI Styling (Keep as before) ---
st.set_page_config(page_title="Road Accident Insights", layout="wide")

def get_img_base64(file_path):
    # ... (keep existing function)
    try:
        # Determine image type from file extension
        _, ext = os.path.splitext(file_path)
        img_format = ext.lower().strip('.')
        if img_format == 'jpg':
            img_format = 'jpeg' # Common alternative
        if img_format not in ['png', 'jpeg', 'gif', 'webp', 'bmp']:
             st.warning(f"Unsupported background image format: {img_format}. Trying anyway.")
             # Default to png if unsure, might not work
             img_format = 'png'

        with open(file_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode()
            return f"data:image/{img_format};base64,{encoded}" # Use correct image format
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error loading background image: {e}")
        return None

img_base64_data = get_img_base64(BACKGROUND_IMAGE) if BACKGROUND_IMAGE else None
background_css = f"""
    background-image: url("{img_base64_data}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
""" if img_base64_data else "background-color: #22223B;"

st.markdown(f"""
    <style>
    /* ... (keep existing CSS styles) ... */
    .stApp {{
        {background_css}
        color: white !important; /* Default app text white */
    }}
    /* General text styling for app elements */
    h1, h2, h3, h4, h5, h6, p, label, .stTextInput > label, .stSelectbox > label, .stSlider > label, .stMultiSelect > label {{
        color: white !important;
        text-shadow: 1px 1px 5px black;
    }}
    .main-container {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 90vh;
        text-align: center;
    }}
    .main-container h1 {{
        font-size: 48px; /* Adjusted size */
        font-weight: bold;
        text-shadow: 3px 3px 10px black;
        margin-bottom: 50px;
    }}
    .btn-container {{
        display: flex;
        gap: 30px;
        justify-content: center;
    }}
     /* Button Style */
    .stButton>button {{
        background-color: #ffffff20 !important;
        color: white !important;
        font-weight: bold !important;
        border: 2px solid white !important;
        border-radius: 10px !important;
        padding: 15px 25px !important;
        font-size: 16px !important;
        text-shadow: 1px 1px 2px black !important;
        transition: all 0.2s ease-in-out !important;
        white-space: nowrap !important; /* prevent wrapping */
    }}
    .stButton>button:hover {{
        background-color: #ffffff40 !important;
        transform: scale(1.05) !important;
    }}
    /* Ensure dropdowns/selects are readable on dark background */
    .stSelectbox [data-baseweb="select"] > div:first-child {{
        background-color: rgba(255, 255, 255, 0.1);
        color: white; /* Text inside the dropdown box */
        text-shadow: 1px 1px 2px black; /* Add shadow to text in box */
    }}
     /* Style dropdown options list */
    div[data-baseweb="popover"] li {{
        background-color: #444 !important; /* Darker background for options */
        color: white !important; /* White text for options */
    }}
    div[data-baseweb="popover"] li:hover {{
        background-color: #666 !important; /* Lighter gray on hover */
    }}

    .stMultiSelect [data-baseweb="select"] > div:first-child {{
         background-color: rgba(255, 255, 255, 0.1);
         color: white; /* Text inside the multiselect box */
         text-shadow: 1px 1px 2px black; /* Add shadow to text in box */
    }}

    /* Style slider labels */
    .stSlider > label {{
        color: white !important;
        text-shadow: 1px 1px 5px black;
    }}
    </style>
""", unsafe_allow_html=True)


# --- Data Loading and Preprocessing (Keep as before) ---
@st.cache_data
def load_data():
    # ... (keep existing data loading and cleaning logic) ...
    try:
        # Try reading with default UTF-8 encoding first
        try:
            df = pd.read_csv(DATA_FILE, low_memory=False, encoding='utf-8')
        except UnicodeDecodeError:
            # If UTF-8 fails, try Latin-1 as a common alternative
            st.warning("UTF-8 decoding failed, trying Latin-1 encoding...")
            df = pd.read_csv(DATA_FILE, low_memory=False, encoding='latin1')

        # --- Basic Cleaning ---
        # Define critical columns needed for the app to function
        base_critical_cols = ['Accident_Severity', 'Weather_Conditions', 'Road_Type', 'Speed_limit', 'Light_Conditions', 'Road_Surface_Conditions']
        # Add columns needed for prediction/analysis if not already included
        extended_critical_cols = base_critical_cols + ['Day_of_Week', 'Urban_or_Rural_Area']

        # Check if critical columns exist
        missing_base_cols = [col for col in base_critical_cols if col not in df.columns]
        if missing_base_cols:
             st.error(f"Critical columns missing from the dataset: {missing_base_cols}. Cannot proceed.")
             st.stop()

        # Convert relevant columns to numeric, coercing errors *before* dropping NaNs
        for col in extended_critical_cols:
             if col in df.columns: # Check if column exists before trying conversion
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows where essential columns have NaN after coercion
        df.dropna(subset=base_critical_cols, inplace=True)

        # --- Feature Engineering for Prediction ---
        if 'Accident_Severity' in df.columns:
             df['Is_Severe'] = df['Accident_Severity'].apply(lambda x: 1 if x in [1, 2] else 0)

        # Ensure integer types for codes AFTER handling NaNs
        int_cols = ['Weather_Conditions', 'Road_Type', 'Light_Conditions', 'Road_Surface_Conditions', 'Day_of_Week', 'Urban_or_Rural_Area']
        for col in int_cols:
            if col in df.columns:
                df.dropna(subset=[col], inplace=True)
                if not df.empty:
                    if col in df.columns and not df[col].isnull().all():
                         # Before converting to int, ensure values are valid (not in exclude list if possible, though primary filtering is in UI)
                         # We mainly need the correct type here. UI filtering handles presentation.
                         try:
                             df[col] = df[col].astype(int)
                         except ValueError:
                              st.warning(f"Could not convert column '{col}' to int after dropping NaNs. Contains non-integer values.")
                              # Attempt conversion to numeric again, then int
                              df[col] = pd.to_numeric(df[col], errors='coerce')
                              df.dropna(subset=[col], inplace=True)
                              if col in df.columns and not df[col].isnull().all():
                                  df[col] = df[col].astype(int)
                              else:
                                   st.error(f"Failed to convert '{col}' to integer type. Check data.")
                                   # Remove column from further processing if it's problematic?
                                   # Or stop execution? For now, just warn.


                    elif col not in df.columns:
                        st.warning(f"Column '{col}' was removed during NaN dropping.")
                    else: # Column exists but is all NaN
                        st.warning(f"Column '{col}' contains only NaN values after dropping. Cannot convert to int.")
                else:
                    st.warning(f"DataFrame became empty after dropping NaNs in column '{col}'.")

        if df.empty:
             st.error("No data remaining after initial cleaning and NaN handling. Cannot proceed.")
             st.stop()

        return df

    except FileNotFoundError:
        st.error(f"Error: Data file not found at path '{DATA_FILE}'.")
        st.stop()
    except pd.errors.EmptyDataError:
        st.error(f"Error: The data file '{DATA_FILE}' is empty.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during data loading: {e}")
        st.stop()

df = load_data()

# --- Model Training (Keep as before) ---
prediction_features = [
    'Weather_Conditions', 'Road_Type', 'Speed_limit', 'Light_Conditions',
    'Road_Surface_Conditions', 'Day_of_Week', 'Urban_or_Rural_Area'
]
# ... (keep existing model training logic) ...
missing_features = [f for f in prediction_features if f not in df.columns]
if missing_features:
    st.error(f"Error: The following features required for prediction are missing from the loaded data: {missing_features}")
    st.error("Please check your CSV file column names or adjust the 'prediction_features' list in the code.")
    st.stop()

if 'Is_Severe' not in df.columns:
     st.error("Error: Target variable 'Is_Severe' could not be created. Check 'Accident_Severity' column processing.")
     st.stop()

X = df[prediction_features].copy()
y = df['Is_Severe']

for col in X.columns:
    if X[col].isnull().any():
        mode_val = X[col].mode()
        if not mode_val.empty:
             X[col].fillna(mode_val[0], inplace=True)
        else:
             st.warning(f"Could not determine mode for column '{col}'. Filling NaNs with 0.")
             X[col].fillna(0, inplace=True)

for col in X.select_dtypes(include='object').columns:
    st.warning(f"Column '{col}' is object type, attempting Label Encoding.")
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

cart_clf = None
if not X.empty and not y.empty:
    try:
        if y.nunique() > 1 and all(y.value_counts() >= 2):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        else:
            st.warning("Could not stratify data split due to insufficient samples in minority class. Performing regular split.")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        cart_clf = DecisionTreeClassifier(criterion='gini', random_state=42, max_depth=7, min_samples_leaf=10)
        cart_clf.fit(X_train, y_train)

    except ValueError as e:
        st.error(f"Error during model training or data splitting: {e}")
        st.error("Check if data types are correct and if there's enough data after cleaning.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during model setup: {e}")
        st.stop()

else:
    st.error("Error: Not enough data available to train the prediction model after cleaning (X or y is empty).")
    st.stop()


# --- Session state setup (Keep as before) ---
if 'page' not in st.session_state:
    st.session_state.page = "home"

# --- Page Navigation ---

def home_page():
    # Add custom CSS for styling and animation
    st.markdown("""
        <style>
        .main-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 90vh;
            text-align: center;
            animation: slideIn 1s ease-out forwards;
            opacity: 0;
        }

        @keyframes slideIn {
            0% {
                transform: translateX(-100%);
                opacity: 0;
            }
            100% {
                transform: translateX(0);
                opacity: 1;
            }
        }

        .custom-button {
            background-color: #ffffff20 !important;
            color: white !important;
            font-weight: bold !important;
            border: 2px solid white !important;
            border-radius: 12px !important;
            padding: 15px 30px !important;
            font-size: 18px !important;
            text-shadow: 1px 1px 2px black !important;
            transition: all 0.3s ease-in-out !important;
            margin: 15px 0 !important;
            width: 300px;
        }

        .custom-button:hover {
            background-color: #ffffff40 !important;
            transform: scale(1.05);
        }

        </style>
    """, unsafe_allow_html=True)

    # Main animated container
    st.markdown("""
        <div class="main-container">
            <h1>🚦 Road Accident Analysis & Insights</h1>
        </div>
    """, unsafe_allow_html=True)

    # Display one button per row, centered
    col_center = st.columns([1, 2, 1])[1]  # Centered layout
    with col_center:
        if st.button("🚨 Predict Accident Severity", key="severity_btn"):
            st.session_state.page = "predict_severity"
            st.rerun()
        if st.button("🗺️ Statistical Information", key="cluster_btn"):
            st.session_state.page = "cluster"
            st.rerun()
        if st.button("📊 Analyze Conditions", key="analyze_btn"):
            st.session_state.page = "analyze"
            st.rerun()

    # Apply custom class to buttons (after rendering)
    st.markdown("""
        <script>
        const buttons = parent.document.querySelectorAll('button');
        buttons.forEach(btn => {
            if (!btn.classList.contains('custom-button')) {
                btn.classList.add('custom-button');
            }
        });
        </script>
    """, unsafe_allow_html=True)


# --- MODIFIED HELPER FUNCTIONS ---
# --- MODIFIED HELPER FUNCTIONS ---
def create_selectbox(label, col_name, map_dict):
    """Creates a selectbox, excluding codes not in map_dict OR in CODES_TO_EXCLUDE."""
    if col_name not in df.columns:
         st.error(f"Column '{col_name}' not found for '{label}' selectbox.")
         return None

    unique_vals = sorted(df[col_name].unique())
    options = {} # {description: code}

    for k in unique_vals:
        # 1. Skip codes universally excluded (like -1, 8, 9, 3 etc.)
        if k in CODES_TO_EXCLUDE:
            continue
        # 2. Get description ONLY if the code is in the specific valid map
        desc = map_dict.get(k)
        if desc is not None:
            options[desc] = k
        # Codes present in data but NOT in the map_dict AND NOT in CODES_TO_EXCLUDE
        # will be silently ignored and not shown as "Unknown Code X".

    if not options:
         st.warning(f"No valid, selectable options found for {label} after filtering.")
         # Provide a default or placeholder if possible/necessary, or return None
         # Example: return list(map_dict.values())[0] # Return the first valid code? Risky.
         return None # Indicate failure to create valid options

    # Sort options alphabetically by description for user-friendliness
    sorted_options_keys = sorted(options.keys())

    # Add a default selection index if needed, e.g., index=0
    selected_desc = st.selectbox(label, options=sorted_options_keys) # index=0

    # Return the code associated with the selected description
    # Handle case where selectbox might be empty (though previous check should prevent)
    return options.get(selected_desc)


def create_multiselect(label, col_name, map_dict):
    """Creates a multiselect, excluding codes not in map_dict OR in CODES_TO_EXCLUDE."""
    if col_name not in df.columns:
         st.error(f"Column '{col_name}' not found for '{label}' multiselect.")
         return [] # Return empty list

    unique_vals = sorted(df[col_name].unique())
    options = {} # {description: code}

    for k in unique_vals:
         # 1. Skip codes universally excluded
        if k in CODES_TO_EXCLUDE:
            continue
        # 2. Get description ONLY if the code is in the specific valid map
        desc = map_dict.get(k)
        if desc is not None:
            options[desc] = k
        # Ignore codes not in map_dict and not in CODES_TO_EXCLUDE

    if not options:
         st.warning(f"No valid, selectable options found for {label} after filtering.")
         return [] # Return empty list

    # Sort options alphabetically by description
    sorted_options_keys = sorted(options.keys())

    selected_descs = st.multiselect(label, options=sorted_options_keys)
    # Return list of codes corresponding to selected descriptions
    return [options[desc] for desc in selected_descs]

# --- UPDATED PAGE FUNCTIONS (using modified helpers) ---
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
import joblib

# Example dummy training (replace X, y with your real data)
X = df[prediction_features]
y_severity = df['Accident_Severity']
y_casualties = df['Number_of_Casualties']
y_accidents = df['Accident_Index'].map(lambda x: 1)  # Dummy example

# Train models
cart_clf = DecisionTreeClassifier().fit(X, y_severity)
casualty_regressor = LinearRegression().fit(X, y_casualties)
accident_regressor = LinearRegression().fit(X, y_accidents)

# Save models
import os
os.makedirs("models", exist_ok=True)

joblib.dump(cart_clf, "models/cart_clf.pkl")
joblib.dump(casualty_regressor, "models/casualty_regressor.pkl")
joblib.dump(accident_regressor, "models/accident_regressor.pkl")

def predict_severity_page():
    st.title("🚨 Predict Accident Severity & Impact")
    st.write("Enter the conditions to predict accident severity, casualties, and number of accidents.")

    if cart_clf is None or casualty_regressor is None or accident_regressor is None:
        st.error("One or more prediction models are missing. Please ensure all models are loaded.")
        return

    input_values = {}

    # Input fields
    input_values['Weather_Conditions'] = create_selectbox("Select Weather Condition", 'Weather_Conditions', weather_map)
    input_values['Road_Type'] = create_selectbox("Select Road Type", 'Road_Type', road_type_map)

    # Speed Limit
    if 'Speed_limit' in df.columns:
        min_speed = int(df['Speed_limit'].min())
        max_speed = int(df['Speed_limit'].max())
        if pd.isna(min_speed) or pd.isna(max_speed) or np.isinf(min_speed) or np.isinf(max_speed):
            st.error("Invalid speed limits found.")
            input_values['Speed_limit'] = None
        else:
            median_speed = int(df['Speed_limit'].median())
            if not min_speed <= median_speed <= max_speed:
                median_speed = min_speed
            input_values['Speed_limit'] = st.slider("Select Speed Limit (km/h)", min_speed, max_speed, value=median_speed)
    else:
        st.error("'Speed_limit' column not found.")
        input_values['Speed_limit'] = None

    input_values['Light_Conditions'] = create_selectbox("Select Light Conditions", 'Light_Conditions', light_conditions_map)
    input_values['Road_Surface_Conditions'] = create_selectbox("Select Road Surface", 'Road_Surface_Conditions', road_surface_map)

    if 'Day_of_Week' in df.columns:
        input_values['Day_of_Week'] = create_selectbox("Select Day of Week", 'Day_of_Week', day_map)
    else:
        st.error("'Day_of_Week' column not found.")
        input_values['Day_of_Week'] = None

    if 'Urban_or_Rural_Area' in df.columns:
        input_values['Urban_or_Rural_Area'] = create_selectbox("Select Area Type", 'Urban_or_Rural_Area', urban_rural_map)
    else:
        st.error("'Urban_or_Rural_Area' column not found.")
        input_values['Urban_or_Rural_Area'] = None

    if st.button("🔮 Predict"):
        required_inputs = {k: v for k, v in input_values.items() if k != 'Speed_limit' or 'Speed_limit' in df.columns}

        if None in required_inputs.values():
            st.error("Please fill out all required fields.")
        else:
            try:
                input_list = [input_values.get(feature) for feature in prediction_features]
                if None in input_list:
                    st.error("Missing values detected before prediction.")
                    return

                input_data = pd.DataFrame([input_list], columns=prediction_features)

                # --- Prediction ---
                # Predict severity only
                severity_pred = cart_clf.predict(input_data)[0]
                severity_proba = cart_clf.predict_proba(input_data)[0]
                severity_text = severity_map.get(severity_pred, "Slight")
                if severity_pred == 1:
                    st.error(f"**Severity: {severity_text}**")
                else:
                    st.success(f"**Severity: {severity_text}**")
                total_casualties = df['Number_of_Casualties'].sum()
                total_accidents = df.shape[0]  # total number of rows = total accidents

                st.info(f"**Total Number of Casualties:** {total_casualties:,}")
                st.info(f"**Total Number of Accidents :** {total_accidents:,}")
                if severity_pred == 1:
                    st.warning("⚠️ **Drive carefully. Conditions may be hazardous.**")
                else:
                    st.success("✅ **Conditions look relatively safe. Stay alert and cautious in Driving.**")

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.error("Please verify your inputs and models.")

    if st.button("🔙 Back to Home"):
        st.session_state.page = "home"
        st.rerun()
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.cluster import DBSCAN

def perform_clustering(df, eps=0.01, min_samples=2):
    coords = df[['longitude', 'latitude']]
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    df['Cluster'] = db.labels_
    return df

def cluster_page():
    st.title("🧭 Statistical Information")
    # Apply DBSCAN clustering
    df_clustered = perform_clustering(df.copy())
    df_clusters  = df_clustered[df_clustered['Cluster'] != -1]

    if df_clusters.empty:
        st.warning("No clusters found. Try adjusting clustering parameters.")
        return

    # map numeric severity → descriptive label
    severity_map = {1: 'Fatal', 2: 'Serious', 3: 'Slight'}
    df_clusters['Severity_Label'] = df_clusters['Accident_Severity'].map(severity_map)

    # build distribution by cluster & labelled severity
    severity_dist = (
        df_clusters
         .groupby(['Cluster', 'Severity_Label'])
         .size()
         .unstack(fill_value=0)
    )

    st.subheader("🚦 Severity Distribution")

    # Line Chart for Severity Trends (with descriptive labels)
    fig, ax = plt.subplots(figsize=(10, 6))
    for label in severity_dist.columns:
        ax.plot(
            severity_dist.index,
            severity_dist[label],
            marker='o',
            label=label
        )
    ax.set_title("Severity Trend")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Accidents")
    ax.legend(title="Severity")
    st.pyplot(fig)

    # 🧊 Optional 3D Scatter Plot
    st.subheader("🧊 3D Visualization")
    fig2 = px.scatter_3d(
        df_clusters,
        x='Speed_limit',
        y='Number_of_Casualties',
        z='Accident_Severity',    # you could switch to 'Severity_Label' here too
        color='Cluster',
        title='3D Comparison by Key Attributes',
        labels={
            'Speed_limit': 'Speed Limit (km/h)',
            'Number_of_Casualties': 'Casualties',
            'Accident_Severity': 'Severity (1=Fatal,2=Serious,3=Slight)'
        }
    )
    st.plotly_chart(fig2)

    if st.button("🔙 Back to Home"):
        st.session_state.page = "home"
        st.rerun()

def analyze_page():
    st.title("📊 Analyze Accident Conditions")
    st.write("Explore accident statistics based on different conditions.")

    # --- Analysis Options ---
    # Use the VALID maps for selection descriptions
    analysis_feature_options_base = {
        'Weather': ('Weather_Conditions', weather_map),
        'Road Type': ('Road_Type', road_type_map),
        'Light Condition': ('Light_Conditions', light_conditions_map),
        'Road Surface': ('Road_Surface_Conditions', road_surface_map),
        'Day of Week': ('Day_of_Week', day_map),
        'Area Type': ('Urban_or_Rural_Area', urban_rural_map)
    }
    # Filter options based on columns actually present in df
    analysis_feature_options = {
        k: v for k, v in analysis_feature_options_base.items() if v[0] in df.columns
    }

    if not analysis_feature_options:
         st.error("No suitable columns found for analysis in the dataset.")
         return

    selected_analysis_desc = st.selectbox("Select Condition to Analyze", options=list(analysis_feature_options.keys()))
    selected_analysis_col, current_map = analysis_feature_options[selected_analysis_desc]

    # --- Perform Analysis ---
    st.subheader(f"Accident Counts by {selected_analysis_desc}")

    # Group and count ALL data initially
    analysis_counts_raw = df[selected_analysis_col].value_counts().reset_index()
    analysis_counts_raw.columns = [selected_analysis_col, 'Count']

    # Map codes to descriptions using the VALID map, KEEPING ONLY VALID codes for display
    analysis_counts = analysis_counts_raw.copy()
    analysis_counts[selected_analysis_desc] = analysis_counts[selected_analysis_col].map(current_map) # Use map directly
    # Filter out rows where mapping failed (i.e., code wasn't in the valid map)
    analysis_counts = analysis_counts.dropna(subset=[selected_analysis_desc])
    # Also filter out codes that are universally excluded
    analysis_counts = analysis_counts[~analysis_counts[selected_analysis_col].isin(CODES_TO_EXCLUDE)]


    if analysis_counts.empty:
        st.warning(f"No accidents found with valid '{selected_analysis_desc}' codes after filtering.")
    else:
        display_col = selected_analysis_desc
        # Sort by count
        analysis_counts = analysis_counts.sort_values(by='Count', ascending=False)

        # --- Visualization (Keep as before, but uses filtered analysis_counts) ---
        plt.style.use('default')
        TEXT_COLOR = 'black'# Smaller figure size
        fig, ax = plt.subplots(figsize=(4, 2))  
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        plot_data = analysis_counts.head(15)
        sns.barplot(data=plot_data,y=display_col,x='Count',hue=display_col,palette='viridis',legend=False,ax=ax,orient='h')
        ax.set_title(f"Top Accident Counts by {selected_analysis_desc}", color=TEXT_COLOR, fontsize=7)
        ax.set_xlabel("Number of Accidents", color=TEXT_COLOR, fontsize=5)
        ax.set_ylabel(selected_analysis_desc, color=TEXT_COLOR, fontsize=5)
        ax.tick_params(axis='x', colors=TEXT_COLOR, labelsize=4)
        ax.tick_params(axis='y', colors=TEXT_COLOR, labelsize=4)
        plt.tight_layout()
        st.pyplot(fig)

        # Optionally show severity distribution, also filtered
        if 'Is_Severe' in df.columns:
            st.subheader(f"Severity Distribution by {selected_analysis_desc}")
            try:
                # Group by the original column code, but only for codes present in the filtered analysis_counts
                valid_codes_for_analysis = analysis_counts[selected_analysis_col].unique()
                df_filtered_for_sev = df[df[selected_analysis_col].isin(valid_codes_for_analysis)]

                if not df_filtered_for_sev.empty:
                    severity_dist = df_filtered_for_sev.groupby(selected_analysis_col)['Is_Severe'].value_counts(normalize=True).unstack().fillna(0)
                    col_map = {}
                    if 0 in severity_dist.columns: col_map[0] = 'Slight (%)'
                    if 1 in severity_dist.columns: col_map[1] = 'Serious/Fatal (%)'
                    severity_dist.rename(columns=col_map, inplace=True)
                    if 'Slight (%)' not in severity_dist.columns: severity_dist['Slight (%)'] = 0
                    if 'Serious/Fatal (%)' not in severity_dist.columns: severity_dist['Serious/Fatal (%)'] = 0
                    severity_dist *= 100
                    # Map index codes to descriptions using the valid map
                    severity_dist.index = severity_dist.index.map(current_map)
                    # Remove any rows where mapping might have failed (shouldn't happen here)
                    severity_dist.dropna(inplace=True)

                    if not severity_dist.empty:
                         display_cols_sev = ['Slight (%)', 'Serious/Fatal (%)']
                         st.dataframe(severity_dist[display_cols_sev].head(10), use_container_width=True)
                    else:
                        st.warning("No severity data to display for the selected valid conditions.")
                else:
                    st.warning("No accidents found with valid conditions to calculate severity distribution.")

            except Exception as e:
                 st.warning(f"Could not calculate or display severity distribution: {e}")

    if st.button("🔙 Back to Home"):
        st.session_state.page = "home"
        st.rerun()


# --- Main App Router (Keep as before) ---
if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "predict_severity":
    predict_severity_page()
elif st.session_state.page=="cluster":
    cluster_page()
elif st.session_state.page == "analyze":
    analyze_page()
else:
    st.session_state.page = "home"
    st.rerun()