import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor

# ----------- Load Dataset -----------

DATA_PATH = r"D:\intership\Phy. Men. Sen. Health Ai\fake_worker_health_data.csv"

df = pd.read_csv(DATA_PATH)

# ----------- Prepare Target Scores -----------

readiness_map_score = {"Poor": 1, "Fair": 2, "Good": 3, "Excellent": 4}
df['Physical_Score'] = df['Overall Physical Readiness'].map(readiness_map_score)

# ----------- Feature Engineering -----------

df['BMI'] = df['Weight (kg)'] / ((df['Height (cm)'] / 100) ** 2)

categorical_cols = [
    'Gender', 'Food Intake Quality', 'Alcohol Consumed in Last 24h', 'Had Rest Breaks',
    'Wearing PPE', 'Has Chronic Conditions', 'Any Injury in Last 7 Days',
    'Hydration Level', 'Overall Physical Readiness'
]

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

features = [
    'Age', 'Gender', 'Height (cm)', 'Weight (kg)', 'BMI', 'Sleep Hours Last Night',
    'Food Intake Quality', 'Alcohol Consumed in Last 24h', 'Working Hours Today',
    'Had Rest Breaks', 'Last Medical Checkup (months ago)', 'Wearing PPE',
    'Has Chronic Conditions', 'Any Injury in Last 7 Days',
    'Hydration Level', 'Body Temperature (°C)', 'Overall Physical Readiness'
]

X = df[features].copy()  # Create a copy to avoid pandas warning
y_physical = df['Physical_Score']

num_feats = ['Age', 'Height (cm)', 'Weight (kg)', 'BMI', 'Sleep Hours Last Night',
             'Working Hours Today', 'Last Medical Checkup (months ago)', 'Body Temperature (°C)']

scaler = StandardScaler()
X.loc[:, num_feats] = scaler.fit_transform(X[num_feats])  # Use .loc to avoid warning

# ----------- Train Model -----------

def tune_and_train_xgb_regressor(X, y):
    xgb = XGBRegressor(objective='reg:squarederror', random_state=42)
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0]
    }
    # Use n_jobs=1 to avoid memory issues and reduce n_iter
    rnd_search = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=5,
                                    scoring='neg_root_mean_squared_error', cv=3, random_state=42, n_jobs=1)
    rnd_search.fit(X, y)
    print(f"Best params for regressor: {rnd_search.best_params_}")
    return rnd_search.best_estimator_

print("Training Physical Health Model...")
model_physical = tune_and_train_xgb_regressor(X, y_physical)

# ----------- Save Models and Preprocessors -----------

# Save the trained model in multiple formats
print("Saving model files...")

# Save as pickle file
with open('physical_health_model.pkl', 'wb') as f:
    pickle.dump(model_physical, f)

# Save as joblib file (more efficient for sklearn models)
joblib.dump(model_physical, 'physical_health_model.joblib')

# Save the scaler
joblib.dump(scaler, 'feature_scaler.joblib')

# Save label encoders
joblib.dump(label_encoders, 'label_encoders.joblib')

# Save feature names and numerical features list for future use
model_metadata = {
    'features': features,
    'numerical_features': num_feats,
    'categorical_columns': categorical_cols
}
joblib.dump(model_metadata, 'model_metadata.joblib')

print("Model files saved successfully!")
print("Files created:")
print("- physical_health_model.pkl")
print("- physical_health_model.joblib")
print("- feature_scaler.joblib")
print("- label_encoders.joblib")
print("- model_metadata.joblib")

# ----------- Interactive Input Function -----------

def get_user_input():
    print("\nAnswer the following worker parameters:")

    def ask_int(prompt, min_val=None, max_val=None):
        while True:
            try:
                val = int(input(prompt))
                if min_val is not None and val < min_val:
                    print(f"Value should be >= {min_val}")
                    continue
                if max_val is not None and val > max_val:
                    print(f"Value should be <= {max_val}")
                    continue
                return val
            except:
                print("Invalid input. Please enter an integer.")

    def ask_choice(prompt, options):
        options_str = "/".join(str(opt) for opt in options)
        while True:
            val = input(f"{prompt} ({options_str}): ")
            if val in options:
                return val
            print(f"Invalid input. Choose from {options_str}")

    Age = ask_int("Age (18-65): ", 18, 65)
    Gender = ask_choice("Gender", label_encoders['Gender'].classes_)
    Height = ask_int("Height (cm, 160-200): ", 160, 200)
    Weight = ask_int("Weight (kg, 50-120): ", 50, 120)
    Sleep = ask_int("Sleep Hours Last Night (0-12): ", 0, 12)
    Food_Quality = ask_choice("Food Intake Quality", label_encoders['Food Intake Quality'].classes_)
    Alcohol = ask_choice("Alcohol Consumed in Last 24h", label_encoders['Alcohol Consumed in Last 24h'].classes_)
    Working_Hours = ask_int("Working Hours Today (0-16): ", 0, 16)
    Rest_Breaks = ask_choice("Had Rest Breaks", label_encoders['Had Rest Breaks'].classes_)
    Last_Checkup = ask_int("Last Medical Checkup (months ago, 0-60): ", 0, 60)
    Wearing_PPE = ask_choice("Wearing PPE", label_encoders['Wearing PPE'].classes_)
    Chronic_Cond = ask_choice("Has Chronic Conditions", label_encoders['Has Chronic Conditions'].classes_)
    Injury = ask_choice("Any Injury in Last 7 Days", label_encoders['Any Injury in Last 7 Days'].classes_)
    Hydration = ask_choice("Hydration Level", label_encoders['Hydration Level'].classes_)

    while True:
        Body_Temp_raw = input("Body Temperature (°C, 36.5-38.5, press Enter if unknown): ").strip()
        if Body_Temp_raw == '':
            Body_Temp = 37.5
            break
        try:
            Body_Temp = float(Body_Temp_raw)
            if 36.5 <= Body_Temp <= 38.5:
                break
            else:
                print("Body temperature must be between 36.5 and 38.5")
        except:
            print("Invalid input")

    Readiness = ask_choice("Overall Physical Readiness", label_encoders['Overall Physical Readiness'].classes_)

    BMI = Weight / ((Height / 100) ** 2)

    input_dict = {
        'Age': Age,
        'Gender': label_encoders['Gender'].transform([Gender])[0],
        'Height (cm)': Height,
        'Weight (kg)': Weight,
        'BMI': BMI,
        'Sleep Hours Last Night': Sleep,
        'Food Intake Quality': label_encoders['Food Intake Quality'].transform([Food_Quality])[0],
        'Alcohol Consumed in Last 24h': label_encoders['Alcohol Consumed in Last 24h'].transform([Alcohol])[0],
        'Working Hours Today': Working_Hours,
        'Had Rest Breaks': label_encoders['Had Rest Breaks'].transform([Rest_Breaks])[0],
        'Last Medical Checkup (months ago)': Last_Checkup,
        'Wearing PPE': label_encoders['Wearing PPE'].transform([Wearing_PPE])[0],
        'Has Chronic Conditions': label_encoders['Has Chronic Conditions'].transform([Chronic_Cond])[0],
        'Any Injury in Last 7 Days': label_encoders['Any Injury in Last 7 Days'].transform([Injury])[0],
        'Hydration Level': label_encoders['Hydration Level'].transform([Hydration])[0],
        'Body Temperature (°C)': Body_Temp,
        'Overall Physical Readiness': label_encoders['Overall Physical Readiness'].transform([Readiness])[0]
    }

    input_df = pd.DataFrame([input_dict])
    input_df.loc[:, num_feats] = scaler.transform(input_df[num_feats])  # Use .loc to avoid warning

    return input_df

# ----------- Main Program -----------

if __name__ == "__main__":
    print("=== Worker Physical Health Prediction System ===")

    user_features_df = get_user_input()

    pred_physical = model_physical.predict(user_features_df)[0]
    physical_percent = ((pred_physical - 1) / 3) * 100  # Rescale 1-4 to 0-100%

    print("\n=== Physical Health Report ===")
    print(f"Physical Health Score: {physical_percent:.2f}%")
    
    # Physical health interpretation
    if physical_percent >= 85:
        physical_desc = "Excellent"
    elif physical_percent >= 70:
        physical_desc = "Good"
    elif physical_percent >= 50:
        physical_desc = "Fair"
    else:
        physical_desc = "Poor"
    
    print(f"Physical Health Assessment: {physical_desc}")