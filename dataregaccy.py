import pandas as pd
import numpy as np
import random
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor, XGBClassifier

# ----------- Load Dataset -----------

DATA_PATH = r"D:\intership\Phy. Men. Sen. Health Ai\fake_worker_health_data.csv"

df = pd.read_csv(DATA_PATH)

# ----------- Prepare Target Scores -----------

readiness_map_score = {"Poor": 1, "Fair": 2, "Good": 3, "Excellent": 4}
df['Physical_Score'] = df['Overall Physical Readiness'].map(readiness_map_score)

df['Sentiment_Score'] = (df['Sleep Hours Last Night'] / 12) - (df['Alcohol Consumed in Last 24h'] == 'Yes')*0.3
df['Sentiment_Score'] = df['Sentiment_Score'].clip(0,1)

def mental_health_label(row):
    if row['Physical_Score'] >= 3 and row['Sentiment_Score'] > 0.7:
        return 0  # Stable
    elif row['Physical_Score'] >= 2 and row['Sentiment_Score'] > 0.4:
        return 1  # Watch
    else:
        return 2  # Needs Help

df['Mental_Health'] = df.apply(mental_health_label, axis=1)

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

X = df[features]
y_physical = df['Physical_Score']
y_mental = df['Mental_Health']

num_feats = ['Age', 'Height (cm)', 'Weight (kg)', 'BMI', 'Sleep Hours Last Night',
             'Working Hours Today', 'Last Medical Checkup (months ago)', 'Body Temperature (°C)']

scaler = StandardScaler()
X[num_feats] = scaler.fit_transform(X[num_feats])

# ----------- Train Models -----------

def tune_and_train_xgb_regressor(X, y):
    xgb = XGBRegressor(objective='reg:squarederror', random_state=42)
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0]
    }
    rnd_search = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=10,
                                    scoring='neg_root_mean_squared_error', cv=3, random_state=42, n_jobs=-1)
    rnd_search.fit(X, y)
    print(f"Best params for regressor: {rnd_search.best_params_}")
    return rnd_search.best_estimator_

def tune_and_train_xgb_classifier(X, y):
    xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.6, 0.8, 1.0]
    }
    rnd_search = RandomizedSearchCV(xgb, param_distributions=param_dist, n_iter=10,
                                    scoring='accuracy', cv=3, random_state=42, n_jobs=-1)
    rnd_search.fit(X, y)
    print(f"Best params for classifier: {rnd_search.best_params_}")
    return rnd_search.best_estimator_

print("Training Physical Health Model...")
model_physical = tune_and_train_xgb_regressor(X, y_physical)

print("Training Mental Health Classifier...")
model_mental = tune_and_train_xgb_classifier(X, y_mental)

# ----------- Load questions from mental.txt -----------

def load_questions(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        return lines
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return []

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
    input_df[num_feats] = scaler.transform(input_df[num_feats])

    return input_df

# ----------- Function to ask mental health questions -----------

def ask_mental_health_questions(filename):
    questions = load_questions(filename)
    if not questions:
        print(f"No questions found in {filename}. Skipping mental health questionnaire.")
        return 0

    print(f"\nPlease answer the following mental health questions from {filename}:")
    print("Rate each question on a scale from 1 (lowest) to 5 (highest).")

    total_score = 0
    for i, question in enumerate(questions, 1):
        while True:
            try:
                ans = int(input(f"{i}. {question}\nYour score (1-5): "))
                if 1 <= ans <= 5:
                    total_score += ans
                    break
                else:
                    print("Score must be between 1 and 5.")
            except:
                print("Invalid input. Please enter an integer between 1 and 5.")

    max_score = len(questions) * 5
    percentage_score = (total_score / max_score) * 100
    return percentage_score

# ----------- Main Program -----------

if __name__ == "__main__":
    print("=== Worker Health Prediction System ===")

    user_features_df = get_user_input()

    pred_physical = model_physical.predict(user_features_df)[0]
    physical_percent = ((pred_physical - 1) / 3) * 100  # Rescale 1-4 to 0-100%

    mental_score = ask_mental_health_questions('mental.txt')

    print("\n--- Individual Scores ---")
    print(f"Physical Health Score: {physical_percent:.2f}%")
    print(f"Mental Health Score (from questionnaire): {mental_score:.2f}%")

    # Overall health score: 50% physical + 50% mental
    overall_score = (physical_percent * 0.5) + (mental_score * 0.5)

    # Mental health interpretation
    if mental_score >= 80:
        mental_health_desc = "Excellent (Stable)"
    elif mental_score >= 60:
        mental_health_desc = "Good (Watch)"
    elif mental_score >= 40:
        mental_health_desc = "Fair (Needs Attention)"
    else:
        mental_health_desc = "Poor (Needs Immediate Help)"

    print("\n=== Overall Health Report ===")
    print(f"Overall Health Score (weighted): {overall_score:.2f}%")
    print(f"Mental Health Assessment: {mental_health_desc}")
