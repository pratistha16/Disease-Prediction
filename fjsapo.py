import pandas as pd
import numpy as np
import joblib
import os

# Local model file paths for simpler pre-trained models
model_paths = {
    "Bone Marrow": r"E:\sem 3\MSCP\models\bonemarror.pkl",
    "Depression": r"E:\sem 3\MSCP\models\Depression.pkl",
    "Wilson Disease": r"E:\sem 3\MSCP\models\willamdiease.pkl"
}

disease_configs = {
    "Bone Disease": "bone_disease",
    "Bone Marrow": "bone_marrow",
    "Depression": "depression",
    "HIV": "hiv",
    "Skin Cancer": "skin_cancer",
    "Wilson Disease": "wilson_disease"
}

def fitness_function(model, x):
    prob = model.predict_proba([x])[0]
    return prob[1]  # Assuming binary classification, positive class confidence

def run_fjapso_optimization(model, x, weights, n_particles=5, n_iterations=10):
    dim = len(x)
    particles = np.random.rand(n_particles, dim)
    velocities = np.zeros((n_particles, dim))

    personal_best_positions = np.copy(particles)
    personal_best_scores = np.array([fitness_function(model, x * p * weights) for p in particles])
    global_best_position = personal_best_positions[np.argmax(personal_best_scores)]

    for _ in range(n_iterations):
        for i in range(n_particles):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            velocities[i] = (
                0.5 * velocities[i]
                + 0.8 * r1 * (personal_best_positions[i] - particles[i])
                + 0.9 * r2 * (global_best_position - particles[i])
            )
            particles[i] += velocities[i]
            particles[i] = np.clip(particles[i], 0, 1)

            score = fitness_function(model, x * particles[i] * weights)
            if score > personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = particles[i]

        global_best_position = personal_best_positions[np.argmax(personal_best_scores)]

    best_score = np.max(personal_best_scores)
    prediction = 1 if best_score >= 0.5 else 0
    return prediction

def predict_diseases(patient_data: dict):
    result = {}

    for disease, prefix in disease_configs.items():
        try:
            # Use local pre-trained basic models
            if disease in model_paths:
                model = joblib.load(model_paths[disease])
                input_df = pd.DataFrame([patient_data])
                prediction = model.predict(input_df)
                result[disease] = prediction[0]
                continue

            # Use FJAPSO-based model prediction
            model = joblib.load(f"{prefix}_model.joblib")
            scaler = joblib.load(f"{prefix}_scaler.joblib")
            weights = joblib.load(f"{prefix}_weights.npy")
            encoders = joblib.load(f"{prefix}_encoders.joblib")

            input_df = pd.DataFrame([patient_data])

            for col, le in encoders.items():
                input_df[col] = le.transform([str(input_df[col].values[0])])

            x_scaled = scaler.transform(input_df)[0]

            # FJAPSO Optimization to predict label
            prediction = run_fjapso_optimization(model, x_scaled, weights)

            # Decode target label
            target_encoder_path = f"{prefix}_target_encoder.joblib"
            if os.path.exists(target_encoder_path):
                le_target = joblib.load(target_encoder_path)
                prediction = le_target.inverse_transform([prediction])[0]

            result[disease] = prediction
        except Exception as e:
            result[disease] = f"Error: {str(e)}"

    return result
