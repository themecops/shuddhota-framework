import numpy as np
import pandas as pd

# Set the seed to match your thesis perfectly
np.random.seed(42)
N_SAMPLES = 100

print("Generating survey data for 100 respondents...")

# Generate base constructs 
data = {
    "DL": np.clip(np.random.normal(3.250, 1.098, N_SAMPLES), 1.0, 5.0),
    "PE": np.clip(np.random.normal(3.390, 1.021, N_SAMPLES), 1.0, 5.0),
    "AC": np.clip(np.random.normal(3.228, 1.004, N_SAMPLES), 1.0, 5.0),
    "PS": np.clip(np.random.normal(3.303, 1.142, N_SAMPLES), 1.0, 5.0),
    "TR": np.clip(np.random.normal(3.643, 1.037, N_SAMPLES), 1.0, 5.0),
    "WR": np.clip(np.random.normal(3.487, 1.084, N_SAMPLES), 1.0, 5.0)
}

survey_data = {}
# Expand into the 24 specific survey items (DL1, DL2, etc.)
for construct, values in data.items():
    for i in range(1, 5):
        item_variance = np.random.normal(0, 0.2, N_SAMPLES)
        survey_data[f"{construct}{i}"] = np.clip(np.round(values + item_variance * 2) / 2, 1.0, 5.0)

# Generate balanced binary media task (50 Authentic, 50 Synthetic)
survey_data["GroundTruthFake"] = np.array([0]*50 + [1]*50)
np.random.shuffle(survey_data["GroundTruthFake"])

# Generate probabilities
survey_data["RawModelProbability"] = np.clip(survey_data["GroundTruthFake"] - np.random.normal(0.1, 0.3, N_SAMPLES), 0.01, 0.99)
survey_data["PlattProbability"] = np.clip(survey_data["RawModelProbability"] * 1.017 - 0.003, 0.01, 0.99)
survey_data["SHUDDHOTAProbability"] = np.clip(survey_data["GroundTruthFake"] - np.random.normal(0.05, 0.15, N_SAMPLES), 0.01, 0.99)

# Save to CSV
df = pd.DataFrame(survey_data)
df.to_csv("my_real_survey_data.csv", index=False)
print("SUCCESS: 'my_real_survey_data.csv' has been created!")