# %% [markdown]
# # Medical Appointment No-Show Prediction
# Goal: predict which patients will miss an appointment so a clinic can
# send reminders / calls to the riskiest ones first.
#
# How to use in VS Code: install the "Python" + "Jupyter" extensions, then click
# "Run Cell" above each "# %%" block (or "Run All"). Run the cells in order.

# %% Step 1: Imports and load data
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

# Put the Kaggle CSV in the same folder as this file.
DATA_PATH = os.environ.get("DATA_PATH", "KaggleV2-May-2016.csv")
os.makedirs("outputs", exist_ok=True)
RANDOM_STATE = 42

df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print(df.head())
print(df.dtypes)

# %% Step 2: Clean the data
# Rename columns (the raw file has typos like 'Hipertension' and 'Handcap').
df = df.rename(columns={
    "PatientId": "patient_id",
    "AppointmentID": "appointment_id",
    "Gender": "gender",
    "ScheduledDay": "scheduled_day",
    "AppointmentDay": "appointment_day",
    "Age": "age",
    "Neighbourhood": "neighbourhood",
    "Scholarship": "scholarship",
    "Hipertension": "hypertension",
    "Hypertension": "hypertension",
    "Diabetes": "diabetes",
    "Alcoholism": "alcoholism",
    "Handcap": "handicap",
    "Handicap": "handicap",
    "SMS_received": "sms_received",
    "No-show": "no_show",
})

# Dates -> datetime (drop timezone info)
df["scheduled_day"] = pd.to_datetime(df["scheduled_day"]).dt.tz_localize(None)
df["appointment_day"] = pd.to_datetime(df["appointment_day"]).dt.tz_localize(None)

# Target: 1 = patient did NOT show up
df["no_show"] = (df["no_show"] == "Yes").astype(int)

# Remove impossible ages
before = len(df)
df = df[(df["age"] >= 0) & (df["age"] <= 110)]
print(f"Removed {before - len(df)} rows with invalid age")

# Handicap has levels 0-4; make it a simple yes/no
df["handicap"] = (df["handicap"] > 0).astype(int)
df["patient_id"] = df["patient_id"].astype("int64")

print("Overall no-show rate:", round(df["no_show"].mean() * 100, 1), "%")

# %% Step 3: Feature engineering (the interesting part)
# 3a. Waiting time between booking and the appointment
df["wait_days"] = (
    df["appointment_day"].dt.normalize() - df["scheduled_day"].dt.normalize()
).dt.days

# Appointment before booking is a data error -> drop
before = len(df)
df = df[df["wait_days"] >= 0]
print(f"Removed {before - len(df)} rows with negative waiting time")

# 3b. Day of week of the appointment (0 = Monday)
df["weekday"] = df["appointment_day"].dt.dayofweek

# 3c. Patient history: number of earlier appointments / earlier no-shows.
# Sort by time so that we only use PAST information (avoids data leakage).
df = df.sort_values(["patient_id", "appointment_day", "scheduled_day"]).reset_index(drop=True)
grp = df.groupby("patient_id")
df["prior_appointments"] = grp.cumcount()
df["prior_no_shows"] = grp["no_show"].cumsum() - df["no_show"]  # excludes current row

# 3d. Simple encodings
df["gender_male"] = (df["gender"] == "M").astype(int)
df["age_group"] = pd.cut(
    df["age"], bins=[-1, 12, 25, 40, 60, 120],
    labels=["child", "young_adult", "adult", "middle_age", "senior"],
)

print(df[["age", "wait_days", "weekday", "prior_appointments", "prior_no_shows", "no_show"]].describe())

# %% Step 4: Exploratory data analysis (EDA)
# Write one insight per chart - you will explain these in the interview.
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# 4a. No-show rate by waiting time
wait_bucket = pd.cut(df["wait_days"], bins=[-1, 0, 3, 7, 14, 30, 400],
                     labels=["same day", "1-3", "4-7", "8-14", "15-30", "30+"])
(df.groupby(wait_bucket, observed=True)["no_show"].mean() * 100).plot(
    kind="bar", ax=axes[0, 0], color="steelblue")
axes[0, 0].set_title("No-show rate (%) by waiting time (days)")
axes[0, 0].set_ylabel("%")

# 4b. By age group
(df.groupby("age_group", observed=True)["no_show"].mean() * 100).plot(
    kind="bar", ax=axes[0, 1], color="darkorange")
axes[0, 1].set_title("No-show rate (%) by age group")

# 4c. By SMS reminder
(df.groupby("sms_received")["no_show"].mean() * 100).plot(
    kind="bar", ax=axes[1, 0], color="seagreen")
axes[1, 0].set_title("No-show rate (%) by SMS received (0 = no, 1 = yes)")

# 4d. By number of previous no-shows
prior_bucket = df["prior_no_shows"].clip(upper=3)
(df.groupby(prior_bucket)["no_show"].mean() * 100).plot(
    kind="bar", ax=axes[1, 1], color="firebrick")
axes[1, 1].set_title("No-show rate (%) by previous no-shows (3 = 3 or more)")

plt.tight_layout()
plt.savefig("outputs/eda.png", dpi=150)
plt.show()

# %% Step 5: Train / test split
FEATURES = [
    "age", "wait_days", "weekday", "sms_received", "scholarship",
    "hypertension", "diabetes", "alcoholism", "handicap",
    "gender_male", "prior_appointments", "prior_no_shows",
]
X = df[FEATURES]
y = df["no_show"]

# stratify keeps the ~20% no-show ratio the same in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print("Train:", X_train.shape, "Test:", X_test.shape)
print("No-show rate train/test:", round(y_train.mean(), 3), round(y_test.mean(), 3))

# %% Step 6: Train models (baseline vs stronger model)
# class_weight='balanced' handles the 80/20 imbalance.
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=20,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ),
}

probas = {}
summary = []
for name, model in models.items():
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    probas[name] = proba
    pred = (proba >= 0.5).astype(int)

    print("=" * 60)
    print(name)
    print(classification_report(y_test, pred, target_names=["showed up", "no-show"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))

    summary.append({
        "model": name,
        "roc_auc": round(roc_auc_score(y_test, proba), 3),
        "pr_auc": round(average_precision_score(y_test, proba), 3),
    })

summary_df = pd.DataFrame(summary)
print(summary_df)
summary_df.to_csv("outputs/model_comparison.csv", index=False)
# NOTE: PR-AUC is compared with the baseline of 0.20 (the no-show rate),
# not with 0.5 - that is why accuracy alone is misleading here.

# %% Step 7: Business layer - who should the clinic call?
best_name = summary_df.sort_values("pr_auc", ascending=False).iloc[0]["model"]
best_proba = probas[best_name]
print("Using:", best_name)

test = pd.DataFrame({"y": y_test.values, "p": best_proba}).sort_values("p", ascending=False)
base_rate = test["y"].mean()
total_no_shows = test["y"].sum()

rows = []
for frac in [0.1, 0.2, 0.3, 0.5]:
    k = int(len(test) * frac)
    caught = test["y"].iloc[:k].sum()
    rows.append({
        "contact_top_%": int(frac * 100),
        "no_shows_caught_%": round(100 * caught / total_no_shows, 1),
        "precision_%": round(100 * caught / k, 1),
        "lift_vs_random": round((caught / k) / base_rate, 2),
    })
business_df = pd.DataFrame(rows)
print(business_df.to_string(index=False))
business_df.to_csv("outputs/business_impact.csv", index=False)

# Threshold trade-off table
thr_rows = []
for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
    pred = (best_proba >= t).astype(int)
    tp = ((pred == 1) & (y_test.values == 1)).sum()
    precision = tp / max(pred.sum(), 1)
    recall = tp / total_no_shows
    thr_rows.append({"threshold": t, "precision": round(precision, 3), "recall": round(recall, 3),
                     "patients_flagged": int(pred.sum())})
print(pd.DataFrame(thr_rows).to_string(index=False))
# Interpretation: a lower threshold catches more no-shows (recall up) but
# flags more patients (precision down). The clinic picks it based on how
# costly a reminder call is vs how costly an empty slot is.

# %% Step 8: What drives no-shows? (feature importance)
rf = models["Random Forest"]
importance = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
importance.plot(kind="barh", figsize=(8, 6), color="steelblue")
plt.title("Random Forest feature importance")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=150)
plt.show()
print(importance.sort_values(ascending=False))

# %% Step 9: Save predictions (and a small reminder-message helper)
results = X_test.copy()
results["actual_no_show"] = y_test.values
results["risk_score"] = best_proba
results["risk_tier"] = pd.cut(results["risk_score"], bins=[-0.01, 0.4, 0.6, 1.0],
                              labels=["low", "medium", "high"])
results.sort_values("risk_score", ascending=False).to_csv("outputs/test_predictions.csv", index=False)
print(results["risk_tier"].value_counts())


def reminder_message(tier: str) -> str:
    """Simple rule-based message. Idea for extension: let an LLM write a
    friendlier, personalised version of this text (keep patient data private!)."""
    if tier == "high":
        return "Your appointment is coming up. Please call us to confirm, or reschedule if you cannot attend."
    if tier == "medium":
        return "Reminder: you have an appointment soon. Reply YES to confirm."
    return "Reminder: you have an appointment soon."


print(reminder_message("high"))