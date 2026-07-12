import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report


# Load loan book data
df = pd.read_csv("loan_data.csv")


# Create additional risk-related features
df["debt_to_income"] = (
    df["total_debt_outstanding"] /
    df["income"]
)

df["loan_to_income"] = (
    df["loan_amt_outstanding"] /
    df["income"]
)


# Define model inputs and target variable
features = [
    "credit_lines_outstanding",
    "loan_amt_outstanding",
    "total_debt_outstanding",
    "income",
    "years_employed",
    "fico_score",
    "debt_to_income",
    "loan_to_income"
]

X = df[features]
y = df["default"]


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Logistic regression model
# Scaling is required because the features have different ranges
log_reg_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(random_state=42))
])


# Random forest model
# Tree models do not require feature scaling
random_forest_model = Pipeline([
    ("model", RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42
    ))
])


# Train both models
log_reg_model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)


# Compare model performance using ROC-AUC
lr_predictions = log_reg_model.predict_proba(X_test)[:, 1]
rf_predictions = random_forest_model.predict_proba(X_test)[:, 1]

lr_auc = roc_auc_score(y_test, lr_predictions)
rf_auc = roc_auc_score(y_test, rf_predictions)

print(f"Logistic Regression ROC-AUC: {lr_auc:.4f}")
print(f"Random Forest ROC-AUC: {rf_auc:.4f}")


# Select logistic regression due to interpretability requirements
# which are important in credit risk modelling
final_model = log_reg_model

final_model.fit(X, y)


def predict_expected_loss(customer_data, model=final_model, recovery_rate=0.10):
    """
    Predicts probability of default and expected loss for a borrower.
    """

    # Check that all raw borrower inputs are provided
    required_features = [
        "credit_lines_outstanding",
        "loan_amt_outstanding",
        "total_debt_outstanding",
        "income",
        "years_employed",
        "fico_score"
    ]

    missing_features = set(required_features) - set(customer_data.keys())

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )


    # Calculate additional features for the borrower
    customer_data["debt_to_income"] = (
        customer_data["total_debt_outstanding"] /
        customer_data["income"]
    )

    customer_data["loan_to_income"] = (
        customer_data["loan_amt_outstanding"] /
        customer_data["income"]
    )


    # Convert borrower details into model input format
    input_data = pd.DataFrame(
        [customer_data]
    )[features]


    # Predict probability of default
    probability_of_default = (
        model.predict_proba(input_data)[0][1]
    )


    # Calculate expected loss
    exposure_at_default = (
        customer_data["loan_amt_outstanding"]
    )

    loss_given_default = 1 - recovery_rate

    expected_loss = (
        probability_of_default
        * loss_given_default
        * exposure_at_default
    )


    return {
        "Probability of Default": probability_of_default,
        "Exposure at Default": exposure_at_default,
        "Loss Given Default": loss_given_default,
        "Expected Loss": expected_loss
    }


# Example borrower
test_borrower = {
    "credit_lines_outstanding": 4,
    "loan_amt_outstanding": 5000,
    "total_debt_outstanding": 9000,
    "income": 30000,
    "years_employed": 1,
    "fico_score": 580
}


# Generate expected loss estimate
result = predict_expected_loss(test_borrower)


print("\nLoan Risk Estimate")
print("------------------")
print(f"Probability of Default: {result['Probability of Default']:.2%}")
print(f"Exposure at Default: ${result['Exposure at Default']:,.2f}")
print(f"Loss Given Default: {result['Loss Given Default']:.2%}")
print(f"Expected Loss: ${result['Expected Loss']:,.2f}")
