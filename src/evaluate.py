import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score

def evaluate_model():
    model_path = "models/churn_model.pkl"
    test_data_path = "data/test_data.csv"

    try:
        model = joblib.load(model_path)
        test_df = pd.read_csv(test_data_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    X_test = test_df.drop(columns=['Churn'])
    Y_test = test_df['Churn']

    Y_pred = model.predict(X_test)
    Y_prob = model.predict_proba(X_test)[:, 1]

    print(classification_report(Y_test, Y_pred))

    print(f"ROC-AUC Score: {roc_auc_score(Y_test, Y_prob): .4f}")
    print(f"PR-AUC Score: {average_precision_score(Y_test, Y_prob): .4f}")

    cm = confusion_matrix(Y_test, Y_pred)
    print(cm)
    print(f" - True Negative: {cm[0][0]}")
    print(f" - False Positive: {cm[0][1]}")
    print(f" - False Negative: {cm[1][0]}")
    print(f" - True Positive: {cm[1][1]}")

if __name__ == "__main__":
    evaluate_model()