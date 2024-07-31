from sklearn.metrics import classification_report, accuracy_score

# Predicting and evaluating the model
y_red_pred = model_red.predict(X_red_test)
y_white_pred = model_white.predict(X_white_test)

print("Red Wine Model Evaluation:")
print(classification_report(y_red_test, y_red_pred))
print("Accuracy:", accuracy_score(y_red_test, y_red_pred))

print("\nWhite Wine Model Evaluation:")
print(classification_report(y_white_test, y_white_pred))
print("Accuracy:", accuracy_score(y_white_test, y_white_pred))
