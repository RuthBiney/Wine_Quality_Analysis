from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Splitting the data into training and testing sets
X_red_train, X_red_test, y_red_train, y_red_test = train_test_split(X_red_scaled, y_red, test_size=0.2, random_state=42)
X_white_train, X_white_test, y_white_train, y_white_test = train_test_split(X_white_scaled, y_white, test_size=0.2, random_state=42)

# Building the model
model_red = RandomForestClassifier(random_state=42)
model_white = RandomForestClassifier(random_state=42)

# Training the model
model_red.fit(X_red_train, y_red_train)
model_white.fit(X_white_train, y_white_train)
