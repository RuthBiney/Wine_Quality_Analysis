from sklearn.preprocessing import StandardScaler

# Separating features and target variable
X_red = red_wine.drop('quality', axis=1)
y_red = red_wine['quality']
X_white = white_wine.drop('quality', axis=1)
y_white = white_wine['quality']

# Standardizing the features
scaler = StandardScaler()
X_red_scaled = scaler.fit_transform(X_red)
X_white_scaled = scaler.fit_transform(X_white)
