import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

#Cargamos dataset
df = pd.read_csv("Student_performance_data _.csv")

#Variables predictoras
X = df.drop(columns=["StudentID", "GPA", "GradeClass"])
y = df["GPA"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Generamos polinomio de grado 2
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

#Dividimos en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y, test_size=0.2, random_state=42
)

# ===== Modelo Ridge =====
ridge = Ridge(alpha=1.0)  # alpha controla la regularización
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

print("\n--- Ridge (Polinomio grado 2) ---")
print("MSE:", mean_squared_error(y_test, y_pred_ridge))
print("MAE:", mean_absolute_error(y_test, y_pred_ridge))
print("R²:", r2_score(y_test, y_pred_ridge))
print("CV R²:", cross_val_score(ridge, X_poly, y, cv=5, scoring="r2").mean())

#Modelo Lasso
lasso = Lasso(alpha=0.001, max_iter=10000)  # alpha pequeño para no ser agresivo
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

print("\n--- Lasso (Polinomio grado 2) ---")
print("MSE:", mean_squared_error(y_test, y_pred_lasso))
print("MAE:", mean_absolute_error(y_test, y_pred_lasso))
print("R²:", r2_score(y_test, y_pred_lasso))
print("CV R²:", cross_val_score(lasso, X_poly, y, cv=5, scoring="r2").mean())
