import numpy as np
import matplotlib.pyplot as plt

class SimpleLinearRegression:
    def __init__(self):
        self.intercept = None
        self.coef = None

    def fit(self, X, y):
        # Calcular los coeficientes de regresión
        X_mean = np.mean(X)
        y_mean = np.mean(y)
        numerator = np.sum((X - X_mean) * (y - y_mean))
        denominator = np.sum((X - X_mean) ** 2)
        self.coef = numerator / denominator
        self.intercept = y_mean - self.coef * X_mean

    def predict(self, X):
        return self.intercept + self.coef * X

# Datos del caso Benetton
sales = np.array([651, 762, 856, 1063, 1190, 1298, 1421, 1440, 1518])
advertising = np.array([23, 26, 30, 34, 43, 48, 52, 57, 58])

# Crear una instancia del modelo de regresión lineal
model = SimpleLinearRegression()

# Ajustar el modelo a los datos
model.fit(advertising, sales)

# Imprimir la ecuación de regresión
equation = f'Ecuación de Regresión: Sales = {model.intercept:.2f} + {model.coef:.2f} * Advertising'
print(equation)

# Realizar cinco predicciones
values_to_predict = [40, 45, 50, 55, 60]
predicted_sales = [model.predict(value) for value in values_to_predict]

# Imprimir las predicciones
for value, sales_estimate in zip(values_to_predict, predicted_sales):
    print(f'Para Advertising = {value}, Sales estimado = {sales_estimate:.2f}')

# Graficar la regresión
plt.scatter(advertising, sales, label='Datos reales')
plt.plot(advertising, model.predict(advertising), color='red', label='Regresión Lineal')
plt.scatter(values_to_predict, predicted_sales, color='green', marker='o', label='Predicciones')
plt.xlabel('Advertising')
plt.ylabel('Sales')
plt.title('Regresión Lineal Simple para Caso Benetton')
plt.legend()
plt.show()
