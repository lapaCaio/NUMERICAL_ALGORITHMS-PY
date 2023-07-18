# DESENVOLVIDO POR:
# PEDRO LUCAS NASCIMENTO MOREIRA MACHADO
# KAIO STEFAN CAMPOS NUNES
# CAIO PEREIRA LAPA

import numpy as np


# Ajuste do modelo u(t): N = B0 + B1t
def fit_model_1(idade, quantidade):
    # Calcular os coeficientes B0 e B1 usando as fórmulas de regressão linear
    n = len(idade)
    soma_x = sum(idade)
    soma_y = sum(quantidade)
    soma_x_quadrado = sum(x ** 2 for x in idade)
    soma_xy = sum(x * y for x, y in zip(idade, quantidade))

    B1 = (n * soma_xy - soma_x * soma_y) / (n * soma_x_quadrado - soma_x ** 2)
    B0 = (soma_y - B1 * soma_x) / n

    N_hat1 = [B0 + B1 * t for t in idade]

    return N_hat1


# Ajuste do modelo u(t): N = B0 + B1t + B2t²
def fit_model_2(idade, quantidade):
    # Calcular os coeficientes B0, B1 e B2 usando as fórmulas de regressão polinomial
    n = len(idade)
    soma_x = sum(idade)
    soma_y = sum(quantidade)
    soma_x_quadrado = sum(x ** 2 for x in idade)
    soma_x_cubico = sum(x ** 3 for x in idade)
    soma_x_quarta = sum(x ** 4 for x in idade)
    soma_xy = sum(x * y for x, y in zip(idade, quantidade))
    soma_x_quadrado_y = sum(x ** 2 * y for x, y in zip(idade, quantidade))

    A = [[n, soma_x, soma_x_quadrado],
         [soma_x, soma_x_quadrado, soma_x_cubico],
         [soma_x_quadrado, soma_x_cubico, soma_x_quarta]]

    B = [soma_y, soma_xy, soma_x_quadrado_y]

    B0, B1, B2 = np.linalg.solve(A, B)

    N_hat2 = [B0 + B1 * t + B2 * t ** 2 for t in idade]

    return N_hat2


# Ajuste do modelo u(t): N = B0 * exp(B1t)
def fit_model_3(idade, quantidade):
    # Calcular os coeficientes B0 e B1 usando as fórmulas de regressão exponencial
    n = len(idade)
    soma_x = sum(idade)
    soma_y = sum([np.log(y) for y in quantidade])
    soma_x_quadrado = sum(x ** 2 for x in idade)
    soma_xy = sum(x * np.log(y) for x, y in zip(idade, quantidade))

    B1 = (n * soma_xy - soma_x * soma_y) / (n * soma_x_quadrado - soma_x ** 2)
    B0 = np.exp((soma_y - B1 * soma_x) / n)

    N_hat3 = [B0 * np.exp(B1 * t) for t in idade]

    return N_hat3


# Calcular o coeficiente de determinação (r²)
def calculate_r_squared(quantidade, N_hat):
    media_quantidade = sum(quantidade) / len(quantidade)
    numerador = sum((y - y_hat) ** 2 for y, y_hat in zip(quantidade, N_hat))
    denominador = sum((y - media_quantidade) ** 2 for y in quantidade)
    r_squared = 1 - numerador / denominador
    return r_squared


# Determinar o modelo com o maior coeficiente de determinação (r²)
def determine_best_model(r_squared_values):
    return r_squared_values.index(max(r_squared_values))  # Índice do modelo com o maior r²
