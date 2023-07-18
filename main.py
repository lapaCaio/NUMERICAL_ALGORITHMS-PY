import numpy as np
import matplotlib.pyplot as plt
import file_utils as fu


# Ajuste do modelo u(t): N = ?0 + ?1t
def fit_model_1(age, quantity):
    A1 = np.vstack([np.ones(len(age)), age]).T
    beta1 = np.linalg.lstsq(A1, quantity, rcond=None)[0]
    N_hat1 = beta1[0] + beta1[1] * np.array(age)
    return N_hat1


# Ajuste do modelo u(t): N = ?0 + ?1t + ?2t^2
def fit_model_2(age, quantity):
    A2 = np.vstack([np.ones(len(age)), age, np.power(age, 2)]).T
    beta2 = np.linalg.lstsq(A2, quantity, rcond=None)[0]
    N_hat2 = beta2[0] + beta2[1] * np.array(age) + beta2[2] * np.power(np.array(age), 2)
    return N_hat2


# Ajuste do modelo u(t): N = ?0 * exp(?1t)
def fit_model_3(age, quantity):
    A3 = np.vstack([np.ones(len(age)), np.array(age)]).T
    log_quantity = np.log(quantity)
    beta3 = np.linalg.lstsq(A3, log_quantity, rcond=None)[0]
    N_hat3 = np.exp(beta3[0] + beta3[1] * np.array(age))
    return N_hat3


# Determinar o modelo com o maior coeficiente de determinação (r²)
def calculate_r_squared(quantity, N_hat):
    r_squared = 1 - np.sum((quantity - N_hat) ** 2) / np.sum((quantity - np.mean(quantity)) ** 2)
    return r_squared


def determine_best_model(r_squared_values):
    return np.argmax(r_squared_values)  # Índice do modelo com o maior r²


def main():
    age, quantity = fu.read_matrix_from_file('input.txt')

    # Ajuste dos modelos
    N_hat1 = fit_model_1(age, quantity)
    N_hat2 = fit_model_2(age, quantity)
    N_hat3 = fit_model_3(age, quantity)

    # Cálculo do coeficiente de determinação (r²) para cada modelo
    r_squared1 = calculate_r_squared(quantity, N_hat1)
    r_squared2 = calculate_r_squared(quantity, N_hat2)
    r_squared3 = calculate_r_squared(quantity, N_hat3)

    # Determinar o modelo com o maior coeficiente de determinação (r²)
    r_squared_values = [r_squared1, r_squared2, r_squared3]
    best_model_index = determine_best_model(r_squared_values)

    # Plotar os dados e as curvas ajustadas
    plt.scatter(age, quantity, color='blue', label='Data')
    plt.plot(age, N_hat1, color='red', label='Modelo 1: N = β0 + β1t')
    plt.plot(age, N_hat2, color='green', label='Modelo 2: N = β0 + β1t + β2t^2')
    plt.plot(age, N_hat3, color='purple', label='Modelo 3: N = β0 * exp(β1t)')
    plt.xlabel('Idade (anos)')
    plt.ylabel('Quantidade de Carbono-14')
    plt.title('Ajuste dos modelos aos dados')
    plt.legend()
    plt.show()

    # Imprimir o coeficiente de determinação (r²) para cada modelo
    print('Coeficiente de Determinação (r²):')

    print('Modelo 1: N = B0 + B1t - r² =', r_squared1)
    print('Modelo 2: N = B0 + B1t + B2t^2 - r² =', r_squared2)
    print('Modelo 3: N = B0 * exp(B1t) - r² =', r_squared3)

    # Imprimir o modelo com o melhor coeficiente de determinação
    if best_model_index == 0:
        print('O Modelo 1 tem melhor qualidade.')
    elif best_model_index == 1:
        print('O Modelo 2 tem melhor qualidade.')
    else:
        print('O Modelo 3 tem melhor qualidade.')

    # Salva os resultados no arquivo de output
    fu.save_results_to_file('output.txt', [r_squared1, r_squared2, r_squared3], best_model_index)


if __name__ == "__main__":
    main()
