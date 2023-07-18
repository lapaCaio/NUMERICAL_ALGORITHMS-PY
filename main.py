# DESENVOLVIDO POR:
# PEDRO LUCAS NASCIMENTO MOREIRA MACHADO
# KAIO STEFAN CAMPOS NUNES
# CAIO PEREIRA LAPA


import graphs as g
import file_utils as fu
import algorithms as a


def main():
    idade, quantidade = fu.read_matrix_from_file('input.txt')

    # Ajuste dos modelos
    N_hat1 = a.fit_model_1(idade, quantidade)
    N_hat2 = a.fit_model_2(idade, quantidade)
    N_hat3 = a.fit_model_3(idade, quantidade)

    # Cálculo do coeficiente de determinação (r^2) para cada modelo
    r_squared1 = a.calculate_r_squared(quantidade, N_hat1)
    r_squared2 = a.calculate_r_squared(quantidade, N_hat2)
    r_squared3 = a.calculate_r_squared(quantidade, N_hat3)

    # Determinar o modelo com o maior coeficiente de determinação (r^2)
    r_squared_values = [r_squared1, r_squared2, r_squared3]
    best_model_index = a.determine_best_model(r_squared_values)

    # Plotar os dados e as curvas ajustadas
    g.plot_graficos(idade, quantidade, N_hat1, N_hat2, N_hat3)

    # Imprimir o coeficiente de determinação (r^2) para cada modelo
    print('Coeficiente de Determinação (r^2):')

    print('Modelo 1: N = B0 + B1t - r² =', r_squared1)
    print('Modelo 2: N = B0 + B1t + B2t² - r² =', r_squared2)
    print('Modelo 3: N = B0 * exp(B1t) - r² =', r_squared3)

    # Imprimir o modelo com o melhor coeficiente de determinação
    if best_model_index == 0:
        print('O Modelo 1 tem melhor qualidade.')
    elif best_model_index == 1:
        print('O Modelo 2 tem melhor qualidade.')
    else:
        print('O Modelo 3 tem melhor qualidade.')

    # Salvar os resultados no arquivo de saída
    fu.save_results_to_file('output.txt', [r_squared1, r_squared2, r_squared3], best_model_index)


if __name__ == "__main__":
    main()
