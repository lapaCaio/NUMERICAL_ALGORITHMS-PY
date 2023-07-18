# DESENVOLVIDO POR:
# PEDRO LUCAS NASCIMENTO MOREIRA MACHADO
# KAIO STEFAN CAMPOS NUNES
# CAIO PEREIRA LAPA

import matplotlib.pyplot as plt


def plot_graficos(idade, quantidade, N_hat1, N_hat2, N_hat3):
    # Plotar os dados e as curvas ajustadas
    plt.scatter(idade, quantidade, color='blue', label='Dados')
    plt.plot(idade, N_hat1, color='red', label='Modelo 1: N = B0 + B1t')
    plt.plot(idade, N_hat2, color='green', label='Modelo 2: N = B0 + B1t + B2t^2')
    plt.plot(idade, N_hat3, color='purple', label='Modelo 3: N = B0 * exp(B1t)')
    plt.xlabel('Idade (anos)')
    plt.ylabel('Quantidade de Carbono-14')
    plt.title('Ajuste dos modelos aos dados')
    plt.legend()
    plt.show()
