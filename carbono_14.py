import numpy as np
import matplotlib.pyplot as plt

# Dados de referência - Quantidade de carbono-14 nas amostras
# Coloque os dados de referência na forma de listas de idade (t) e quantidade (N)
idade = [77, 131, 136, 177, 186, 244, 323, 334, 371, 459, 509, 537, 597, 689, 701, 798, 848, 890, 960, 1060, 1064, 1070,
         1159, 1177, 1215, 1287, 1313, 1354, 1422, 1510, 1577, 1582, 1609, 1625, 1686, 1764, 1841, 1846, 1901, 1963,
         2001, 2022, 2031, 2084, 2169, 2196, 2234, 2312]

quantidade = [59861545127, 59572980868, 59437138543, 58913989631, 58979835517, 58768052408, 57311820970, 57734782072,
              57776445703, 56866618736, 56723009875, 56531494197, 56223285835, 55203107473, 55122704014, 54777186462,
              54547384762, 54271895392, 53415840481, 52870997216, 52745368905, 52506948155, 52440235651, 52826361174,
              51786776587, 51435845232, 51273955360, 51019735808, 50900833780, 50363851068, 49558811422, 49428714833,
              49366510689, 49270654651, 49106843188, 48845561215, 48294428313, 48165287627, 47845812585, 47588217390,
              47270353646, 47150409778, 47099084756, 46798005196, 46519144898, 46268072773, 45756272155, 45524579654]


# Ajuste do modelo u(t): N = β0 + β1t
A1 = np.vstack([np.ones(len(idade)), idade]).T
beta1 = np.linalg.lstsq(A1, quantidade, rcond=None)[0]
N_hat1 = beta1[0] + beta1[1] * np.array(idade)
r_squared1 = 1 - np.sum((quantidade - N_hat1) ** 2) / np.sum((quantidade - np.mean(quantidade)) ** 2)

# Ajuste do modelo u(t): N = β0 + β1t + β2t^2
A2 = np.vstack([np.ones(len(idade)), idade, np.power(idade, 2)]).T
beta2 = np.linalg.lstsq(A2, quantidade, rcond=None)[0]
N_hat2 = beta2[0] + beta2[1] * np.array(idade) + beta2[2] * np.power(np.array(idade), 2)
r_squared2 = 1 - np.sum((quantidade - N_hat2) ** 2) / np.sum((quantidade - np.mean(quantidade)) ** 2)

# Ajuste do modelo u(t): N = β0 * exp(β1t)
A3 = np.vstack([np.ones(len(idade)), np.array(idade)]).T
log_quantidade = np.log(quantidade)
beta3 = np.linalg.lstsq(A3, log_quantidade, rcond=None)[0]
N_hat3 = np.exp(beta3[0] + beta3[1] * np.array(idade))
r_squared3 = 1 - np.sum((quantidade - N_hat3) ** 2) / np.sum((quantidade - np.mean(quantidade)) ** 2)

# Determinar o modelo com o maior coeficiente de determinação (r²)
r_squared_values = [r_squared1, r_squared2, r_squared3]
best_model_index = np.argmax(r_squared_values)  # Índice do modelo com o maior r²

# Plotar os dados e as curvas ajustadas
plt.scatter(idade, quantidade, color='blue', label='Dados')
plt.plot(idade, N_hat1, color='red', label='Modelo 1: N = β0 + β1t')
plt.plot(idade, N_hat2, color='green', label='Modelo 2: N = β0 + β1t + β2t^2')
plt.plot(idade, N_hat3, color='purple', label='Modelo 3: N = β0 * exp(β1t)')
plt.xlabel('Idade (anos)')
plt.ylabel('Quantidade de Carbono-14')
plt.title('Ajuste dos modelos aos dados')
plt.legend()
plt.show()

# Imprimir o coeficiente de determinação (r²) para cada modelo
print('Coeficiente de Determinação (r²):')
print('Modelo 1: N = β0 + β1t - r² =', r_squared1)
print('Modelo 2: N = β0 + β1t + β2t^2 - r² =', r_squared2)
print('Modelo 3: N = β0 * exp(β1t) - r² =', r_squared3)

# Imprimir o modelo com o melhor coeficiente de determinação
if best_model_index == 0:
    print('O Modelo 1 tem melhor qualidade.')
elif best_model_index == 1:
    print('O Modelo 2 tem melhor qualidade.')
else:
    print('O Modelo 3 tem melhor qualidade.')
