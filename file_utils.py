def read_matrix_from_file(file_name):
    list1 = []
    list2 = []

    try:
        with open(file_name, 'r') as file:
            for line in file:
                values = line.strip().split()
                if len(values) == 2:
                    list1.append(int(values[0]))
                    list2.append(int(values[1]))
                else:
                    print(f"A linha '{line.strip()}' não possui dois valores e será ignorada.")

    except FileNotFoundError:
        print(f"Arquivo '{file_name}' não encontrado.")

    return list1, list2


def save_results_to_file(file_path, r_squared_values, best_model_index):
    with open(file_path, 'w') as file:
        file.write('Coeficiente de Determinação (r^2):\n')
        file.write('Modelo 1: N = β0 + β1t - r^2 = {}\n'.format(r_squared_values[0]))
        file.write('Modelo 2: N = β0 + β1t + β2t^2 - r^2 = {}\n'.format(r_squared_values[1]))
        file.write('Modelo 3: N = β0 * exp(β1t) - r^2 = {}\n'.format(r_squared_values[2]))

        if best_model_index == 0:
            file.write('O Modelo 1 tem melhor qualidade.\n')
        elif best_model_index == 1:
            file.write('O Modelo 2 tem melhor qualidade.\n')
        else:
            file.write('O Modelo 3 tem melhor qualidade.\n')
