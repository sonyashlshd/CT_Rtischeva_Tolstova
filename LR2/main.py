import copy
import random

X = [[1, 1, 1],
     [1, 1, 0],
     [1, 0, 1],
     [0, 1, 1]]


def identity_matrix(size):
    """Создает единичную матрицу размера size x size."""
    return [[1 if j == i else 0 for j in range(size)] for i in range(size)]


def vector_sum(vector):
    """Суммирует элементы вектора."""
    return sum(vector)


def multiply_vector_matrix(vector, matrix):
    """Умножает вектор на матрицу."""
    result = []
    for i in range(len(matrix[0])):
        total = sum(matrix[j][i] * vector[j] for j in range(len(matrix)))
        result.append(total % 2)
    return result


def add_vectors(v1, v2):
    """Складывает два вектора по модулю 2."""
    return [(v1[i] + v2[i]) % 2 for i in range(len(v1))]


def is_vector_in_matrix(vector, matrix):
    """Проверяет, находится ли вектор в матрице."""
    for row in matrix:
        if all(vector[j] == row[j] for j in range(len(vector))):
            return True
    return False


def get_vector_index(vector, matrix):
    """Возвращает индекс вектора в матрице или -1, если не найден."""
    for index, row in enumerate(matrix):
        if all(vector[j] == row[j] for j in range(len(vector))):
            return index
    return -1


def concatenate_horizontal(m1, m2):
    """Объединяет две матрицы по горизонтали."""
    return [m1_row + m2_row for m1_row, m2_row in zip(m1, m2)]


def concatenate_vertical(m1, m2):
    """Объединяет две матрицы по вертикали."""
    return copy.deepcopy(m1) + m2


def generate_random_matrix(k, n):
    """Генерирует случайную матрицу с заданными условиями."""
    valid = False
    while not valid:
        valid = True
        matrix = []
        for _ in range(k):
            row = [random.randint(0, 1) for _ in range(n)]
            matrix.append(row)

        # Проверка условий на количество единиц в строках и их комбинациях
        for row in matrix:
            if vector_sum(row) < 4:
                valid = False
                break

        for i in range(k - 1):
            for j in range(i + 1, k):
                combined = add_vectors(matrix[i], matrix[j])
                if vector_sum(combined) < 3:
                    valid = False
                    break

        for i in range(k - 2):
            for j in range(i + 1, k - 1):
                for m in range(j + 1, k):
                    combined = add_vectors(add_vectors(matrix[i], matrix[j]), matrix[m])
                    if vector_sum(combined) < 2:
                        valid = False
                        break

        for i in range(k - 3):
            for j in range(i + 1, k - 2):
                for m in range(j + 1, k - 1):
                    for l in range(m + 1, k):
                        combined = add_vectors(add_vectors(matrix[i], matrix[j]), add_vectors(matrix[m], matrix[l]))
                        if vector_sum(combined) < 1:
                            valid = False
                            break

    return matrix


def generate_U(g):
    """Генерирует множество U из матрицы G."""
    U_set = []
    G_copy = copy.deepcopy(g)

    # Добавляем столбцы из G в U
    for j in range(len(G_copy[0])):
        column = [G_copy[i][j] for i in range(len(G_copy))]
        if not is_vector_in_matrix(column, U_set):
            U_set.append(column)

    # Генерация линейных комбинаций
    found_new_combination = True
    while found_new_combination:
        found_new_combination = False
        for i in range(len(U_set)):
            for j in range(i + 1, len(U_set)):
                new_vector = add_vectors(U_set[i], U_set[j])
                if not is_vector_in_matrix(new_vector, U_set):
                    U_set.append(new_vector)
                    found_new_combination = True

    return U_set


def generate_error_vector(length, num_errors):
    """Генерирует вектор ошибок указанной длины с заданным количеством единиц."""
    error_vector = [0] * length
    indices_used = set()

    while len(indices_used) < num_errors:
        index = random.randint(0, length - 1)
        if index not in indices_used:
            error_vector[index] = 1
            indices_used.add(index)

    return error_vector


def generate_V(u_set, g_matrix):
    """Генерирует множество V из U и G."""
    V_set = []

    for u_vector in u_set:
        V_set.append(multiply_vector_matrix(u_vector, g_matrix))

    return V_set


def correct_single_error(H_matrix, syndrome, word):
    """Исправляет одно ошибочное слово на основе синдрома."""
    index = get_vector_index(syndrome, H_matrix)

    if index != -1:
        word[index] ^= 1
    else:
        print("Синдром не найден в матрице H.")

    return word


def correct_double_errors(H_matrix, syndrome, word):
    """Исправляет два ошибочных слова на основе синдрома."""
    first_index = -1
    second_index = -1

    for i in range(len(H_matrix)):
        if get_vector_index(syndrome, H_matrix) == i:
            first_index = i
            break

        for j in range(i + 1, len(H_matrix)):
            if is_vector_in_matrix(syndrome, [add_vectors(H_matrix[i], H_matrix[j])]):
                first_index = i
                second_index = j

    if first_index == -1:
        print("Синдром не найден в матрице синдромов.")
        return word

    word[first_index] ^= 1

    if second_index != -1:
        word[second_index] ^= 1

    return word


def first_part():
    K = 4
    N = 7
    print("\nПервая часть----------\n")

    print("Матрица X:")
    for row in X:
        print(row)

    print("\nМатрица G:")
    G_matrix = concatenate_horizontal(identity_matrix(K), X)
    for row in G_matrix:
        print(row)

    print("\nМатрица H:")
    H_matrix = concatenate_vertical(X, identity_matrix(N - K))
    for row in H_matrix:
        print(row)

    U_generated = generate_U(G_matrix)
    print("\nМножество U:")
    print(U_generated[0])

    V_generated = generate_V(U_generated, G_matrix)
    v_word = V_generated[0]
    print("\nКодовое слово:")
    print(v_word)

    single_error_vector = generate_error_vector(N, 1)
    print("\ne (одна ошибка):")
    print(single_error_vector)

    word_with_single_error = add_vectors(v_word, single_error_vector)
    print("\nКодовое слово с одной ошибкой:")
    print(word_with_single_error)

    syndrome_single_error = multiply_vector_matrix(word_with_single_error, H_matrix)
    print("\nСиндром кодового слова с одной ошибкой:")
    print(syndrome_single_error)

    corrected_word_single = correct_single_error(H_matrix, syndrome_single_error, word_with_single_error)
    print("\nИсправленное кодовое слово с одной ошибкой:")
    print(corrected_word_single)

    print("\nПроверка:")
    print(multiply_vector_matrix(corrected_word_single, H_matrix))

    double_error_vector = generate_error_vector(N, 2)
    print("\ne (две ошибки):")
    print(double_error_vector)

    word_with_double_errors = add_vectors(v_word, double_error_vector)
    print("\nКодовое слово с двумя ошибками:")
    print(word_with_double_errors)

    syndrome_double_errors = multiply_vector_matrix(word_with_double_errors, H_matrix)
    print("\nСиндром кодового слова с двумя ошибками:")
    print(syndrome_double_errors)

    corrected_word_double = correct_double_errors(H_matrix, syndrome_double_errors, word_with_double_errors)
    print("\nИсправленное кодовое слово с двумя ошибками:")
    print(corrected_word_double)

    print("\nПроверка:")
    print(multiply_vector_matrix(corrected_word_double, H_matrix))


def second_part():
    N = 11
    K = 4
    print("\nВторая часть----------\n")

    X_second_gen = generate_random_matrix(K, N - K)

    print("Матрица X:")

    for row in X_second_gen:
        print(row)

    G_second_gen = concatenate_horizontal(identity_matrix(K), X_second_gen)

    print("\nМатрица G:")

    for row in G_second_gen:
        print(row)

    H_second_gen = concatenate_vertical(X_second_gen, identity_matrix(N - K))

    print("\nМатрица H:")

    for row in H_second_gen:
        print(row)

    U_second_gen = generate_U(G_second_gen)

    print("\nМножество U:")
    # Выводим только первый элемент для краткости.
    if U_second_gen:
        print(U_second_gen[0])

    V_second_gen = generate_V(U_second_gen, G_second_gen)
    v_word_second = V_second_gen[0]

    print("\nКодовое слово:")
    print(v_word_second)

    e_single_error_second = generate_error_vector(N, 1)

    print("\ne (одна ошибка):")
    print(e_single_error_second)

    word_with_error_3 = add_vectors(v_word_second, e_single_error_second)
    print("\nКодовое слово с одной ошибкой:")
    print(word_with_error_3)

    syndrome_3 = multiply_vector_matrix(word_with_error_3, H_second_gen)
    print("\nСиндром кодового слова с одной ошибкой:")
    print(syndrome_3)

    # Исправление кода с одной ошибкой.
    corrected_word_single = correct_single_error(H_second_gen, syndrome_3, word_with_error_3)
    print("\nИсправленное кодовое слово с одной ошибкой:")
    print(corrected_word_single)

    # Проверка.
    print("\nПроверка:")
    print(multiply_vector_matrix(corrected_word_single, H_second_gen))

    # Обработка двух ошибок.
    e_double_error_second = generate_error_vector(N, 2)
    print("\ne (две ошибки):")
    print(e_double_error_second)

    word_with_errors_two = add_vectors(v_word_second, e_double_error_second)
    print("\nКодовое слово с двумя ошибками:")
    print(word_with_errors_two)

    syndrome_two = multiply_vector_matrix(word_with_errors_two, H_second_gen)
    print("\nСиндром кодового слова с двумя ошибками:")
    print(syndrome_two)

    # Исправление кода с двумя ошибками.
    corrected_word_two = correct_double_errors(H_second_gen, syndrome_two, word_with_errors_two)
    print("\nИсправленное кодовое слово с двумя ошибками:")
    print(corrected_word_two)

    print("\nПроверка:")
    print(multiply_vector_matrix(corrected_word_two, H_second_gen))

    # Обработка трех ошибок.
    e_triple_error = generate_error_vector(N, 3)
    print("\ne (три ошибки):")
    print(e_triple_error)

    word_with_three_errors = add_vectors(v_word_second, e_triple_error)
    print("\nКодовое слово с тремя ошибками:")
    print(word_with_three_errors)

    syndrome_three = multiply_vector_matrix(word_with_three_errors, H_second_gen)
    print("\nСиндром кодового слова с тремя ошибками:")
    print(syndrome_three)

    # Исправление кода с тремя ошибками.
    corrected_word_three = correct_double_errors(H_second_gen, syndrome_three, word_with_three_errors)
    print("\nИсправленное кодовое слово с тремя ошибками:")
    print(corrected_word_three)

    print("\nПроверка:")
    print(multiply_vector_matrix(corrected_word_three, H_second_gen))


first_part()
second_part()
