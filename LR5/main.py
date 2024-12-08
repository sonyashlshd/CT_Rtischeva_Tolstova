import numpy as np
from itertools import product, combinations
import math
import random
from operator import itemgetter


# Генерация бинарной матрицы с заданным количеством столбцов
def generate_basis(num_cols):
    basis = list(product([0, 1], repeat=num_cols))
    return np.array([x[::-1] for x in basis])


# Вычисление f_I
def compute_f(words, indices):
    result = 1
    for j in indices:
        result *= (words[j] + 1) % 2
    return result


# Вычисление f_I_t
def compute_f_t(words, indices, t):
    result = 1
    for j in indices:
        result *= (words[j] + t[j] + 1) % 2
    return result


# Получение вектора v_I
def get_vector_v_I(indices, m):
    if not indices:
        return np.ones(2 ** m, int)

    vector = []
    for words in generate_basis(m):
        value = compute_f(words, indices)
        vector.append(value)
    return vector


# Получение вектора v_I_t
def get_vector_v_I_t(indices, m, t):
    if not indices:
        return np.ones(2 ** m, int)

    vector = []
    for words in generate_basis(m):
        value = compute_f_t(words, indices, t)
        vector.append(value)
    return vector


# Генерация всех комбинаций индексов I
def generate_I_combinations(m, max_size):
    indices = np.arange(m)
    combinations_list = []

    for j in range(len(indices) + 1):
        temp_combinations = list(combinations(indices, j))
        for combo in temp_combinations:
            if len(combo) <= max_size:
                combinations_list.append(combo)

    return combinations_list


# Сортировка комбинаций индексов I
def sort_combinations(I, m):
    max_length = max(len(combo) for combo in I)
    sorted_result = []

    for k in range(max_length + 1):
        s = sum(range(m - k + 1))
        for combo in I:
            if len(combo) == k:
                total_sum = sum(combo)
                if (total_sum == s) or (s != 1 and s % 2 == 1 and total_sum == s + 1):
                    if combo not in sorted_result:
                        sorted_result.append(combo)

    # Добавление оставшихся комбинаций
    for combo in I:
        if combo not in sorted_result:
            total_sum = sum(combo)
            for existing_combo in sorted_result:
                if len(existing_combo) == len(combo) and sum(existing_combo) == total_sum:
                    sorted_result.insert(sorted_result.index(existing_combo) + 1, combo)

    return sorted_result


# Сортировка для мажоритарного декодирования
def major_sort(m, r):
    iterable = np.arange(m)
    temp_combinations = list(combinations(iterable, r))

    if temp_combinations and len(temp_combinations[0]) != 0:
        temp_combinations.sort(key=itemgetter(len(temp_combinations[0]) - 1))

    result = []
    for combo in temp_combinations:
        result.append(combo)

    return result


# Размер порождающей матрицы для кода Рида-Маллера
def rid_maller_size(r, m):
    return sum(math.comb(m, i) for i in range(r + 1))


# Формирование порождающей матрицы G для кода Рида-Маллера
def rid_maller(r, m):
    size = rid_maller_size(r, m)
    matrix = np.zeros((size, pow(2, m)), dtype=int)

    index = 0
    for i in sort_combinations(generate_I_combinations(m, r), m):
        matrix[index] = get_vector_v_I(i, m)
        index += 1

    return matrix


# Получение комплиментарного множества индексов I
def get_complement(indices, m):
    return [i for i in range(m) if i not in indices]


# Поиск строк с f_I равным 1
def find_H_I(indices, m):
    H_I = []

    for words in generate_basis(m):
        if compute_f(words, indices) == 1:
            H_I.append(words)

    return H_I


# Генерация слова с заданным количеством ошибок
def create_word_with_errors(G, r, m, error_count) -> np.ndarray:
    u = np.random.randint(0, 2, rid_maller_size(r, m))

    print("Исходное слово:\n", u)

    u = u.dot(G) % 2
    print("Кодовое слово:\n", u)

    error_vector = np.zeros(len(u), dtype=int)

    # Установка ошибок в случайные позиции
    error_indices = random.sample(range(len(u)), error_count)

    for idx in error_indices:
        error_vector[idx] = 1

    print("Вектор ошибок кратности", error_count, '\n', error_vector)

    u ^= error_vector
    return u


# Мажоритарное декодирование
def major_decoding_algorithm(w, r, m, size):
    current_r = r
    w_r = w.copy()

    Mi = np.zeros(size, dtype=int)

    max_weight_threshold = pow(2, m - r - 1) - 1
    index = 0

    while True:
        for J in major_sort(m, current_r):
            zeros_count = ones_count = 0

            for t in find_H_I(J, m):
                complement_indices = get_complement(J, m)
                V_t = get_vector_v_I_t(complement_indices, m, t)

                c_value = np.dot(w_r, V_t) % 2

                if c_value == 0:
                    zeros_count += 1
                else:
                    ones_count += 1

            # Проверка условий для выхода из цикла декодирования
            if zeros_count > max_weight_threshold and ones_count > max_weight_threshold:
                return

            if zeros_count > pow(2, m - current_r - 1):
                Mi[index] = 0
                index += 1

            if ones_count > pow(2, m - current_r - 1):
                Mi[index] = 1
                index += 1

                V_J = get_vector_v_I(J, m)
                w_r = (w_r + V_J) % 2

        # Обработка уменьшения текущего уровня r
        if current_r > 0:
            if len(w_r) < max_weight_threshold:
                for J in major_sort(m, r + 1):
                    Mi[index] = 0
                    index += 1

                break

            current_r -= 1

        else:
            break

    Mi = Mi[::-1]
    return Mi


G_matrix = rid_maller(2, 4)
G_copy_matrix = np.copy(G_matrix)

print("Порождающая матрица G(2,4): \n", G_matrix)

error_word_single_error = create_word_with_errors(G_matrix, 2, 4, 1)
print("Слово с одной ошибкой: \n", error_word_single_error)

decoded_word_single_error = major_decoding_algorithm(error_word_single_error, 2, 4,
                                                     len(G_matrix))
if decoded_word_single_error is None:
    print("Необходима повторная отправка сообщения!")
else:
    print("Исправленное слово: \n", decoded_word_single_error)
    verification_result_single_error = decoded_word_single_error.dot(G_copy_matrix) % 2
    print("Результат умножения исправленного слова на матрицу G: \n", verification_result_single_error)

error_word_double_error = create_word_with_errors(G_matrix, 2, 4, 2)
print("Слово с двумя ошибками: \n", error_word_double_error)

decoded_word_double_error = major_decoding_algorithm(error_word_double_error,
                                                     2,
                                                     4,
                                                     len(G_matrix))
if decoded_word_double_error is None:
    print("Необходима повторная отправка сообщения!")
else:
    print("Исправленное слово: \n", decoded_word_double_error)
    verification_result_double_error = decoded_word_double_error.dot(G_copy_matrix) % 2
    print("Результат умножения исправленного слова на матрицу G: \n", verification_result_double_error)
