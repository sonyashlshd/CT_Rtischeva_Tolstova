import copy
import random


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
    for i in range(0, len(matrix)):
        a = 0
        for j in range(0, len(matrix[0])):
            if vector[j] == matrix[i][j]:
                a += 1
        if a == len(matrix[0]):
            return True
    return False


def get_vector_index(vector, matrix):
    """Возвращает индекс вектора в матрице или -1, если не найден."""
    for i in range(0, len(matrix)):
        a = 0
        for j in range(0, len(matrix[0])):
            if vector[j] == matrix[i][j]:
                a += 1
        if a == len(matrix[0]):
            return i
    return -1


def concatenate_horizontal(m1, m2):
    """Объединяет две матрицы по горизонтали."""
    return [m1_row + m2_row for m1_row, m2_row in zip(m1, m2)]


def concatenate_vertical(m1, m2):
    """Объединяет две матрицы по вертикали."""
    return copy.deepcopy(m1) + m2


def generate_random_matrix(n, k):
    """Генерирует случайную матрицу с заданными условиями."""
    X = []
    x = []
    for i in range(0, n - k):
        x.append(0)
    flag = True
    while flag:
        flag = False
        for j in range(0, len(x)):
            if x[len(x) - j - 1] == 0:
                x[len(x) - j - 1] = 1
                break
            else:
                x[len(x) - j - 1] = 0
        if (sum(x) > 1):
            X.append(copy.copy(x))
        if len(X) != k:
            flag = True
    return X


def generate_U(g):
    """Генерирует множество U из матрицы G."""
    U = []
    G = copy.copy(g)
    for j in range(0, len(G[0])):
        u = []
        for i in range(0, len(G)):
            u.append(G[i][j])
        if not is_vector_in_matrix(u, U):
            U.append(u)
    flag = True
    while flag:
        flag = False
        for i in range(0, len(U)):
            for j in range(i + 1, len(U)):
                if len(U) == 2047:
                    return U
                if not is_vector_in_matrix(add_vectors(U[i], U[j]), U):
                    U.append(add_vectors(U[i], U[j]))
                    flag = True
    return U


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


def generate_V(u, g):
    """Генерирует множество V из U и G."""
    U = copy.copy(u)
    G = copy.copy(g)
    V = []
    for i in range(0, len(U)):
        V.append(multiply_vector_matrix(U[i], G))
    return V


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
    k = -1
    d = -1
    for i in range(len(H_matrix)):
        if is_vector_in_matrix(syndrome, [H_matrix[i]]):
            k = i
            break
        for j in range(i + 1, len(H_matrix)):
            if is_vector_in_matrix(syndrome, [add_vectors(H_matrix[i], H_matrix[j])]):
                k = i
                d = j
    if k == -1:
        print("Такого синдрома нет в матрице синдромов")
    else:
        word[k] += 1
        word[k] %= 2
        if d != -1:
            word[d] += 1
            word[d] %= 2
    return word


def correct_three_errors(H_matrix, syndrome, word):
    """Исправляет слово с тремя ошибками на основе синдрома."""
    indices = [-1, -1, -1]  # Индексы для ошибок
    found = False

    for i in range(len(H_matrix)):
        if is_vector_in_matrix(syndrome, [H_matrix[i]]):
            indices[0] = i
            found = True
            break

        for j in range(i + 1, len(H_matrix)):
            if is_vector_in_matrix(syndrome, [add_vectors(H_matrix[i], H_matrix[j])]):
                indices[0] = i
                indices[1] = j
                found = True
                break

            for k in range(j + 1, len(H_matrix)):
                if is_vector_in_matrix(syndrome, [add_vectors(add_vectors(H_matrix[i], H_matrix[j]), H_matrix[k])]):
                    indices[0] = i
                    indices[1] = j
                    indices[2] = k
                    found = True
                    break

            if found:
                break

        if found:
            break

    if indices[0] == -1:
        print("Синдром не найден в матрице синдромов.")
    else:
        for idx in indices:
            if idx != -1:
                word[idx] ^= 1  # Исправляем ошибку (инвертируем бит)

    return word


def correct_four_errors(H_matrix, syndrome, word):
    """Исправляет слово с четырьмя ошибками на основе синдрома."""
    indices = [-1, -1, -1, -1]
    found = False

    for i in range(len(H_matrix)):
        if is_vector_in_matrix(syndrome, [H_matrix[i]]):
            indices[0] = i
            found = True
            break

        for j in range(i + 1, len(H_matrix)):
            if is_vector_in_matrix(syndrome, [add_vectors(H_matrix[i], H_matrix[j])]):
                indices[0] = i
                indices[1] = j
                found = True
                break

            for k in range(j + 1, len(H_matrix)):
                if is_vector_in_matrix(syndrome, [add_vectors(add_vectors(H_matrix[i], H_matrix[j]), H_matrix[k])]):
                    indices[0] = i
                    indices[1] = j
                    indices[2] = k
                    found = True
                    break

                for l in range(k + 1, len(H_matrix)):
                    if is_vector_in_matrix(syndrome, [
                        add_vectors(add_vectors(add_vectors(H_matrix[i], H_matrix[j]), H_matrix[k]), H_matrix[l])]):
                        indices[0] = i
                        indices[1] = j
                        indices[2] = k
                        indices[3] = l
                        found = True
                        break

                if found:
                    break

            if found:
                break

        if found:
            break

    if indices[0] == -1:
        print("Синдром не найден в матрице синдромов.")
    else:
        for idx in indices:
            if idx != -1:
                word[idx] ^= 1  # Исправляем ошибку (инвертируем бит)

    return word


def G1(r):
    n = 2 ** r - 1
    k = 2 ** r - r - 1
    return concatenate_horizontal(identity_matrix(k), generate_random_matrix(n, k))


def H1(r):
    n = 2 ** r - 1
    k = 2 ** r - r - 1
    return concatenate_vertical(generate_random_matrix(n, k), identity_matrix(n - k))


def G2(r):
    G = G1(r)
    for i in range(len(G)):
        a_count = sum(G[i])
        G[i].append(0 if a_count % 2 == 0 else 1)
    return G


def H2(r):
    H = H1(r)
    parity_check_row = [0] * len(H[0])
    H.append(parity_check_row)
    for i in range(len(H)):
        H[i].append(1)
    return H


def fun1(R):
    r = R

    G = G1(r)
    print("\nG = ")
    for k in range(0, len(G)):
        print(G[k])

    H = H1(r)
    print("\nH = ")
    for k in range(0, len(H)):
        print(H[k])

    U1 = generate_U(G)
    print("\nU = ")
    print(U1[random.randint(0, len(U1) - 1)])

    V1 = generate_V(U1, G)
    v = V1[random.randint(0, len(V1) - 1)]
    print("\nкодовое слово = ")
    print(v)

    e1 = generate_error_vector(len(v), 1)
    print("\ne1 = ")
    print(e1)

    word1 = add_vectors(v, e1)
    print("\nкодовое слово с одной ошибкой = ")
    print(word1)

    sindrom1 = multiply_vector_matrix(word1, H)
    print("\nсиндром кодового слова с одной ошибкой = ")
    print(sindrom1)

    print("\nисправленное кодовое слово c одной ошибкой = ")
    correct_single_error(H, sindrom1, word1)
    print(word1)

    print("\nпроверка = ")
    print(multiply_vector_matrix(word1, H))

    e2 = generate_error_vector(len(v), 2)
    print("\ne2 = ")
    print(e2)

    word2 = add_vectors(v, e2)
    print("\nкодовое слово с двумя ошибками = ")
    print(word2)

    sindrom2 = multiply_vector_matrix(word2, H)
    print("\nсиндром кодового слова с двумя ошибками = ")
    print(sindrom2)

    print("\nисправленное кодовое слово c двумя ошибками = ")
    correct_double_errors(H, sindrom2, word2)
    print(word2)

    print("\nпроверка = ")
    print(multiply_vector_matrix(word2, H))

    e3 = generate_error_vector(len(v), 3)
    print("\ne3 = ")
    print(e3)

    word3 = add_vectors(v, e3)
    print("\nкодовое слово с тремя ошибками = ")
    print(word3)

    sindrom3 = multiply_vector_matrix(word3, H)
    print("\nсиндром кодового слова с тремя ошибками = ")
    print(sindrom3)

    print("\nисправленное кодовое слово c тремя ошибками = ")
    correct_three_errors(H, sindrom3, word3)
    print(word3)

    print("\nпроверка = ")
    print(multiply_vector_matrix(word3, H))


def fun2(R):
    r = R

    G = G2(r)
    print("\nG = ")
    for k in range(0, len(G)):
        print(G[k])

    H = H2(r)
    print("\nH = ")
    for k in range(0, len(H)):
        print(H[k])

    U1 = generate_U(G)
    print("\nU = ")
    print(U1[random.randint(0, len(U1) - 1)])

    V1 = generate_V(U1, G)
    v = V1[random.randint(0, len(V1) - 1)]
    print("\nкодовое слово = ")
    print(v)

    e2 = generate_error_vector(len(v), 2)
    print("\ne2 = ")
    print(e2)

    word2 = add_vectors(v, e2)
    print("\nкодовое слово с двумя ошибками = ")
    print(word2)

    sindrom2 = multiply_vector_matrix(word2, H)
    print("\nсиндром кодового слова с двумя ошибками = ")
    print(sindrom2)

    print("\nисправленное кодовое слово c двумя ошибками = ")
    correct_double_errors(H, sindrom2, word2)
    print(word2)

    print("\nпроверка = ")
    print(multiply_vector_matrix(word2, H))

    e3 = generate_error_vector(len(v), 3)
    print("\ne3 = ")
    print(e3)

    word3 = add_vectors(v, e3)
    print("\nкодовое слово с тремя ошибками = ")
    print(word3)

    sindrom3 = multiply_vector_matrix(word3, H)
    print("\nсиндром кодового слова с тремя ошибками = ")
    print(sindrom3)

    print("\nисправленное кодовое слово c тремя ошибками = ")
    correct_three_errors(H, sindrom3, word3)
    print(word3)

    print("\nпроверка = ")
    print(multiply_vector_matrix(word3, H))

    e4 = generate_error_vector(len(v), 4)
    print("\ne4 = ")
    print(e4)

    word4 = add_vectors(v, e4)
    print("\nкодовое слово с четырьмя ошибками = ")
    print(word4)

    sindrom4 = multiply_vector_matrix(word4, H)
    print("\nсиндром кодового слова с четырьмя ошибками = ")
    print(sindrom4)

    print("\nисправленное кодовое слово c четырьмя ошибками = ")
    correct_four_errors(H, sindrom4, word4)
    print(word4)

    print("\nпроверка = ")
    print(multiply_vector_matrix(word4, H))


fun1(2)
fun1(3)
fun1(4)
fun2(2)
fun2(3)
fun2(4)
