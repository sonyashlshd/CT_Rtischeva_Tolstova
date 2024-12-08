import copy
import random

B = [[1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
     [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
     [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1],
     [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
     [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
     [1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
     [0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1],
     [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
     [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1],
     [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
     [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], ]


def identity_matrix(size):
    """Создает единичную матрицу размера size x size."""
    return [[1 if j == i else 0 for j in range(size)] for i in range(size)]


def vector_sum(vector):
    """Суммирует элементы вектора."""
    return sum(vector)


def multiply_vector_matrix(vector, matrix):
    """Умножает вектор на матрицу."""
    result = []
    for i in range(0, len(matrix[0])):
        total = sum(matrix[j][i] * vector[j] for j in range(0, len(matrix)))
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


def generate_U(h, b):
    """Генерирует множество U из H, B."""
    flag = True
    while flag:
        flag = False
        u = []
        w = []
        H = copy.copy(h)
        B = copy.copy(b)
        for i in range(0, len(H)):
            w.append(random.randint(0, 1))
        s = multiply_vector_matrix(w, H)

        sB = multiply_vector_matrix(s, B)
        if sum(sB) < 4:
            for i in range(0, len(s)):
                u.append(0)
            for i in range(0, len(s)):
                u.append(sB[i])

        for i in range(0, len(B)):
            if sum(add_vectors(sB, B[i])) < 3:
                for j in range(0, len(s)):
                    if j == i:
                        u.append(1)
                    else:
                        u.append(0)
                u = add_vectors(sB, B[i])
                break
        if len(u) == 0:
            flag = True
    return u


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
    return multiply_vector_matrix(U, G)


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


def GoleyG():
    return concatenate_horizontal(identity_matrix(12), B)


def GoleyH():
    return concatenate_vertical(identity_matrix(12), B)


def zeros(len1, len2):
    M = []
    for i in range(len1):
        v = []
        for j in range(len2):
            v.append(0)
        M.append(v)
    return M


def ones(len1, len2):
    M = []
    for i in range(len1):
        v = []
        for j in range(len2):
            v.append(1)
        M.append(v)
    return M


def eye(len1):
    M = []
    for i in range(len1):
        v = []
        for j in range(len1):
            if i == j:
                v.append(1)
            else:
                v.append(0)
        M.append(v)
    return M


def kron(matrix1, matrix2):
    K = copy.copy(matrix2)
    for i in range(len(matrix1[0]) - 1):
        K = concatenate_horizontal(K, matrix2)
    line = copy.copy(K)
    for i in range(len(matrix1) - 1):
        K = concatenate_vertical(K, line)
    k = []
    for i in range(len(K)):
        u = []
        for j in range(len(K[0])):
            u.append(K[i][j] * matrix1[i // len(matrix2)][j // len(matrix2[0])])
        k.append(u)
    return k


def dot(matrix1, matrix2):
    D = zeros(len(matrix1), len(matrix2[0]))
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            a = 0
            for k in range(len(matrix1[0])):
                a += matrix1[i][k] * matrix2[k][j]
            D[i][j] = a
    return D


def RM(r, m):
    if 0 < r < m:
        G11_2 = RM(r, m - 1)
        G22 = RM(r - 1, m - 1)

        G_left = concatenate_vertical(G11_2, zeros(len(G22), len(G11_2[0])))
        G_right = concatenate_vertical(G11_2, G22)
        return concatenate_horizontal(G_left, G_right)
    elif r == 0:
        return ones(1, 2 ** m)
    elif r == m:
        G_top = RM(r - 1, m)
        bottom_matrix = zeros(1, 2 ** m)
        bottom_matrix[0][len(bottom_matrix[0]) - 1] = 1
        return concatenate_vertical(G_top, bottom_matrix)


def Kroneker_H():
    return [[1, 1], [1, -1]]


def H_m_i(i, m):
    mult_kron = kron(kron(eye(2 ** (m - i)), Kroneker_H()), eye(2 ** (i - 1)))
    return mult_kron


def w_(w):
    for i in range(len(w)):
        if w[i] == 0:
            w[i] = -1
    return w


def get_correct_word_RM(w, m):
    w_new = dot([w], H_m_i(1, m))
    for i in range(2, m + 1):
        w_new = dot(w_new, H_m_i(i, m))
    w_new = w_new[0]
    index_of_max = max(range(len(w_new)), key=w_new.__getitem__)
    W = list(map(int, bin(index_of_max)[2:].zfill(m)[::-1]))
    u = [0] * (m + 1)
    u[1:m + 1] = W
    if w_new[index_of_max] > 0:
        u[0] = 1

    return u


def ost(v):
    v = copy.copy(v)
    for i in range(len(v)):
        v[i] %= 2
    return v


print('Первая часть')
G = GoleyG()
print("\nG = ")
for k in range(0, len(G)):
    print(G[k])
H = GoleyH()
print("\nH = ")
for k in range(0, len(H)):
    print(H[k])

u = generate_U(H, B)
print("\nU = ")
print(u)

v = generate_V(u, G)
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

print('Вторая часть')
r = 1
m = 3
print("Код Рида-Маллера:(", r, ",", m, ")")
G = RM(r, m)
print("\nG = ")
for k in range(0, len(G)):
    print(G[k])
u = [1, 1, 0, 0]
print("Исходное слово\n", u)
w = ost(multiply_vector_matrix(u, G))
print("Закодированное слово:\n", w)
err = generate_error_vector(2 ** m, 1)
print("Однократная ошибка:\n", err)
word_with_error = ost(add_vectors(err, w))
print("Слово с однократной ошибкой:\n", word_with_error)
print("Исправленное слово:\n", get_correct_word_RM(word_with_error, m))
print("Исходное слово\n", u)
err = generate_error_vector(2 ** m, 2)
print("Двухкратнаяя ошибка:\n", err)
word_with_error = ost(add_vectors(err, w))
print("Слово с двухкратной ошибкой:\n", word_with_error)
print("Исправленное слово:\n", get_correct_word_RM(word_with_error, m))
r = 1
m = 4
print("Код Рида-Маллера:(", r, ",", m, ")")
G = RM(r, m)
print("\nG = ")
for k in range(0, len(G)):
    print(G[k])
u = [1, 1, 0, 0, 1]
print("Исходное слово\n", u)
w = ost(multiply_vector_matrix(u, G))
print("Закодированное слово:\n", w)
err = generate_error_vector(2 ** m, 1)
print("Однократная ошибка:\n", err)
word_with_error = ost(add_vectors(err, w))
print("Слово с однократной ошибкой:\n", word_with_error)
print("Исправленное слово:\n", get_correct_word_RM(word_with_error, m))
print("Исходное слово\n", u)
err = generate_error_vector(2 ** m, 2)
print("Двухкратнаяя ошибка:\n", err)
word_with_error = ost(add_vectors(err, w))
print("Слово с двухкратной ошибкой:\n", word_with_error)
print("Исправленное слово:\n", get_correct_word_RM(word_with_error, m))
print("Трёхкратная ошибка:\n", err)
print("Исходное слово\n", u)
err = generate_error_vector(2 ** m, 3)
word_with_error = ost(add_vectors(err, w))
print("Слово с трёхкратной ошибкой:\n", word_with_error)
print("Исправленное слово:\n", get_correct_word_RM(word_with_error, m))
print("Исходное слово\n", u)
err = generate_error_vector(2 ** m, 4)
word_with_error = ost(add_vectors(err, w))
print("Слово с четырёхкратной ошибкой:\n", word_with_error)
print("Исправленное слово:\n", get_correct_word_RM(word_with_error, m))