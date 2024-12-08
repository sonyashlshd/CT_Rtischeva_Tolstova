import copy


def isZero(matrix_row, start_index, end_index):
    return all(matrix_row[i] == 0 for i in range(start_index, end_index))


def REF(s):
    matrix = [copy.copy(row) for row in s]
    num_rows = len(matrix)
    num_cols = len(matrix[0])

    for col in range(num_cols):
        for row in range(num_rows):
            if matrix[row][col] == 1 and isZero(matrix[row], 0, col):
                for other_row in range(num_rows):
                    if matrix[other_row][col] == 1 and other_row != row and isZero(matrix[other_row], 0, col):
                        for j in range(num_cols):
                            matrix[other_row][j] ^= matrix[row][j]
    return matrix


def PREF(s):
    matrix = [copy.copy(row) for row in s]
    num_rows = len(matrix)
    num_cols = len(matrix[0])

    for col in range(num_cols):
        for row in range(num_rows):
            if matrix[row][col] == 1 and isZero(matrix[row], 0, col):
                for other_row in range(num_rows):
                    if matrix[other_row][col] == 1 and other_row != row:
                        for j in range(num_cols):
                            matrix[other_row][j] ^= matrix[row][j]
    return matrix


def remove_zero_rows(matrix):
    return [row for row in matrix if not isZero(row, 0, len(row))]


class LinearCode:
    def __init__(self):
        self.__S = []

    @property
    def S(self):
        return remove_zero_rows(self.__S)

    @S.setter
    def S(self, value):
        self.__S = value

    def S_REF(self):
        return remove_zero_rows(REF(self.S))

    def S_PREF(self):
        return remove_zero_rows(PREF(self.S))

    def n(self):
        return len(self.S[0])

    def k(self):
        return len(self.S)

    def lead_indices(self):
        leads = []
        for col in range(len(self.S[0])):
            for s in self.S_PREF():
                if s[col] == 1 and isZero(s, 0, col):
                    leads.append(col)
                    break
        return leads

    def X(self):
        X_matrix = self.S_PREF()
        leads = self.lead_indices()
        for i in range(len(X_matrix)):
            for j in leads:
                X_matrix[i].pop(j - leads.index(j))
        return X_matrix

    def H(self):
        X = self.X()
        E = []
        lead = self.lead_indices()
        H = []
        for j in range(0, len(X[0])):
            r = []
            for i in range(0, len(X[0])):
                if i == len(E):
                    r.append(1)
                else:
                    r.append(0)
            E.append(r)
        xi = 0
        ii = 0
        for i in range(0, len(X) + len(X[0])):
            if i == lead[xi]:
                H.append(X[xi])
                if xi < len(lead) - 1:
                    xi += 1
            else:
                H.append(E[ii])
                if ii < len(E) - 1:
                    ii += 1
        return H

    def U(self):
        U_set = []
        G_matrix = self.S_REF()

        for j in range(len(G_matrix[0])):
            u_vector = [G_matrix[i][j] for i in range(len(G_matrix))]
            if not located(u_vector, U_set):
                U_set.append(u_vector)

        flag = True
        while flag:
            flag = False
            new_combinations = []
            for i in range(len(U_set)):
                for j in range(i + 1, len(U_set)):
                    combined_vector = sum_vectors(U_set[i], U_set[j])
                    if not located(combined_vector, U_set) and combined_vector not in new_combinations:
                        new_combinations.append(combined_vector)
                        flag = True
            U_set.extend(new_combinations)

        return U_set

    def V(self):
        U_set = self.U()
        G_matrix = self.S_REF()

        V_set = [vector_multiply(u_vector, G_matrix) for u_vector in U_set]

        return V_set

    def d(self):
        V_set = self.V()
        min_distance = float('inf')

        for i in range(len(V_set)):
            for k in range(i + 1, len(V_set)):
                distance = sum(v1 != v2 for v1, v2 in zip(V_set[i], V_set[k]))
                min_distance = min(min_distance, distance)

        return min_distance

    def t(self):
        return self.d() - 1

    def e2(self, indexV):
        v_vector = self.V()[indexV]
        H_matrix = self.H()

        # Try to find a double error vector e2
        for i in range(len(v_vector)):
            for j in range(i + 1, len(v_vector)):
                e2_vector = [0] * len(v_vector)
                e2_vector[i] = 1
                e2_vector[j] = 1

                ve2_vector = sum_vectors(v_vector, e2_vector)
                ve2H_result = vector_multiply(ve2_vector, H_matrix)

                if all(x == 0 for x in ve2H_result):  # Check if ve2H is zero vector
                    return e2_vector


def sum_vectors(v1, v2):
    return [(x + y) % 2 for x, y in zip(v1, v2)]


def located(vector, matrix):
    return any(all(vector[j] == row[j] for j in range(len(vector))) for row in matrix)


def vector_multiply(vector, matrix):
    vM = []
    for i in range(0, len(matrix[0])):
        c = 0
        for j in range(0, len(matrix)):
            c += matrix[j][i] * vector[j]
        vM.append(c % 2)
    return vM


LC = LinearCode()
LC.S = [[1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0],
        [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]]

print("S:")
for row in LC.S:
    print(row)

print("G (REF):")
for row in LC.S_REF():
    print(row)

print("n =", LC.n())
print("k =", LC.k())

print("G* (PREF):")
for row in LC.S_PREF():
    print(row)

print("Lead indices:", LC.lead_indices())
print("X:")
for row in LC.X():
    print(row)

print("H:")
for row in LC.H():
    print(row)

print("U:")
for row in LC.U():
    print(row)

print("V:")
for row in LC.V():
    print(row)

v = LC.V()[0]
print()
print("v = ")
print(v)

vH = vector_multiply(v, LC.H())
print()
print("vH = ")
print(vH)
print()

print("Minimum distance d =", LC.d())
print("t =", LC.t())

e1 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
print("e1 =", e1)
ve1 = sum_vectors(e1, v)
print("v + e1 =", ve1)
print("(v + e1)@H =", vector_multiply(ve1, LC.H()))

print()
e2 = LC.e2(0)
print("e2 =", e2)
ve2 = sum_vectors(e2, v)
print("v + e2 =", ve2)
print("(v + e2)@H =", vector_multiply(ve2, LC.H()))