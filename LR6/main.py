import numpy as np
import random


def generate_random_bits(length):
    """Генерирует случайный битовый массив заданной длины."""
    return np.random.randint(0, 2, length)


def polynomial_multiply(a, b):
    """Умножает два полинома в GF(2)."""
    result = np.zeros(len(a) + len(b) - 1, dtype=int)
    for i in range(len(a)):
        for j in range(len(b)):
            result[i + j] ^= a[i] & b[j]  # Сложение по модулю 2
    return result


def encode_with_errors(generating_poly, message_length, error_count):
    """Кодирует сообщение с заданным количеством ошибок."""
    input_bits = generate_random_bits(message_length)
    print("Исходное сообщение:", input_bits)

    encoded_message = polynomial_multiply(input_bits, generating_poly) % 2

    for _ in range(error_count):
        error_position = random.randint(0, len(encoded_message) - 1)
        encoded_message[error_position] ^= 1  # Вводим ошибку
    return encoded_message


def create_error_packet(total_length, packet_size):
    """Создает пакет ошибок заданного размера."""
    error_packet = np.zeros(total_length, dtype=int)
    start_index = random.randint(0, total_length - packet_size)

    for i in range(packet_size):
        error_packet[(start_index + i) % total_length] = random.randint(0, 1)

    print('Сгенерированный пакет ошибок:', error_packet)
    return error_packet


def is_error_within_limit(error_vector, limit):
    """Проверяет, есть ли ошибки в пределах заданного лимита."""
    trimmed_error = np.trim_zeros(error_vector)
    return len(trimmed_error) <= limit and len(trimmed_error) != 0


def encode_with_error_packet(generating_poly, message_length, packet_size):
    """Кодирует сообщение с пакетом ошибок."""
    input_bits = generate_random_bits(message_length)
    print("Исходное сообщение:", input_bits)

    encoded_message = polynomial_multiply(input_bits, generating_poly) % 2
    return encoded_message + create_error_packet(len(encoded_message), packet_size)


def decode_message(generating_poly, max_errors, received_message, is_packet):
    """Декодирует полученное сообщение."""
    n = len(received_message)

    remainder = np.polydiv(received_message, generating_poly)[1] % 2
    for i in range(n):
        error_vector = np.zeros(n, dtype=int)
        error_vector[n - i - 1] = 1

        multiplied_result = polynomial_multiply(remainder.astype(int), error_vector.astype(int)) % 2
        syndrome = np.polydiv(multiplied_result, generating_poly)[1] % 2

        if is_packet:
            if is_error_within_limit(syndrome, max_errors):
                correction_vector = np.zeros(n, dtype=int)
                correction_vector[i] = 1
                corrected_message = polynomial_multiply(correction_vector.astype(int), syndrome.astype(int)) % 2

                # Ensure both arrays are of the same length before addition
                updated_length = max(len(received_message), len(corrected_message))
                updated_received = np.pad(received_message, (0, updated_length - len(received_message)), 'constant')
                updated_corrected = np.pad(corrected_message, (0, updated_length - len(corrected_message)), 'constant')

                updated_message = (updated_received + updated_corrected) % 2

                decoded_result = np.polydiv(updated_message.astype(int), generating_poly)[0] % 2
                padding = [0] * (len(generating_poly) - len(decoded_result))
                return np.concatenate([padding, decoded_result]).astype(int)  # Приведение к целому типу
        else:
            if sum(syndrome) <= max_errors:
                correction_vector = np.zeros(n, dtype=int)
                correction_vector[i] = 1
                corrected_message = polynomial_multiply(correction_vector.astype(int), syndrome.astype(int)) % 2

                # Ensure both arrays are of the same length before addition
                updated_length = max(len(received_message), len(corrected_message))
                updated_received = np.pad(received_message, (0, updated_length - len(received_message)), 'constant')
                updated_corrected = np.pad(corrected_message, (0, updated_length - len(corrected_message)), 'constant')

                updated_message = (updated_received + updated_corrected) % 2

                decoded_result = np.polydiv(updated_message.astype(int), generating_poly)[0] % 2
                padding = [0] * (len(generating_poly) - len(decoded_result))
                return np.concatenate([padding, decoded_result]).astype(int)  # Приведение к целому типу

    return None


if __name__ == '__main__':
    # Настройки для кода (7,4)
    n1 = 7
    k1 = 4
    t1 = 1
    g1_poly = np.array([1, 0, 1, 1])  # g(x) для кода (7,4)

    print("Порождающий многочлен g1:", g1_poly)

    for error_count in range(1, 4):
        print(f"\nОшибок: {error_count}")
        encoded_with_errors_74 = encode_with_errors(g1_poly, k1 - 1, error_count)
        decoded_74 = decode_message(g1_poly, t1, encoded_with_errors_74.astype(int), False)
        print("Декодированное сообщение:", decoded_74)

    # Настройки для кода (15,9)
    n2 = 15
    k2 = 9
    t2 = 3
    g2_poly = np.array([1, 0, 0, 0, 1, 1])  # g(x) для кода (15.9), замените на правильный многочлен

    print("\nПорождающий многочлен g2:", g2_poly)

    for packet_size in range(1, 5):
        print(f"\nОшибок: {packet_size}")
        encoded_with_packet_errors_159 = encode_with_error_packet(g2_poly, k2 - 1, packet_size)
        decoded_159 = decode_message(g2_poly, packet_size, encoded_with_packet_errors_159.astype(int), True)
        print("Декодированное сообщение:", decoded_159)