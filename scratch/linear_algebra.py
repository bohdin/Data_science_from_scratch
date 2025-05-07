from typing import List, Tuple, Callable
import math

Vector = List[float]
Matrix = List[List[float]]


def add(v: Vector, w: Vector) -> Vector:
    assert len(v) == len(w), "vectors must be the same length"

    return [v_i + w_i for v_i, w_i in zip(v, w)]


def subtract(v: Vector, w: Vector) -> Vector:
    assert len(v) == len(w), "vectors must be the same length"

    return [v_i - w_i for v_i, w_i in zip(v, w)]


def vector_sum(vectors: List[Vector]) -> Vector:
    num_elem = len(vectors[0])

    assert all(len(v) == num_elem for v in vectors), "different sizes"

    return [sum(vector[i] for vector in vectors) for i in range(num_elem)]


def scalar_multiply(c: float, v: Vector) -> Vector:
    return [c * v_i for v_i in v]


def vector_mean(vectors: List[Vector]) -> Vector:
    n = len(vectors)

    return scalar_multiply(1 / n, vector_sum(vectors))


def dot(v: Vector, w: Vector) -> float:
    assert len(v) == len(w), "vectors must be the same length"

    return sum(v_i * w_i for v_i, w_i in zip(v, w))


def sum_of_squares(v: Vector) -> float:

    return dot(v, v)


def magnitude(v: Vector) -> float:

    return math.sqrt(sum_of_squares(v))


def distance(v: Vector, w: Vector) -> float:

    return magnitude(subtract(v, w))


def shape(A: Matrix) -> Tuple[int, int]:
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0

    return num_rows, num_cols


def get_row(A: Matrix, i: int) -> Vector:
    return A[i]


def get_column(A: Matrix, j: int) -> Vector:
    return [A_i[j] for A_i in A]


def make_matrix(
    num_rows: int, num_cols: int, entry_fn: Callable[[int, int], float]
) -> Matrix:

    return [[entry_fn(i, j) for j in range(num_cols)] for i in range(num_rows)]


def identity_matrix(n: int) -> Matrix:

    return make_matrix(n, n, lambda i, j: 1 if i == j else 0)


if __name__ == "__main__":
    assert add([1, 2, 3], [4, 5, 6]) == [5, 7, 9], "add([1, 2, 3], [4, 5, 6]) == [5, 7, 9]"

    assert subtract([5, 7, 9], [4, 5, 6]) == [1, 2, 3], "subtract([5, 7, 9], [4, 5, 6]) == [1, 2, 3]"

    assert vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]]) == [16, 20], "vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]]) == [16, 20]"

    assert scalar_multiply(2, [1, 2, 3]) == [2, 4, 6], "scalar_multiply(2, [1, 2, 3]) == [2, 4, 6]"

    assert vector_mean([[1, 2], [3, 4], [5, 6]]) == [3, 4], "vector_mean([[1, 2], [3, 4], [5, 6]]) == [3, 4]"

    assert dot([1, 2, 3], [4, 5, 6]) == 32, "dot([1, 2, 3], [4, 5, 6]) == 32"

    assert sum_of_squares([1, 2, 3]) == 14, "sum_of_squares([1, 2, 3]) == 14"

    assert magnitude([3, 4]) == 5, "magnitude([3, 4]) == 5"

    assert distance([1, 2, 3], [4, 2, 7]) == 5, "distance([1, 2, 3], [4, 2, 7]) == 5"

    assert shape([[1, 2, 3], [4, 5, 6]]) == (2, 3), "shape([[1, 2, 3], [4, 5, 6]]) == (2, 3)"

    assert get_row([[1, 2, 3], [4, 5, 6]], 0) == [1, 2, 3], "get_row([[1, 2, 3], [4, 5, 6]], 0) == [1, 2, 3]"

    assert get_column([[1, 2, 3], [4, 5, 6]], 2) == [3, 6], "get_column([[1, 2, 3], [4, 5, 6]], 2) == [3, 6]"

    assert make_matrix(3, 3, lambda i, j: i + j) == [[0, 1, 2], [1, 2, 3], [2, 3, 4]], "assert make_matrix(3, 3, lambda i, j: i + j) == [[0, 1, 2], [1, 2, 3], [2, 3, 4]]"

    assert identity_matrix(5) == [
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
], "identity_matrix(5) == [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1],]"
