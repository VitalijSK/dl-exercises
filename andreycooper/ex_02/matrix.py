import random
import operator
import sys


class MatrixError(BaseException):
    """ Класс исключения для матриц """
    pass


class Matrix(object):
    """Простой класс матрицы в Python
    Основные операции линейной алгебры реализованы
    путем перегрузки операторов 
    """

    def __init__(self, n, m, init=True):
        """Конструктор

        #Аргументы:
            n    :  int, число строк
            m    :  int, число столбцов 
            init :  (необязательный параметр), логический.
                    если False, то создается пустой массив
        """
        if init:
            # создаем массив нулей
            self.array = [[0] * m for x in range(n)]
        else:
            self.array = []

        self.n = n
        self.m = m

    def __getitem__(self, idx):
        """Перегрузка оператора получения элемента массива 
        """
        # проверяем, если индекс - это список индексов
        if isinstance(idx, tuple):
            if len(idx) == 2:
                return self.array[idx[0]][idx[1]]
            else:
                # у матрицы есть только строки и столбцы
                raise MatrixError("Matrix has only two shapes!")
        else:
            return self.array[idx]

    def __setitem__(self, idx, item):
        """Перегрузка оператора присваивания 
        """
        # проверяем, если индекс - это список индексов
        if isinstance(idx, tuple):
            if len(idx) == 2:
                self.array[idx[0]][idx[1]] = item
            else:
                # у матрицы есть только строки и столбцы
                raise MatrixError("Matrix has only two shapes!")
        else:
            self.array[idx] = item

    def __str__(self):
        """Переопределяем метод вывода матрицы в консоль
        """
        s = '\n'.join([' '.join([str(item) for item in row]) for row in self.array])
        return s + '\n'

    def get_rank(self):
        """Получить число строк и столбцов
        """
        return self.n, self.m

    def __eq__(self, mat):
        """ Проверка на равенство """

        return mat.array == self.array

    def transpose(self):
        """ Транспонированное представление матрицы 
            Aij = Aji
        """
        transposed_matrix = Matrix(self.m, self.n)

        for i in range(self.n):
            for j in range(self.m):
                transposed_matrix[j][i] = self[i][j]

        return transposed_matrix

    def __sum(self, matrix, addition):
        if self.get_rank() != matrix.get_rank():
            raise MatrixError("Trying to add matrices of varying rank!")

        result_matrix = Matrix(self.n, self.m)

        for i in range(self.n):
            for j in range(self.m):
                result_matrix[i][j] = addition(self[i][j], matrix[i][j])

        return result_matrix

    def __add__(self, mat):
        """ Переопределение операции сложения "+"
        для матриц
        Cij = Aij+Bij
        """

        return self.__sum(mat, lambda x, y: x + y)

    def __sub__(self, mat):
        """ Переопределение операции вычитания "-"
        для матриц
        Cij = Aij-Bij
        """

        return self.__sum(mat, lambda x, y: x - y)

    def __mul__(self, mat):
        """Произведение Адамара или поточечное умножение"""
        mulmat = Matrix(self.n, self.m)  # результирующая матрица

        # если второй аргумент - число, то 
        # просто умножить каждый элемент на это число
        if isinstance(mat, int) or isinstance(mat, float):
            for i in range(self.n):
                for j in range(self.m):
                    mulmat[i][j] = self.array[i][j] * mat
            return mulmat
        else:
            # для поточечного перемножения матриц  
            # их размерности должны быть одинаковыми
            if self.n != mat.n or self.m != mat.m:
                raise MatrixError("Matrices cannot be multiplied!")

            for i in range(self.n):
                for j in range(self.m):
                    mulmat[i][j] = self.array[i][j] * mat[i][j]
            return mulmat

    def dot(self, matrix):
        """
        Матричное умножение
        Cij = sum(Aik*Bkj)
        """

        matrix_n, matrix_m = matrix.get_rank()

        # для перемножения матриц число столбцов одной 
        # должно равняться числу строк в другой
        if self.m != matrix_n:
            raise MatrixError("Matrices cannot be multiplied!")

        result_matrix = Matrix(self.n, matrix.m)

        for i in range(result_matrix.n):
            for j in range(result_matrix.m):
                result_matrix[i][j] = sum(self[i][k] * matrix[k][j] for k in range(self.m))

        return result_matrix

    @classmethod
    def _make_matrix(cls, array):
        """Переопределение конструктора
        """
        n = len(array)
        m = len(array[0])
        # Validity check
        if any([len(row) != m for row in array[1:]]):
            raise MatrixError("inconsistent row length")
        mat = Matrix(n, m, init=False)
        mat.array = array

        return mat

    @classmethod
    def fromList(cls, list_of_lists):
        """ Создание матрицы напрямую из списка """

        # E.g: Matrix.from_list([[1 2 3], [4,5,6], [7,8,9]])

        array = list_of_lists[:]
        return cls._make_matrix(array)

    @classmethod
    def makeId(cls, n):
        """ Создать единичную матрицу размера (nxn) """

        array = [[0] * n for x in range(n)]
        idx = 0
        for row in array:
            row[idx] = 1
            idx += 1

        return cls.fromList(array)


if __name__ == "__main__":
    a = Matrix.fromList([[1, 2], [3, 4]])
    print(a)
    print(a * 2)
    print('Identity:')
    print(Matrix.makeId(3))
    print(a * a)
    b = Matrix.fromList([[6, 4, 24], [1, -9, 8]])
    print("B: \n", b)
    print("Transposed: \n", b.transpose())
    c = Matrix.fromList([[1, 1], [1, 1]])
    print("Sum a + c: \n", a + c)
    print("Subtraction a - c: \n", a - c)
    pass
