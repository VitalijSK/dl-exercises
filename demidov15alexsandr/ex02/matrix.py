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
            self.array = [[0]*m for x in range(n)]
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
        s='\n'.join([' '.join([str(item) for item in row]) for row in self.array])
        return s + '\n'

    def getRank(self):
        """Получить число строк и столбцов
        """
        return (self.n, self.m)


    def __eq__(self, mat):
        """ Проверка на равенство """

        return (mat.array == self.array)

    def transpose(self):
        matTrans = Matrix(self.m, self.n)
        for i in range(self.n):
            for j in range(self.m):
                matTrans.array[j][i] = self.array[i][j]
        return matTrans

    def __add__(self, mat):
        """ Переопределение операции сложения "+"
        для матриц
        """
        if self.getRank() != mat.getRank():
            raise MatrixError("Trying to add matrixes of varying rank!")
        else:
            matAdd = Matrix(self.n, self.m)
            for i in range(self.n):
                for j in range(self.m):
                    matAdd.array[i][j] = self.array[i][j] + mat.array[i][j]
            return matAdd
        

    def __sub__(self, mat):
        """ Переопределение операции вычитания "-"
        для матриц
        """
        if self.getRank() != mat.getRank():
            raise MatrixError("Trying to add matrixes of varying rank!")

        else:
            matSub = Matrix(self.n, self.m)
            for i in range(self.n):
                for j in range(self.m):
                    matSub.array[i][j] = self.array[i][j] - mat.array[i][j]
            return matSub

    def __mul__(self, mat):
        """Произведение Адамара или поточечное умножение"""
        mulmat = Matrix(self.n, self.m) # результирующая матрица

        # если второй аргумент - число, то 
        # просто умножить каждый элемент на это число
        if isinstance(mat, int) or isinstance(mat, float):
            for i in range(self.n):
                for j in range(self.m):
                    mulmat[i][j] = self.array[i][j]*mat
            return mulmat
        else:
            # для поточечного перемножения матриц  
            # их размерности должны быть одинаковыми
            if (self.n != mat.n or self.m != mat.m):
                raise MatrixError("Matrices cannot be multipled!")
                
            for i in range(self.n):
                for j in range(self.m):
                    mulmat[i][j] = self.array[i][j]*mat[i][j]
            return mulmat

    def dot(self, mat):
        """ Матричное умножение """
        
        matn, matm = mat.getRank()
        
        
        # для перемножения матриц число столбцов одной 
        # должно равняться числу строк в другой
        if (self.m != matn):
            raise MatrixError("Matrices cannot be multipled!")
        else:
            dotmat = Matrix(self.n, matm)
            for i in range(self.n):
                for j in range(matm):
                    for g in range(matn):
                        dotmat.array[i][j] += self.array[i][g]*mat[g][j]
            return dotmat

    @classmethod
    def _makeMatrix(cls, array):
        """Переопределение конструктора
        """
        n = len(array)
        m = len(array[0])
        # Validity check
        if any([len(row) != m for row in array[1:]]):
            raise MatrixError("inconsistent row length")
        mat = Matrix(n,m, init=False)
        mat.array = array

        return mat

    @classmethod
    def fromList(cls, listoflists):
        """ Создание матрицы напрямую из списка """

        # E.g: Matrix.fromList([[1 2 3], [4,5,6], [7,8,9]])

        array = listoflists[:]
        return cls._makeMatrix(array)

    @classmethod
    def makeId(cls, n):
        """ Создать единичную матрицу размера (nxn) """

        array = [[0]*n for x in range(n)]
        idx = 0
        for row in array:
            row[idx] = 1
            idx += 1

        return cls.fromList(array)


if __name__ == "__main__":
    a = Matrix.fromList([[1,2], [3, 4]])
    print(a)
    print(a*2)
    print('Identity:')
    print(Matrix.makeId(3))
    print(a*a)
    pass


