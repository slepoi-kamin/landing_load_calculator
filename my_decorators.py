"""
Created on 15:11:16 29.10.2020.

@author: ulyanovas
"""
import math
import time
from types import FunctionType


class LogTime:
    """
    Класс декоратор. Определяет время работы декорированных функций. Для
    использования необходима функция обертка, которая будет передавать ключ
    словаря и функцию при создании экземпляра класса.
    """
    __percent_times = {}
    __run_times = {}
    __norm_times = {}

    @classmethod
    def get_time(cls, par='norm'):
        if par == '%' or par == 'percent':
            return cls.__percent_times
        elif par == 'time':
            return cls.__run_times
        elif par == 'norm' or par == '1':
            return cls.__norm_times

    @classmethod
    def __update_class_dicts(cls, new_key, new_time):
        # Заполнение словаря со значениями времени работы функции
        cls.__run_times[f'{new_key}'] = new_time

        max_time = max(cls.__run_times.values())
        summ_times = math.fsum(cls.__run_times.values())

        # Словарь с нормированными значениями
        cls.__norm_times = {item[0]: item[1] / max_time
                            for item in cls.__run_times.items()}

        # Словарь с процентными значениями
        cls.__percent_times = {item[0]: item[1] / summ_times
                               for item in cls.__run_times.items()}

    def __init__(self, func, new_key):
        self.run_count = 0  # Кол-во обращений к экземпляру класса
        self.runs_time = []
        self.func = func  # Функция
        self.name = self.__check_key(new_key)

    def __check_key(self, new_key):
        return (f'Run time for {new_key}'
                if new_key else f'Run time for {self.func}')

    def __call__(self, *args, **kwargs):
        self.run_count += 1
        start = time.time()
        self.res = self.func(*args, **kwargs)
        finish = time.time()
        self.__append_runs_time(start, finish)

        LogTime.__update_class_dicts(f'{self.name}__run_{self.run_count}',
                                     self.runs_time[-1])
        return self.res

    def __append_runs_time(self, start, stop):
        curr_time = stop - start
        self.runs_time.append(curr_time if curr_time != 0 else 1e-16)


def logtime(key=None):
    """
    Функция декоратор - обертка для класса декоратора. Декоратор вычисляет
        время работы функции и сохраняет эти результаты в классе LogTime в
        виде словаря.
    :param key: ключ для словаря.
    :return: Экземпляр класса LogTime.
    """
    if type(key) == FunctionType:
        func = key
        new_key = None
        return LogTime(func, new_key)
    else:
        def wrapper(function):
            return LogTime(function, key)
        return wrapper