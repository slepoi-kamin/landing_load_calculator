# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 14:15:49 2020

@author: ulyanovas
"""
import math
import pathlib
from types import FunctionType
from typing import Dict, Any, List, NoReturn, AnyStr

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import matplotlib as mpl
from astropy import units as u
from astropy.constants import g0
import pandas as pd
import numpy as np
from pandas import DataFrame
from scipy.signal import butter, filtfilt, freqz
import os
import time

# %% Разные функции

def get_time():
    start_time = time.time()
    percent_time = {}
    time_fr_start = []

    def norm_time(new_key):
        time_fr_start.append(time.time() - start_time)
        if len(time_fr_start) > 1:
            for key in percent_time:
                percent_time[key] = percent_time[key] * time_fr_start[-2]
            percent_time[new_key] = time_fr_start[-1] - time_fr_start[-2]
            for key in percent_time:
                percent_time[key] = percent_time[key] / time_fr_start[-1]
        else:
            percent_time[new_key] = time_fr_start[-1] / time_fr_start[-1]
        return percent_time
    return norm_time



def df_from_text(text: str, start: str):

    start_index = text.index(start)
    end_index = text.index('[END_TABLE]', start_index)
    data = [text[i] for i in range(start_index + 1, end_index)]
    data_table = [data[i].split('\t') for i in range(2, len(data))]
    data_table = replace_lst(data_table, '', '0')
    col = [data[i].split('\t') for i in range(2)]
    multi_col = pd.MultiIndex.from_arrays(np.array(col))
    df = pd.DataFrame(data_table, columns=multi_col)
    return df


def list_from_text(text: str, start: str):
    start_index = text.index(start)
    end_index = text.index('[END_TABLE]', start_index)
    data = [text[i] for i in range(start_index + 1, end_index)]
    data = [data[i].split('\t') for i in range(len(data))]
    return data


def set_data_types(data, dtypes, par='data'):
    """
    Задает тип данных в data в соответсвии с массивом dtypes.

    Parameters
    ----------
    data : Список значений
        DESCRIPTION.
    dtypes : Список типов данных.
        Типы данных применяются к списку data по строкам.
    par : Строковый параметр
        DESCRIPTION. The default is 'data'. - Стандартная конвертация типов
        данных списка data. Любое другое обозначение - формирование списка
        множителей для списка data

    Returns
    -------
    data2 : Список с форматированными значениями или множителями.

    """
    data2 = []

    if par == 'data':
        for i in range(len(dtypes)):
            if dtypes[i] == 'str':
                data2.append([str(x) for x in data[i]])
            elif dtypes[i] == 'int':
                data2.append([int(x) for x in data[i]])
            else:
                data2.append([float(x) for x in data[i]])
        return data2
    else:
        for i in range(len(dtypes)):
            if dtypes[i] == 'str':
                data2.append(int(data[i]))
            else:
                data2.append(float(data[i]))
        return data2


def unit_assign(str_unit):
    """
    Определение единиц измерения при помощи модуля astropy.units.

    Parameters
    ----------
    str_unit : Строка с единицами измерения
        Поддерживаются: m, mm, cm, km, N, kgf, tf, g, kg, t, deg, rad, s, min,
        none, *, /, string

    Returns
    -------
    TYPE
        1) Единичная величина с единицами измерения модуля astropy.units
        2) Строковое обозначения типа данных для соответсвующей величины
    """
    assert str_unit != str, 'Тип данных аргумента должен быть String'

    # Создание дополнительных единиц силы - кгс и тс
    kgf = g0.value * u.newton
    tf = 1000 * kgf

    # Проверка знака деления
    if str_unit.count('/') != 0:
        str_unit_1 = str_unit[: str_unit.index('/')]
        str_unit_2 = str_unit[str_unit.index('/') + 1:]
        return (unit_assign(str_unit_1)[0] /
                unit_assign(str_unit_2)[0]), 'float'

    # Проверка знака умножения
    elif str_unit.count('*') != 0:
        str_unit_1 = str_unit[: str_unit.index('*')]
        str_unit_2 = str_unit[str_unit.index('*') + 1:]
        return (unit_assign(str_unit_1)[0] *
                unit_assign(str_unit_2)[0]), 'float'

    # Ньютоны
    elif str_unit == "N":
        return 1 * u.newton, 'float'

    # Единицы силы
    elif str_unit == 'kgf' or str_unit == 'tf':
        if str_unit == 'kgf':
            return 1 * kgf, 'float'
        else:
            return 1 * tf, 'float'

    # Единицы длины
    elif str_unit == 'm' or str_unit == 'mm' \
            or str_unit == 'cm' or str_unit == 'km':
        return u.Quantity(str(1) + str_unit), 'float'

    # Единицы массы
    elif str_unit == 'g' or str_unit == 'kg' or str_unit == 't':
        return u.Quantity(str(1) + str_unit), 'float'

    # Единицы измерения угла
    elif str_unit == 'deg' or str_unit == 'rad':
        return u.Quantity(str(1) + str_unit), 'float'

    # Единицы измерения времени
    elif str_unit == 's' or str_unit == 'min':
        return u.Quantity(str(1) + str_unit), 'float'

    # Безразмерные единицы
    elif str_unit == 'none':
        return 1 * u.dimensionless_unscaled, 'float'

    # Строка (string)
    elif str_unit == 'string':
        return 1 * u.dimensionless_unscaled, 'str'


def units_to_SI(units, data):
    """
    Конвертация списка исходных данных в систему СИ.

    Parameters
    ----------
    units : Список строк
        Список содержащий строковые обозначения единиц измерения
    data : Список различных значений
        Комбинированный список. В систему си переводятся только столбцы с
        физическими единицами измерения

    Returns
    -------
    data_SI : Список различных значений
        Значения списка конвертируются в float и string

    """
    # Определение единиц измерения столбцов
    units_old = [unit_assign(x)[0] for x in units]
    # Определение типов данных
    data_types = [unit_assign(x)[1] for x in units]

    # формирование массива коэффициентов перехода к системе СИ
    units_koef = [(x.value * (x.si).value / x.value) for x in units_old]
    units_koef = set_data_types(units_koef, data_types, par='koef')

    # Формирование массива данных в системе СИ, транспонирование списка
    data_SI = list(map(list, zip(*data)))
    data_SI = set_data_types(data_SI, data_types)

    for i in range(len(units_koef)):
        data_SI[i] = [x * units_koef[i] for x in data_SI[i]]
    # Обрантое транспонирование списка
    data_SI = list(map(list, zip(*data_SI)))

    return data_SI


def units_from_SI(df1: pd.DataFrame, units: list):
    """
    Переводит значения в Таблице пандас в требуюмую систему единиц измерения.

    Parameters
    ----------
    df1 : pd.DataFrame
        Таблица значений.
    units : list
        список требуемых единиц измерений по столбцам таблицы.

    Returns
    -------
    df1 : pd.DataFrame
        Таблица пандас в требуемых единицах измерений с добавленными
        подписями единиц измерения.

    """
    # Список единиц измерения в формате astropy.units
    units_old = [unit_assign(x)[0] for x in units]
    # Определение типов данных
    data_types = [unit_assign(x)[1] for x in units]

    # Коэффициенты перехода к требуемой системе единиц
    units_koef = [(x.value * (x.si).value / x.value)**(-1) for x in units_old]
    units_koef = set_data_types(units_koef, data_types, par='koef')

    # Формирование Мультииндекса для Таблицы
    if df1.columns.nlevels == 1:  # Если нет мультииндекса
        col_names = [list(df1.columns), units]
        col_names = list(map(list, zip(*col_names)))  # Транспонирование списка
    else:  # Если Мультииндекс
        col_names = []
        tmp = list(df1.columns)
        for i in range(len(tmp)):
            col_names.append([x for x in tmp[i]] + [units[i]])

    col_names = pd.MultiIndex.from_tuples(col_names)  # Мультииндекс

    if 'str' in data_types:
        tmp_lst = df1.values.tolist()
        tmp_lst2 = []
        for x in tmp_lst:
            tmp_lst2.append([x[i] * units_koef[i] for i in range(len(x))])
        df1 = pd.DataFrame(tmp_lst2, columns=col_names)
    else:
        df1 = pd.DataFrame(
            df1.values * np.array(units_koef),
            columns=col_names)  # Создание Таблицы

    return df1


def df_to_SI(df1):
    cols = np.array(df1.columns.to_numpy().tolist()).T.tolist()
    data = df1.values.tolist()
    data = units_to_SI(cols[0], data)
    df1 = pd.DataFrame(data, columns=cols[1])
    return df1


def units_from_names(data_names: list):
    """
    Формирует список единиц измерения по списку иемн столбцов.

    Parameters
    ----------
    data_names : list
        Список содержащий имена столбцов.

    Returns
    -------
    units : List
        Список, содержащий единицы измерения для столбцов.

    """
    units = []  # Пустой список для единиц измерения
    for name in data_names:
        # Если в имени столбца есть 'S', то единицы измерения - мм
        if 'S' in name or 's_' in name:
            units.append('mm')
        # Если в имени столбца есть 'P', то единицы измерения - т*с
        elif 'P' in name or 'f_wh' in name:
            units.append('tf')
        else:  # Иначе None
            units.append(None)
    return units


def gear_data_to_SI(all_data: [pd.DataFrame]):
    """
    Изменяет входной список с таблицами, \
    Конвертирует таблицы в отчетнуюсистему координат.

    Parameters
    ----------
    all_data : [pd.DataFrame]
        Список таблиц.

    Returns
    -------
    None.

    """
    if all_data == []:  # Если список пуст
        return
    for i in range(len(all_data)):
        # Список единиц измерения по первой строке названий столбцов
        units = units_from_names(
            all_data[i].columns.get_level_values(0).tolist())
        # Перевод в отчетную систему единиц
        all_data[i] = units_from_SI(all_data[i], units)


def replace_lst(lst, value, new_val):
    """
    Замена всех найденных значений списка на другое.

    Parameters
    ----------
    lst : Список
        Исходный список
    value : Значение
        Искомое значение
    new_val : Значение
        Итоговое значение

    Returns
    -------
    lst : Список
        Исправленный список с новыми значениями

    """
    for row in lst:
        for i, val in enumerate(row):
            if val == value:
                row[i] = new_val
    return lst


def clear_duplicates_bycol(data, n_col=0):
    """
    Функция фильтрующая строки с повторяющимися в столбце значениями.

    Parameters
    ----------
    data : Список
        Исходный список
    n_col : номер столбца, optional
        Номер столбца, в котором производится поиск повторяющихся значений.
        The default is 0.

    Returns
    -------
    clear_data : Список
        Отфильтрованный список.

    """
    clear_data = []
    for i in range(len(data) - 1):
        if data[i][n_col] == data[i + 1][n_col]:
            pass
        else:
            clear_data.append(data[i])
    clear_data.append(data[len(data) - 1])

    return clear_data


def import_adss_as_df(f_name):
    """
    Функция импорта файла результатов (spreadshit) АДАМС и преобразования \
        его в data frame pandas.

    Parameters
    ----------
    f_name : String
        Строка с именем файла ADAMS spreadsheet.

    Returns
    -------
    res_data : pandas data frame
        Блок данных пандас. имена столбцов  - названия измерителей в файле.

    """
    # Проверка типа данных аргумента
    assert f_name != str, 'Тип данных аргумента должен быть String'

    # Чтение файла
    with open(f_name, 'r') as tfile:
        text = (tfile.read()).split('\n')  # Разделение переносом строки

    # Удаление пустых строк
    text = [x for x in text if x != '']

    # Формирование списка с именами столбцов
    # Выделение строк не содержащих числовые данные
    col_id = [x for x in text if (x[0] == '"' or x[0] == '\t')]
    col_id = col_id[len(col_id) - 1]  # Удаление всех строк кроме последней
    col_id = col_id.split('\t')  # Разделение табуляторами
    col_id = [x.replace('"', '') for x in col_id]  # Удаление кавычек

    # Формирование списка с числовыми данными
    # Выделение строк содержащих числовые значения, разделение табуляторами
    data = [x.split('\t') for x in text if (x[0] != '"' and x[0] != '\t')]
    # Преобразование в массив numpy, с двойной точностью
    data = np.array(data, dtype='float64')
    # Фильтрация по первому столбцу
    data = np.array(clear_duplicates_bycol(data))

    # Преобразование в data frame pandas
    res_data = pd.DataFrame(data, columns=col_id, dtype=float)

    return res_data


def ad_res_filter(me_names, adams_res):
    """
    Переименовывает столбцы в adams_res в соответсвиии с именами \
    в me_names и удаляет лишние столбцы из adams_res.

    Parameters
    ----------
    me_names : List
        Список имен измерителей.
    adams_res : pandas data frame
        Массив всех результатов из Адамс.

    Returns
    -------
    filtered_adams_res : pandas data frame
        Отфильтрованный массив с новыми именами колонок.
    """
    tmp_names = list(me_names)
    for me in adams_res.columns:
        for name in tmp_names:
            if name in me:
                me = name
                tmp_names.remove(name)
                break

    tmp_names = list(me_names)
    filtered_adams_res = adams_res[tmp_names]

    return filtered_adams_res


def maxmin_of_df(df1: pd.DataFrame, df2: pd.DataFrame, pmax=True):
    """
    Объединяет две таблицы в одну, оставляя наибольшие или наименьшие \
        значения соответствующих ячеек.

    Parameters
    ----------
    df1 : pd.DataFrame
        Первая таблица.
    df2 : pd.DataFrame
        Вторая таблица.
    pmax : bool, optional
        Параметр определения максимальных или минимальных значений.
        The default is True.

    Returns
    -------
    pd.DataFrame
        Объединенная таблица.

    """
    if df1 is None and df2 is None:
        return None
    elif df1 is None:
        return df2
    elif df2 is None:
        return df1
    else:
        df2.columns = df1.columns
        return pd.concat([df1, df2]).max(
            level=0) if pmax else pd.concat([df1, df2]).min(level=0)


def clear_none_from_dict(dict1: dict):
    """
    Удаляет значения None из словаря. Модифицирует исходный словарь.

    Parameters
    ----------
    dict1 : dict
        Обрабатываемый словарь.

    Returns
    -------
    None.

    """
    del_key = []
    for key in dict1:
        if dict1[key] is None:
            del_key.append(key)

    for key in del_key:
        dict1.pop(key)


def check_df_filled(df_list: [pd.DataFrame]):
    """
    Удаляет из списков таблиц пустые таблицы.

    Parameters
    ----------
    df_list : [pd.DataFrame]
        Список таблиц.

    Returns
    -------
    None.

    """
    # Список индексов таблиц списка для дальнейшего удаления
    index_to_delite = []

    for i in range(len(df_list)):
        # Если хотя бы одна размерность массива данных таблицы равна нулю
        if df_list[i].shape[0] == 0 or df_list[i].shape[0] == 1:
            index_to_delite.append(i)
    index_to_delite.reverse()  # Переворачиваем
    for i in index_to_delite:
        df_list.pop(i)  # Удаляем


# %% Работа со строками

def check_content(
        checked_string: str,
        content: str,
        reversed_check: bool = False,
        right=True):
    """
    Проверяет содержится ли символ или строка в строке, если не содержится \
        возвращает оригинальную строку. Если содержится возвращает часть \
        слева от содержимого либо часть справа от содержимого. Поиск может \
        производится либо слева направо либо справа налево до первого \
        совпадения.

    Parameters
    ----------
    checked_string : str
        Проверяемая строка.
    content : str
        Проверяемое вхождение.
    reversed_check : bool, optional
        Обратный поиск. The default is False.
    right : TYPE, optional
        Возвращаемая часть. The default is True.

    Returns
    -------
    str
        Возвращает оригинальную строку, если вхождения не найдено.
        Если найдено возвращает часть слева от содержимого либо часть справа
        от содержимого.

    """
    if content in checked_string:  # Если искомое содержится в строке
        # Возвращается либо часть справа от искомого либо слева, в соответствии
        # с направлением поиска
        if reversed_check:
            return (checked_string[(len(checked_string) -
                                    checked_string[::-1].index(content)):]
                    if right
                    else checked_string[:(len(checked_string) -
                                          checked_string[::-1].
                                          index(content) - 1)])
        else:
            return (checked_string[(checked_string
                                    [:: (reversed_check * (-2) + 1)].
                                    index(content) + 1):]
                    if right
                    else checked_string[:(checked_string
                                          [::(reversed_check * (-2) + 1)].
                                          index(content))])
    # Если искомое не содержится в строке возвращается строка без изменений
    return checked_string

# %% Частотный анализ


def butfilter(data, order, sampl_rate, lowpass, highpass=None, metod='pad'):
    """
    Фильтр Баттерворта без фазового сдвига.

    Parameters
    ----------
    data : Список/одномерный массив
        Массив значений фильтруемого сигнала
    order : integer
        Порядок фильтра.
    sampl_rate : float/integer
        Частота дискретизации сигнала.
    lowpass : float/integer
        Нижняя частота среза.
    highpass : float/integer, optional
        Верхняя частота среза. The default is None.
    metod : string, optional
        Метод фильтрации: 'pad' / 'gust'. The default is 'pad'.

    Returns
    -------
    y : Одномерный массив
        Массив отфильтрованных значений.

    """
    nyq = 0.5 * sampl_rate  # Частота Найквиста
    lowpass = lowpass / nyq  # Нормированная частота среза фильтра

    # Условие определения типа фильтра (низкочастотный / полосовой)
    if highpass is None:  # Низкочастотный
        b, a = butter(order, lowpass)  # Определение коэффициентов фильтра
    else:  # Полосовой
        highpass = highpass / nyq  # Верхняя частота среза
        b, a = butter(order, [lowpass, highpass], btype='band')

    # Двойной (прямой и обратный) линейный цифровой фильтр
    y = filtfilt(b, a, data, method=metod)
    return y


def butfilter_resp(order, sampl_rate, lowpass, highpass=None):
    """
    Частотный отклик фильтра. Позволяет оценить амплитуды пропускаемого \
        сигнала.

    Parameters
    ----------
    order : integer
        Порядок фильтра.
    sampl_rate : float/integer
        Частота дискретизации сигнала.
    lowpass : float/integer
        Нижняя частота среза.
    highpass : float/integer, optional
        Верхняя частота среза. The default is None.

    Returns
    -------
    x : Одномерный массив
        Массив значений по оси х (частота).
    y : Одномерный массив
        Массив значений по оси у (нормированная амплитуда).

    """
    nyq = 0.5 * sampl_rate  # Частота Найквиста
    lowpass = lowpass / nyq  # Нормированная частота среза фильтра

    # Условие определения типа фильтра (низкочастотный / полосовой)
    if highpass is None:  # Низкочастотный
        b, a = butter(order, lowpass)  # Определение коэффициентов фильтра
    else:  # Полосовой
        highpass = highpass / nyq  # Верхняя частота среза
        b, a = butter(order, [lowpass, highpass], btype='band')

    # Определение массивов частоты и амплитуды
    w, h = freqz(b, a, worN=2000)

    # Переход к герцам и абсолютным значениям амплитуды
    x = (sampl_rate * 0.5 / np.pi) * w
    y = abs(h)

    # Обрезание массивов до 2% амплитуды
    y = [val for val in y if val > 0.02]
    x = [x[i] for i in range(len(y))]

    return x, y


def df_butfilt(dataframe, order, lowcut, par=1):
    """
    Фильтр баттерворта для блока данных (pandas data frame), \
    где в первом столбце - время, в остальных данные.

    Parameters
    ----------
    dataframe : pandas data frame
        Таблица пандас. Первый столбец - время, остальные - данные.
    order : integer
        Порядок фильтра баттерворта.
    lowcut : float / integer
        Частота среза фильтра.
    par : integer
        параметр определяющий метод работы функции
        par = 1 копирует блок данных, возвращает новый, отфильтрованныйблок
        данных.
        par = 2 фильтрует существующий блок данных, ничего не возвращает

    Returns
    -------
    df : pandas data frame
        Таблица пандас с отфильтрованными данными.

    """
    if par == 1:
        df = dataframe.copy()  # Копия таблицы пандас
    else:
        df = dataframe  # Ссылка на таблицу пандас

    col = df.columns  # Список имен столбцов таблицы пандас
    # Вычисление частоты дискретизации
    s_rate = 1 / (df[col[0]][1] - df[col[0]][0])
    # Список имен столбцов, которые будут фильтроваться (убираем время)
    col = col[1:]

    # Процедура фильтрации
    for x in col:
        df[x] = butfilter(df[x], order, s_rate, lowcut)

    if par == 1:
        return df

# %% Вычислительный блок


def abs_max(m1: np.ndarray, m2: np.ndarray):
    """
    Поэллементно сравнивает значения двух массивов и возвращает большее \
    по модулю значение с первоначальным знаком.

    Parameters
    ----------
    m1 : numpy array
        Первый массив значений.
    m2 : numpy array
        Второй массив значений.

    Returns
    -------
    absmax : numpy array
        Возвращаемый массив.

    """
    # Сравнение размерностей массивов
    assert m1.shape == m2.shape, 'Не совпадают размерности массивов numpy'

    # Вычисление абсолютных значений для массивов
    am1 = np.absolute(m1)
    am2 = np.absolute(m2)

    # Массив булевых значений, полученных при поэлемментном сравнении массивов
    great = np.greater_equal(am1, am2)

    # Вычисление итогового массива
    absmax = great * m1 + np.absolute(great - 1) * m2

    return absmax


def df_abs_max(df1: pd.DataFrame, df2: pd.DataFrame):
    """
    Поэллементно сравнивает значения блока данных pandas, возвращает \
    блок данных с большими по модулю значениями и первоначальным знаком, \
    названия столбцов и рядов используются от второй таблицы.

    Parameters
    ----------
    df1 : pandas dataframe
        Первый блок данных.
    df2 : pandas dataframe
        Второй блок данных.

    Returns
    -------
    df_absmax : pandas dataframe
        Возвращаемый блок данных.

    """
    # Преобразование в массивы (numpy array)
    m1 = df1.values
    m2 = df2.values

    # Обращение к функции abs_max и формирование блока данных pandas с новыми
    # значениями
    df_absmax = pd.DataFrame(
        abs_max(
            m1,
            m2),
        columns=df2.columns,
        index=df2.index)

    return df_absmax


def add_combcomb_data(df_mg_data: pd.DataFrame):
    """
    Добавляет в таблицу столбцы с суммой Px, разностью Px, суммой Py для \
        передней и задней ООШ.

    Parameters
    ----------
    df_mg_data : pd.DataFrame
        Таблица с результатами для ООШ.

    Returns
    -------
    df_mg_data : pd.DataFrame
        Дополненная таблица с результатами для ООШ.
    """
    # Выделение данных столбцов таблицы
    px1 = df_mg_data['f_wh'][3]['x'].values  # Px ПООШ
    px2 = df_mg_data['f_wh'][5]['x'].values  # Px ЗООШ
    py1 = df_mg_data['f_wh'][3]['y'].values  # Py ПООШ
    py2 = df_mg_data['f_wh'][5]['y'].values  # Py ЗООШ

    # Вычисление почленных сумм и разниц и объединение массивов
    new_data = np.concatenate((px1 + px2, px1 - px2, py1 + py2), axis=1)

    # Создание колонок с мультииндексом для новых данных
    new_columns = pd.MultiIndex.from_tuples(
        [['f_wh', '3+5', 'x', 'none'],
         ['f_wh', '3-5', 'x', 'none'],
         ['f_wh', '3+5', 'y', 'none']])

    new_columns.names = df_mg_data.columns.names

    # Создание таблицы с новыми данными
    df_new_data = pd.DataFrame(
        new_data,
        columns=new_columns,
        index=df_mg_data.index)

    # Объединение таблиц
    df_mg_data = pd.concat([df_new_data, df_mg_data], axis=1)

    return df_mg_data


def max_gear_loads(df_data: pd.DataFrame, df_output: pd.DataFrame):
    """
    Вычисляет максимальные и минимальные значения для одной стойки: \
    'Px_max', 'Px_min', 'Py_max', 'Sa_max', 'Spn_max' \
    Добавляет их в таблицу df_output и возвращает ее.

    Parameters
    ----------
    df_data : pd.DataFrame
        таблица с результатами расчета для опоры
    df_output : pd.DataFrame
        таблица максимальных значений для стойки

    Returns
    -------
    df_output : pd.DataFrame
        Обновленная таблица максимальных значений для стойки

    """
    tmp = []  # Пустой список
    # Наполнение списка значениями
    tmp.append(df_data['f_wh']['x'].max()[0])  # Максимум Х
    tmp.append(df_data['f_wh']['x'].min()[0])  # Минимум Х
    tmp.append(df_data['f_wh']['y'].max()[0])  # Максимум Y
    tmp1 = df_data['s_am']['0'].max()[0]
    tmp.append(tmp1 if tmp1 > 0 else 0.0)  # Максимум S_am, если больше нуля
    tmp1 = df_data['s_pn']['0'].max()[0]
    tmp.append(tmp1 if tmp1 > 0 else 0.0)  # Максимум S_pn, если больше нуля
    # Двумерный список
    tmp = [tmp]

    # На выход - таблица df_output + таблица со именами колонок, как в
    # df_output
    df_output = df_output.append(
        pd.DataFrame(
            tmp,
            columns=df_output.columns),
        ignore_index=True)
    return df_output


def comb_snglgear_loads(df_data: pd.DataFrame, df_output: pd.DataFrame):
    """
    Вычисляет комбинации для одной стойки: \
    'Px_max', 'Py', 'Sa', 'Spn','Px_min', 'Py', 'Sa', 'Spn','Px', 'Py_max', \
        'Sa', 'Spn' \
    Добавляет их в таблицу df_output и возвращает ее.

    Parameters
    ----------
    df_data : pd.DataFrame
        Таблица с результатами расчета.
    df_output : pd.DataFrame
        Дополняемая таблица.

    Returns
    -------
    df_output : pd.DataFrame
        Дополненная таблица.

    """
    tmp = []  # Пустой список

    # Наполнение списка значениями соответствующими Px_max
    tmp_ser = df_data.loc[df_data['f_wh']['x'].idxmax()]  # Поиск ряда значений
    tmp.append(tmp_ser['f_wh']['x'].iloc[0][0])  # Значения Px
    tmp.append(tmp_ser['f_wh']['y'].iloc[0][0])  # Значения Py
    tmp.append(tmp_ser['s_am']['0'].iloc[0][0] if tmp_ser['s_am']
               ['0'].iloc[0][0] > 0 else 0.0)  # Значения Sam
    tmp.append(tmp_ser['s_pn']['0'].iloc[0][0] if tmp_ser['s_pn']
               ['0'].iloc[0][0] > 0 else 0.0)  # Значения Spn

    # Наполнение списка значениями соответствующими Px_min
    tmp_ser = df_data.loc[df_data['f_wh']['x'].idxmin()]  # Поиск ряда значений
    tmp.append(tmp_ser['f_wh']['x'].iloc[0][0])  # Значения Px
    tmp.append(tmp_ser['f_wh']['y'].iloc[0][0])  # Значения Py
    tmp.append(tmp_ser['s_am']['0'].iloc[0][0] if tmp_ser['s_am']
               ['0'].iloc[0][0] > 0 else 0.0)  # Значения Sam
    tmp.append(tmp_ser['s_pn']['0'].iloc[0][0] if tmp_ser['s_pn']
               ['0'].iloc[0][0] > 0 else 0.0)  # Значения Spn

    # Наполнение списка значениями соответствующими Py_max
    tmp_ser = df_data.loc[df_data['f_wh']['y'].idxmin()]  # Поиск ряда значений
    tmp.append(tmp_ser['f_wh']['x'].iloc[0][0])  # Значения Px
    tmp.append(tmp_ser['f_wh']['y'].iloc[0][0])  # Значения Py
    tmp.append(tmp_ser['s_am']['0'].iloc[0][0] if tmp_ser['s_am']
               ['0'].iloc[0][0] > 0 else 0.0)  # Значения Sam
    tmp.append(tmp_ser['s_pn']['0'].iloc[0][0] if tmp_ser['s_pn']
               ['0'].iloc[0][0] > 0 else 0.0)  # Значения Spn

    tmp = [tmp]  # Двумерный список

    # На выход - таблица df_output + таблица со именами колонок, как в
    # df_output
    df_output = df_output.append(
        pd.DataFrame(
            tmp,
            columns=df_output.columns),
        ignore_index=True)
    return df_output


def comb_dblgear_loads(
        df_data: pd.DataFrame,
        df_output: pd.DataFrame,
        mtype: str,
        gear_id: list,
        comp: str,
        par='max'):
    """
    Вычисляет комбинации для двух стоек: \
            ПООШ                      ЗООШ \
    'Px', 'Py', 'Sa', 'Spn','Px', 'Py', 'Sa', 'Spn' \
    Находит максимум или минимум в указанном столбце. \
    Сперва выводит значения для передней опоры (с меньшим номером), \
        затем для задней.

    Parameters
    ----------
    df_data : pd.DataFrame
        Таблица с результатами расчета.
    df_output : pd.DataFrame
        Дополняемая таблица.
    mtype : str
        Тип результатов (f_wh, s_am, s_pn) - первый уровень столбцов.
    gear_id : list
        Список номеров опор. Поиск максимума производится для первой опоры
        в списке.
    comp : str
        Компонент нагрузки(x, y, z) - третий уровень столбцов.
    par : str, optional
        Параметр задающий что ищем - максимум или минимум. Если не максимум,
        то минимум. The default is 'max'.

    Returns
    -------
    df_output : pd.DataFrame
        Дополненная таблица.
    """
    tmp = []  # Пустой список

    # Поиск ряда значений
    if par == 'max':
        # Поиск по максимальному значению
        tmp_ser = df_data.loc[df_data[mtype][gear_id[0]][comp].idxmax()]
    else:
        tmp_ser = df_data.loc[df_data[mtype][gear_id[0]][
            comp].idxmin()]  # Поиск по минимальному значению

    # Наполнение списка соответствующими значениями
    tmp.append(tmp_ser['f_wh'][min(gear_id)]['x'].iloc[0][0])  # Значения Px
    tmp.append(tmp_ser['f_wh'][min(gear_id)]['y'].iloc[0][0])  # Значения Py

    # Значения Sam
    tmp.append(tmp_ser['s_am'][min(gear_id)]['0'].iloc[0][0]
               if tmp_ser['s_am'][min(gear_id)]['0'].iloc[0][0] > 0 else 0.0)

    # Значения Spn
    tmp.append(tmp_ser['s_pn'][min(gear_id)]['0'].iloc[0][0]
               if tmp_ser['s_pn'][min(gear_id)]['0'].iloc[0][0] > 0 else 0.0)

    tmp.append(tmp_ser['f_wh'][max(gear_id)]['x'].iloc[0][0])  # Значения Px
    tmp.append(tmp_ser['f_wh'][max(gear_id)]['y'].iloc[0][0])  # Значения Py

    # Значения Sam
    tmp.append(tmp_ser['s_am'][max(gear_id)]['0'].iloc[0][0]
               if tmp_ser['s_am'][max(gear_id)]['0'].iloc[0][0] > 0 else 0.0)

    # Значения Spn
    tmp.append(tmp_ser['s_pn'][max(gear_id)]['0'].iloc[0][0]
               if tmp_ser['s_pn'][max(gear_id)]['0'].iloc[0][0] > 0 else 0.0)

    tmp = [tmp]  # Двумерный список

    # На выход - таблица df_output + таблица со именами колонок, как в
    # df_output
    df_output = df_output.append(
        pd.DataFrame(
            tmp,
            columns=df_output.columns),
        ignore_index=True)
    return df_output


def comb_sum_gear_loads(
        df_data: pd.DataFrame,
        df_output: pd.DataFrame,
        mtype: str,
        gear_id: list,
        comp: str,
        par='max'):
    """
    Вычисляет комбинации для двух стоек: \
            ПООШ                      ЗООШ \
    'Summ_Pxy', 'Px', 'Py', 'Sa', 'Spn','Px', 'Py', 'Sa', 'Spn' \
    Находит максимум или минимум в указанном столбце. \
    Сперва выводит значения для передней опоры (с меньшим номером), \
        затем для задней.

    Parameters
    ----------
    df_data : pd.DataFrame
        Таблица с результатами расчета.
    df_output : pd.DataFrame
        Дополняемая таблица.
    mtype : str
        Тип результатов (f_wh, s_am, s_pn) - первый уровень столбцов.
    gear_id : list
        Список номеров опор. Поиск максимума производится для первой опоры
        в списке.
    comp : str
        Компонент нагрузки(x, y, z) - третий уровень столбцов.
    par : str, optional
        Параметр задающий что ищем - максимум или минимум. Если не максимум,
        то минимум. The default is 'max'.

    Returns
    -------
    df_output : pd.DataFrame
        Дополненная таблица.
    """
    tmp = []  # Пустой список

    # Поиск ряда значений
    if par == 'max':
        # Поиск по максимальному значению
        tmp_ser = df_data.loc[df_data[mtype][gear_id[0]][comp].idxmax()]
    else:
        tmp_ser = df_data.loc[df_data[mtype][gear_id[0]][
            comp].idxmin()]  # Поиск по минимальному значению

    # Наполнение списка соответствующими значениями
    tmp.append(tmp_ser['f_wh'][gear_id[0]][comp].iloc[0][0])  # Значения суммы

    tmp.append(tmp_ser['f_wh'][3]['x'].iloc[0][0])  # Значения Px
    tmp.append(tmp_ser['f_wh'][3]['y'].iloc[0][0])  # Значения Py
    tmp.append(tmp_ser['s_am'][3]['0'].iloc[0][0] if tmp_ser['s_am']
               [3]['0'].iloc[0][0] > 0 else 0.0)  # Значения Sam
    tmp.append(tmp_ser['s_pn'][3]['0'].iloc[0][0] if tmp_ser['s_pn']
               [3]['0'].iloc[0][0] > 0 else 0.0)  # Значения Spn

    tmp.append(tmp_ser['f_wh'][5]['x'].iloc[0][0])  # Значения Px
    tmp.append(tmp_ser['f_wh'][5]['y'].iloc[0][0])  # Значения Py
    tmp.append(tmp_ser['s_am'][5]['0'].iloc[0][0] if tmp_ser['s_am']
               [5]['0'].iloc[0][0] > 0 else 0.0)  # Значения Sam
    tmp.append(tmp_ser['s_pn'][5]['0'].iloc[0][0] if tmp_ser['s_pn']
               [5]['0'].iloc[0][0] > 0 else 0.0)  # Значения Spn

    tmp = [tmp]  # Двумерный список

    # На выход - таблица df_output + таблица со именами колонок, как в
    # df_output
    df_output = df_output.append(
        pd.DataFrame(
            tmp,
            columns=df_output.columns),
        ignore_index=True)
    return df_output

# %%% Нагрузки на шасси


def calc_fg_loads(gear_data, df_init, res) -> List[Any]:

    # Формирование таблиц
    c_tmp = pd.MultiIndex.from_tuples([('FG', 'Px_max'),
                                       ('FG', 'Px_min'),
                                       ('FG', 'Py_max'),
                                       ('FG', 'Sa_max'),
                                       ('FG', 'Spn_max')])

    # Пустая таблица максимумов ПОШ
    df_fg_max = pd.DataFrame([], columns=c_tmp)
    c_tmp = pd.MultiIndex.from_tuples([('FG_x_max', 'Px_max'),
                                       ('FG_x_max', 'Py'),
                                       ('FG_x_max', 'Sa'),
                                       ('FG_x_max', 'Spn'),
                                       ('FG_x_min', 'Px_min'),
                                       ('FG_x_min', 'Py'),
                                       ('FG_x_min', 'Sa'),
                                       ('FG_x_min', 'Spn'),
                                       ('FG_y_max', 'Px'),
                                       ('FG_y_max', 'Py_max'),
                                       ('FG_y_max', 'Sa'),
                                       ('FG_y_max', 'Spn')])
    # Пустая таблица комбинаций ПОШ
    df_fg_comb = pd.DataFrame([], columns=c_tmp)

    # Наполнение таблиц
    all_fg_data = []
    for df_data in gear_data:
        # Массив данных по передней опоре
        df_fg_data = df_data.xs(1.0, level='agr_number', axis=1)
        # Наполнение массива для вывода графиков
        all_fg_data.append(df_fg_data)

        # Наполнение таблиц
        # Формирование таблицы максимумов
        df_fg_max = max_gear_loads(df_fg_data, df_fg_max)
        df_fg_comb = comb_snglgear_loads(
            df_fg_data, df_fg_comb)  # Формирование таблицы комбинаций

    # Заполнение словаря с результатами
    res['FG'] = [df_fg_max, df_fg_comb]

    # Отсев случаев на основные опоры шасси
    for i in range(len(res['FG'])):
        res['FG'][i] = res['FG'][i][res['FG'][i].index.isin(
            df_init[df_init['type'].isin(['all', 'fg'])].index)]

    # Формирование списка с таблицами для графиков
    all_fg_data_clear: List[Any] = []
    for index in res['FG'][0].index.tolist():
        all_fg_data_clear.append(all_fg_data[index])

    check_df_filled(res['FG'])  # Удаление пустых таблиц из списка

    return all_fg_data_clear


def calc_mg_loads_4(gear_data: pd.DataFrame, df_init: pd.DataFrame, res: dict):

    if True in gear_data[0].columns.get_level_values(
            'agr_number').isin([2, 4]):  # если есть опоры с номером 2 и 4
        all_mg_data = []  # Пустой список для обобщенных результатов по ООШ
        for df_data in gear_data:
            df_mg_left_data = df_data.loc[slice(
                None), (slice(None), [2, 4])]  # Левая сторона
            df_mg_right_data = df_data.loc[slice(
                None), (slice(None), [3, 5])]  # Правая сторона
            # Обобщенный массив по основным опорам для левой и правой стороны
            df_mg_data = df_abs_max(df_mg_left_data, df_mg_right_data)
            # Добавление колонок с суммой и разницей нагрузок
            df_mg_data = add_combcomb_data(df_mg_data)
            all_mg_data.append(df_mg_data)  # Наполнение списка

    else:  # если их нет, то берутся значения только для номеров 3 и 5
        all_mg_data = []  # Пустой список для обобщенных результатов по ООШ
        for df_data in gear_data:
            df_mg_data = df_data.loc[slice(
                None), (slice(None), [3, 5])]  # Правая сторона
            all_mg_data.append(df_mg_data)  # Наполнение списка

    # Формирование таблиц
    df_mgf_max = pd.DataFrame(
        [],
        columns=[
            'Px_max',
            'Px_min',
            'Py_max',
            'Sa_max',
            'Spn_max'])  # Пустая таблица максимумов ПООШ

    df_mgr_max = pd.DataFrame(
        [],
        columns=[
            'Px_max',
            'Px_min',
            'Py_max',
            'Sa_max',
            'Spn_max'])  # Пустая таблица максимумов ЗООШ

    c_tmp = pd.MultiIndex.from_tuples([('MGF', 'Px_max'), ('MGF', 'Py'),
                                       ('MGF', 'Sa'), ('MGF', 'Spn'),
                                       ('MGR', 'Px'), ('MGR', 'Py'),
                                       ('MGR', 'Sa'), ('MGR', 'Spn')])
    # Пустая таблица комбинаций ПООШ
    df_mgf_comb1 = pd.DataFrame([], columns=c_tmp)

    c_tmp = pd.MultiIndex.from_tuples([('MGF', 'Px_min'), ('MGF', 'Py'),
                                       ('MGF', 'Sa'), ('MGF', 'Spn'),
                                       ('MGR', 'Px'), ('MGR', 'Py'),
                                       ('MGR', 'Sa'), ('MGR', 'Spn')])
    # Пустая таблица комбинаций ПООШ
    df_mgf_comb2 = pd.DataFrame([], columns=c_tmp)

    c_tmp = pd.MultiIndex.from_tuples([('MGF', 'Px'), ('MGF', 'Py_max'),
                                       ('MGF', 'Sa'), ('MGF', 'Spn'),
                                       ('MGR', 'Px'), ('MGR', 'Py'),
                                       ('MGR', 'Sa'), ('MGR', 'Spn')])
    # Пустая таблица комбинаций ПООШ
    df_mgf_comb3 = pd.DataFrame([], columns=c_tmp)

    c_tmp = pd.MultiIndex.from_tuples([('MGF', 'Px'), ('MGF', 'Py'),
                                       ('MGF', 'Sa'), ('MGF', 'Spn'),
                                       ('MGR', 'Px_max'), ('MGR', 'Py'),
                                       ('MGR', 'Sa'), ('MGR', 'Spn')])
    # Пустая таблица комбинаций ЗООШ
    df_mgr_comb1 = pd.DataFrame([], columns=c_tmp)

    c_tmp = pd.MultiIndex.from_tuples([('MGF', 'Px'), ('MGF', 'Py'),
                                       ('MGF', 'Sa'), ('MGF', 'Spn'),
                                       ('MGR', 'Px_min'), ('MGR', 'Py'),
                                       ('MGR', 'Sa'), ('MGR', 'Spn')])
    # Пустая таблица комбинаций ЗООШ
    df_mgr_comb2 = pd.DataFrame([], columns=c_tmp)

    c_tmp = pd.MultiIndex.from_tuples([('MGF', 'Px'), ('MGF', 'Py'),
                                       ('MGF', 'Sa'), ('MGF', 'Spn'),
                                       ('MGR', 'Px'), ('MGR', 'Py_max'),
                                       ('MGR', 'Sa'), ('MGR', 'Spn')])
    # Пустая таблица комбинаций ЗООШ
    df_mgr_comb3 = pd.DataFrame([], columns=c_tmp)

    c_tmp = pd.MultiIndex.from_tuples([('ALL', 'Px+Px_max'), ('MGF', 'Px'),
                                       ('MGF', 'Py'), ('MGF', 'Sa'),
                                       ('MGF', 'Spn'), ('MGR', 'Px'),
                                       ('MGR', 'Py'), ('MGR', 'Sa'),
                                       ('MGR', 'Spn')])
    # Пустая таблица суммарных комбинаций
    df_mg_comb_maxpx1 = pd.DataFrame([], columns=c_tmp)

    c_tmp = pd.MultiIndex.from_tuples([('ALL', 'Px+Px_min'), ('MGF', 'Px'),
                                       ('MGF', 'Py'), ('MGF', 'Sa'),
                                       ('MGF', 'Spn'), ('MGR', 'Px'),
                                       ('MGR', 'Py'), ('MGR', 'Sa'),
                                       ('MGR', 'Spn')])
    # Пустая таблица суммарных комбинаций
    df_mg_comb_minpx1 = pd.DataFrame([], columns=c_tmp)

    c_tmp = pd.MultiIndex.from_tuples([('ALL', 'Px-Px_max'), ('MGF', 'Px'),
                                       ('MGF', 'Py'), ('MGF', 'Sa'),
                                       ('MGF', 'Spn'), ('MGR', 'Px'),
                                       ('MGR', 'Py'), ('MGR', 'Sa'),
                                       ('MGR', 'Spn')])
    # Пустая таблица суммарных комбинаций
    df_mg_comb_maxpx2 = pd.DataFrame([], columns=c_tmp)

    c_tmp = pd.MultiIndex.from_tuples([('ALL', 'Px-Px_min'), ('MGF', 'Px'),
                                       ('MGF', 'Py'), ('MGF', 'Sa'),
                                       ('MGF', 'Spn'), ('MGR', 'Px'),
                                       ('MGR', 'Py'), ('MGR', 'Sa'),
                                       ('MGR', 'Spn')])
    # Пустая таблица суммарных комбинаций
    df_mg_comb_minpx2 = pd.DataFrame([], columns=c_tmp)

    c_tmp = pd.MultiIndex.from_tuples([('ALL', 'Py+Py_max'), ('MGF', 'Px'),
                                       ('MGF', 'Py'), ('MGF', 'Sa'),
                                       ('MGF', 'Spn'), ('MGR', 'Px'),
                                       ('MGR', 'Py'), ('MGR', 'Sa'),
                                       ('MGR', 'Spn')])
    # Пустая таблица суммарных комбинаций
    df_mg_comb_maxpy = pd.DataFrame([], columns=c_tmp)

    del c_tmp

    # Наполнение таблиц
    for df_data in all_mg_data:
        df_mgf_data = df_data.xs(
            3, level='agr_number', axis=1)  # Таблица по ПООШ
        df_mgr_data = df_data.xs(
            5, level='agr_number', axis=1)  # Таблица по ЗООШ

        # Формирование таблицы максимумов ПООШ
        df_mgf_max = max_gear_loads(df_mgf_data, df_mgf_max)
        # Формирование таблицы максимумов ЗООШ
        df_mgr_max = max_gear_loads(df_mgr_data, df_mgr_max)

        # Формирование таблицы комбинаций ПООШ
        df_mgf_comb1 = comb_dblgear_loads(
            df_data, df_mgf_comb1, 'f_wh', gear_id=[
                3, 5], comp='x', par='max')

        # Формирование таблицы комбинаций ПООШ
        df_mgf_comb2 = comb_dblgear_loads(
            df_data, df_mgf_comb2, 'f_wh', gear_id=[
                3, 5], comp='x', par='min')

        # Формирование таблицы комбинаций ПООШ
        df_mgf_comb3 = comb_dblgear_loads(
            df_data, df_mgf_comb3, 'f_wh', gear_id=[
                3, 5], comp='y', par='max')

        # Формирование таблицы комбинаций ЗООШ
        df_mgr_comb1 = comb_dblgear_loads(
            df_data, df_mgr_comb1, 'f_wh', gear_id=[
                5, 3], comp='x', par='max')

        # Формирование таблицы комбинаций ЗООШ
        df_mgr_comb2 = comb_dblgear_loads(
            df_data, df_mgr_comb2, 'f_wh', gear_id=[
                5, 3], comp='x', par='min')

        # Формирование таблицы комбинаций ЗООШ
        df_mgr_comb3 = comb_dblgear_loads(
            df_data, df_mgr_comb3, 'f_wh', gear_id=[
                5, 3], comp='y', par='max')

        # Формирование таблицы суммарных комбинаций
        df_mg_comb_maxpx1 = comb_sum_gear_loads(
            df_data,
            df_mg_comb_maxpx1,
            'f_wh',
            gear_id=['3+5'],
            comp='x',
            par='max')

        # Формирование таблицы суммарных комбинаций
        df_mg_comb_maxpx2 = comb_sum_gear_loads(
            df_data,
            df_mg_comb_maxpx2,
            'f_wh',
            gear_id=['3+5'],
            comp='x',
            par='min')

        # Формирование таблицы суммарных комбинаций
        df_mg_comb_minpx1 = comb_sum_gear_loads(
            df_data,
            df_mg_comb_minpx1,
            'f_wh',
            gear_id=['3-5'],
            comp='x',
            par='max')

        # Формирование таблицы суммарных комбинаций
        df_mg_comb_minpx2 = comb_sum_gear_loads(
            df_data,
            df_mg_comb_minpx2,
            'f_wh',
            gear_id=['3-5'],
            comp='x',
            par='min')

        # Формирование таблицы суммарных комбинаций
        df_mg_comb_maxpy = comb_sum_gear_loads(
            df_data,
            df_mg_comb_maxpy,
            'f_wh',
            gear_id=['3+5'],
            comp='y',
            par='max')

    # Объединение таблиц
    df_mg_max = pd.concat([df_mgf_max, df_mgr_max],
                          axis=1, keys=['MGF', 'MGR'])

    # Заполнение словаря с результатами
    res['MG_max'] = [df_mg_max]
    res['MG_comb'] = [
        df_mgf_comb1,
        df_mgf_comb2,
        df_mgf_comb3,
        df_mgr_comb1,
        df_mgr_comb2,
        df_mgr_comb3]
    res['MG_cross_comb'] = [
        df_mg_comb_maxpx1,
        df_mg_comb_maxpx2,
        df_mg_comb_minpx1,
        df_mg_comb_minpx2,
        df_mg_comb_maxpy]

    # Отсев случаев на переднюю опору шасси
    for key in res:
        if 'MG' in key:  # Если в имени ключа 'MG'
            for i in range(len(res[key])):
                res[key][i] = res[key][i][res[key][i].index.isin(
                    df_init[df_init['type'].isin(['all', 'mg'])].index)]

    # Формирование списка с таблицами для графиков
    all_mg_data_clear = []
    for index in res['MG_max'][0].index.tolist():
        all_mg_data_clear.append(all_mg_data[index])

    check_df_filled(res['MG_max'])  # Удаление пустых таблиц из списка
    check_df_filled(res['MG_comb'])  # Удаление пустых таблиц из списка
    check_df_filled(res['MG_cross_comb'])  # Удаление пустых таблиц из списка

    return all_mg_data_clear

# %%% Перегрузки


def overload_distr(
        all_data: [pd.DataFrame],
        init_table: pd.DataFrame,
        level: int or str,
        par='max'):
    """
    Определяет распределение(макс или мин) перегрузок по агрегату с учетом \
        коэффициента безопасности.

    Parameters
    ----------
    all_data : [pd.DataFrame]
        Список с таблицами, содержащими результаты Адамс по агрегату.
    init_table : pd.DataFrame
        Таблица исходных данных.
    level : int or str
        Название или номер уровня имен колонок, которые будут использованы в
        выходной таблице.
    par : TYPE, optional
        Параметр - максимум или минимум. The default is 'max'.

    Raises
    ------
    AssertionError
        Если par не равен max или min.

    Returns
    -------
    df_rez : pd.DataFrame
        Таблица с максимумами(огибающих).

    """
    flag = True
    for df in all_data:
        if df is None:
            flag = False

    if flag:
        # Создание пустой Таблицы с максимумами/минимумами
        df_rez = pd.DataFrame(
            [], columns=all_data[0].columns.get_level_values(level))

        if par == 'max':  # Ищем максимумы с домножением на коэф. безопасности
            for i in range(len(all_data)):
                df_rez.loc[i] = (
                    all_data[i].max(
                        axis=0) *
                    init_table['safety_factor'].loc[i]).values
        elif par == 'min':  # Ищем минимумы с домножением на коэф. безопасности
            for i in range(len(all_data)):
                df_rez.loc[i] = (
                    all_data[i].min(
                        axis=0) *
                    init_table['safety_factor'].loc[i]).values
        else:
            raise AssertionError('Choose min or max')

        # Добавление колонки с номерами случаев
        df_rez = pd.concat([init_table['number'], df_rez],
                           axis=1, join='inner')

        if par == 'max':
            df_rez.loc['MAX'] = df_rez.max()  # Добавление ряда с максимумами
            df_rez['number']['MAX'] = 'MAX'  # Добавление подписи ряда
        else:
            df_rez.loc['MIN'] = df_rez.min()  # Добавление ряда с минимумами
            df_rez['number']['MIN'] = 'MIN'  # Добавление подписи ряда
    else:
        df_rez = None

    return df_rez


def all_overload_data_cre(
        all_adams_res: [pd.DataFrame],
        ass_table: pd.DataFrame,
        aggregate: str,
        component: str,
        coord: str):
    """
    Создает список с Таблицами результатов для различных агрегатов по \
        определенным компонентам нагрузок и для определенных координат.

    Parameters
    ----------
    all_adams_res : [pd.DataFrame]
        Список Таблиц со всеми результатами из Адамса.
    ass_table : pd.DataFrame
        Таблица ассоциации.
    aggregate : str
        Текстовое наименование агрегата.
    component : str
        Текстовое наименование компонента нагрузки.
    coord : str
        Текстовое наименование столбца, содержащего координаты.

    Returns
    -------
    all_tables : [pd.DataFrame]
        Список Таблиц с отфильтрованными значениями.

    """
    df_col = ass_table[ass_table['aggregate'].isin(
        [aggregate])]  # Фильтр массива исходных данных по агрегату
    # Удаление лишних столбцов
    df_col = df_col[['component', coord, 'me_name']]
    if coord == 'me_loc_z':
        df_col[coord] = df_col[coord].abs()
    # Формирование массива с индексами (мультииндекс)
    df_col = pd.MultiIndex.from_frame(df_col)

    all_tables = []  # Пустой список с таблицами результатов и мультииндексом
    for df_res in all_adams_res:
        # Выборка столбцов с данными по имени
        df_data = df_res[df_col.get_level_values('me_name')]
        df_data.columns = df_col  # добавление мультииндекса к данными
        df_data = df_data.sort_index(axis=1)  # Сортировка по названию столбцов
        # df_data = df_data[component] # Отсев по интересующему компоненту
        df_data = df_data.get(component)  # !!!
        all_tables.append(df_data)  # Наполнение списка

    return all_tables


def calc_overloads(
        adams_res: [pd.DataFrame],
        ass_table: pd.DataFrame,
        init_table: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Возвращает словарь с таблицыми распределения перегрузок и огибающих по \
        всем случаям и всем агрегатам.

    Parameters
    ----------
    adams_res : [pd.DataFrame]
        Список таблиц с результатами адамс.
    ass_table : pd.DataFrame
        Таблица ассоциаций.
    init_table : pd.DataFrame
        Таблица исходных данных.

    Returns
    -------
    res_ovlds : {}
        Словарь с результатами.

    """
    # Создание списков со всеми результатами по агрегатам:
    # Фюзеляж
    all_fuz_data = all_overload_data_cre(
        adams_res,
        ass_table,
        aggregate='fuz',
        component='y',
        coord='me_loc_x')

    # Крыло
    all_kr_l_y_data = all_overload_data_cre(
        adams_res,
        ass_table,
        aggregate='kr_l',
        component='y',
        coord='me_loc_z')
    all_kr_r_y_data = all_overload_data_cre(
        adams_res,
        ass_table,
        aggregate='kr_r',
        component='y',
        coord='me_loc_z')
    all_kr_l_x_data = all_overload_data_cre(
        adams_res,
        ass_table,
        aggregate='kr_l',
        component='x',
        coord='me_loc_z')
    all_kr_r_x_data = all_overload_data_cre(
        adams_res,
        ass_table,
        aggregate='kr_r',
        component='x',
        coord='me_loc_z')

    # ГО
    all_go_l_y_data = all_overload_data_cre(
        adams_res,
        ass_table,
        aggregate='go_l',
        component='y',
        coord='me_loc_z')
    all_go_r_y_data = all_overload_data_cre(
        adams_res,
        ass_table,
        aggregate='go_r',
        component='y',
        coord='me_loc_z')
    all_go_l_x_data = all_overload_data_cre(
        adams_res,
        ass_table,
        aggregate='go_l',
        component='x',
        coord='me_loc_z')
    all_go_r_x_data = all_overload_data_cre(
        adams_res,
        ass_table,
        aggregate='go_r',
        component='x',
        coord='me_loc_z')

    # Создание словаря с результатами
    res_ovlds: Dict[str, pd.DataFrame] = {}  # Пустой словарь для результатов
    # Фюзеляж
    res_ovlds['fuz_max'] = overload_distr(
        all_fuz_data, init_table, 'me_loc_x', 'max')

    # Крыло у
    res_ovlds['kr_l_y_max'] = overload_distr(
        all_kr_l_y_data, init_table, 'me_loc_z', 'max')
    res_ovlds['kr_r_y_max'] = overload_distr(
        all_kr_r_y_data, init_table, 'me_loc_z', 'max')

    # Крыло х
    res_ovlds['kr_l_x_max'] = overload_distr(
        all_kr_l_x_data, init_table, 'me_loc_z', 'max')
    res_ovlds['kr_r_x_max'] = overload_distr(
        all_kr_r_x_data, init_table, 'me_loc_z', 'max')
    res_ovlds['kr_l_x_min'] = overload_distr(
        all_kr_l_x_data, init_table, 'me_loc_z', 'min')
    res_ovlds['kr_r_x_min'] = overload_distr(
        all_kr_r_x_data, init_table, 'me_loc_z', 'min')

    # Го у
    res_ovlds['go_l_y_max'] = overload_distr(
        all_go_l_y_data, init_table, 'me_loc_z', 'max')
    res_ovlds['go_r_y_max'] = overload_distr(
        all_go_r_y_data, init_table, 'me_loc_z', 'max')

    # ГО х
    res_ovlds['go_l_x_max'] = overload_distr(
        all_go_l_x_data, init_table, 'me_loc_z', 'max')
    res_ovlds['go_r_x_max'] = overload_distr(
        all_go_r_x_data, init_table, 'me_loc_z', 'max')
    res_ovlds['go_l_x_min'] = overload_distr(
        all_go_l_x_data, init_table, 'me_loc_z', 'min')
    res_ovlds['go_r_x_min'] = overload_distr(
        all_go_r_x_data, init_table, 'me_loc_z', 'min')

    # Обобщение резульатов - объединение результатов для левой и
    # правой части самолета

    # Крыло
    res_ovlds['kr_y_max'] = maxmin_of_df(
        res_ovlds['kr_l_y_max'],
        res_ovlds['kr_r_y_max'],
        pmax=True)
    res_ovlds['kr_x_max'] = maxmin_of_df(
        res_ovlds['kr_l_x_max'],
        res_ovlds['kr_r_x_max'],
        pmax=True)
    res_ovlds['kr_x_min'] = maxmin_of_df(
        res_ovlds['kr_l_x_min'],
        res_ovlds['kr_r_x_min'],
        pmax=False)

    # ГО
    res_ovlds['go_y_max'] = maxmin_of_df(
        res_ovlds['go_l_y_max'],
        res_ovlds['go_r_y_max'],
        pmax=True)
    res_ovlds['go_x_max'] = maxmin_of_df(
        res_ovlds['go_l_x_max'],
        res_ovlds['go_r_x_max'],
        pmax=True)
    res_ovlds['go_x_min'] = maxmin_of_df(
        res_ovlds['go_l_x_min'],
        res_ovlds['go_r_x_min'],
        pmax=False)

    # Очистка словаря от пустых значений - возникают если для агрегатов нет
    # данных
    clear_none_from_dict(res_ovlds)  # чистка словаря

    return res_ovlds


def comb_dv(df_data: pd.DataFrame, pmax=True):
    """
    Возвращает комбинации перегрузок для одного двигателя по трем \
        компонентам в виде массива numpy.

    Parameters
    ----------
    df_data : pd.DataFrame
        Таблица изменения перегрузок от времени для одного двигателя.
        Содержит столбцы с названием x, y, z.
    pmax : bool, optional
        Параметр  - поиск максимальных или минимальных значений.
        The default is True.

    Returns
    -------
    np.array
        Массив значений комбинаций.
    """
    # Поиск ряда значений по компоненте х
    comb_x = df_data.loc[df_data['x'].idxmax()].values.tolist(
    ) if pmax else df_data.loc[df_data['x'].idxmin()].values.tolist()

    # Поиск ряда значений по компоненте у
    comb_y = df_data.loc[df_data['y'].idxmax()].values.tolist(
    ) if pmax else df_data.loc[df_data['y'].idxmin()].values.tolist()

    # Поиск ряда значений по компоненте z
    comb_z = df_data.loc[df_data['z'].idxmax()].values.tolist(
    ) if pmax else df_data.loc[df_data['z'].idxmin()].values.tolist()

    comb = comb_x[0] + comb_y[0] + comb_z[0]

    return np.array(comb)


def calc_dv_overloads(adams_dv_data: [pd.DataFrame],
                      init_table: pd.DataFrame) -> Dict[Any, DataFrame]:
    """
    Основная функция вычисляющая комбинации по двигателям.

    Parameters
    ----------
    adams_dv_data : [pd.DataFrame]
        Список таблиц с результатами из Адамс. Таблицы должны содержать
        правильные колонки.
    init_table : pd.DataFrame
        Таблица Исходных данных.

    Returns
    -------
    res_dv : {}
        Словарь с таблицами комбинаций.

    """
    dv_col = ['1_nx_max', '1_ny', '1_nz', '2_nx', '2_ny_max',
              '2_nz', '3_nx', '3_ny', '3_ny_max']  # Колонки таблицы

    res_dv: Dict[Any, DataFrame]= {}  # Словарь для результатов
    # Используемые колонки
    col_used_0 = adams_dv_data[0].columns.remove_unused_levels()
    # Цикл по используемым инменам колонок на 0 уровне
    for lvl_0 in col_used_0.levels[0]:
        # Используемые колонки
        col_used_1 = adams_dv_data[0][lvl_0].columns.remove_unused_levels()
        # Цикл по используемым колонкам на первом уровне
        for lvl_1 in col_used_1.levels[0]:

            # Временная таблица для максимумов
            df_dv_max = pd.DataFrame([], columns=pd.MultiIndex.from_product(
                [[(lvl_0 + '_' + str(int(lvl_1)) + '_max')], dv_col]))

            # Временная таблица для минимумов
            df_dv_min = pd.DataFrame([], columns=pd.MultiIndex.from_product(
                [[(lvl_0 + '_' + str(int(lvl_1)) + '_min')], dv_col]))

            # Цикл по результатам всех случаев
            for i in range(len(adams_dv_data)):
                # Комбинации по максимумам с учетом коэффициента безопасности
                df_dv_max.loc[i] = (comb_dv(
                    adams_dv_data[i][lvl_0][lvl_1]) *
                    init_table['safety_factor'].loc[i])

                # Комбинации по минимумам с учетом коэффициента безопасности
                df_dv_min.loc[i] = (comb_dv(
                    adams_dv_data[i][lvl_0][lvl_1], pmax=False) *
                    init_table['safety_factor'].loc[i])

            # Запись в словарь результатов
            res_dv[lvl_0 + '_' + str(int(lvl_1)) + '_max'] = df_dv_max
            # Запись в словарь результатов
            res_dv[lvl_0 + '_' + str(int(lvl_1)) + '_min'] = df_dv_min
    return res_dv

# %% Оформление


def format_dict(df: pd.DataFrame, dict_format: dict, level: int = 0):
    """
    Возвращает словарь с маской форматов и пути к столбцам \
        таблицы с мультииндексом: \
        {('MGF', 'Px_min'): "$cf{:.2f}", ...}.

    Parameters
    ----------
    df : pd.DataFrame
        Таблица для которой определяется маска форматов.
    dict_format : dict
        Словать содержаший:
            ключи - по ним осуществляется поиск в названии столбца
            значения - текстовая запись формата, который будет 
            применятся к столбцу.
    level : int, optional
        Уровень поиска в именах столбцов. The default is 0.

    Returns
    -------
    dict_fin : dict
        Словарь формата {('MGF', 'Px_min'): "$cf{:.2f}", ...}.

    """
    tmp = []  # Пустой список
    for i in range(level + 1):  # Наполнение списка названиями столбцов Таблицы
        tmp.append(list(df.columns.get_level_values(i)))

    dict_fin = {}  # Пустой словарь
    for i in range(len(tmp[level])):  # Цикл по всем столбцам таблицы
        # Ключ словаря состоит из пути к столбцу таблицы
        key = tuple([tmp[j][i] for j in range(level + 1)])
        value1 = None  # Значение словаря по умолчанию
        for x in dict_format:  # Цикл по словарю из исходных данных
            # Если ключ данного словаря содержится в названии столбца таблицы
            if x in tmp[level][i]:
                # Значение возвращаемого словаря равно значению данного словаря
                value1 = dict_format[x]
        # Проверка заполненности значения словаря
        assert value1, 'Не найдено искомое значение в строке, \
                        проверьте правильно ли задан словарь'
        dict_fin[key] = value1  # Наполнение словаря

    return dict_fin

# %%% Графики


def figure_cre(size: str = 'A5'):
    """
    Создает пустую диаграмму определенного размера.

    Parameters
    ----------
    size : str, optional
        РАзмер поля графика. The default is 'A5'.

    Returns
    -------
    fig : matplotlib object
        Поле построения.
    ax : matplotlib object
        график.

    """
    f_size = (5.9, 3.5) if size == 'A5' else (9.0, 5.5)  # Размер поля графика

    fig, ax = plt.subplots(figsize=f_size)  # Пустое поле

    # Set the tick labels font Задание шрифта графиков
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('times new roman')
        label.set_fontsize(11)

    ax.grid(
        which='major',
        linewidth=0.5,
        color='black')  # Основные линии сетки
    ax.grid(which='minor', linestyle='--', color='gray',
            linewidth=0.5)  # промежуточные линии сетки
    # Авто расположение сетки по х
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    # Авто расположение сетки по у
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    return fig, ax


def plot_curves(
        ax,
        data_dict: dict,
        legend_col: int = 2,
        bbox=(0.05, 0.0, 0.85, 0.0)):
    r"""
    Строит графики на области построения соответствующей ax.

    Значения (данные для построения) и подписи содержатся в словаре data_dict:
    {
     'xdata' : [данные по х]
     'ydata' : [[данные по у]]
     'data_lables': ['Вертикальная нагрузка Py',
                     'Вертикальная нагрузка Px', 'Pz'],
     'fig_title': 'Вертикальная нагрузка Py',
     'ax_x': r'$\alpha > \beta$',
     'ax_y': 'y1, y2'
     }

    Parameters
    ----------
    ax : matplotlib object
        График матплотлиб.
    data_dict : dict
        Словарь с данными.
    legend_col : int, optional
        Кол-во столбцов в легенде. The default is 2.

    Returns
    -------
    None.

    """
    # Шрифты, используемые на графике
    font_std_11 = mpl.font_manager.FontProperties(
        family='times new roman', size=11)
    font_bold = mpl.font_manager.FontProperties(
        family='times new roman', size=12, weight='bold')
    font_bold_11 = mpl.font_manager.FontProperties(
        family='times new roman', size=11, weight='bold')

    ax.set_xlim(data_dict['xdata'][0],
                data_dict['xdata'][-1])  # Задание нуля по х
    ax.set_title(
        data_dict['fig_title'],
        fontproperties=font_bold)  # Заголовок диаграммы
    ax.set_xlabel(
        data_dict['ax_x'],
        fontproperties=font_bold_11,
        labelpad=0)  # Название оси х
    ax.set_ylabel(
        data_dict['ax_y'],
        fontproperties=font_bold_11,
        labelpad=1)  # Название оси у

    # Определение числа пропусков значений для создания маркера (всего 15
    # маркеров на графике)
    mark = len(data_dict['xdata']) // 15
    mark = mark if mark > 0 else 1  # 1 если равно нулю

    # Используемые типы маркеров
    markers = ['.', 'v', '2', 's', '+', '_', 'x', '*']
    m_size = [8, 6, 8, 5, 8, 8, 7, 8]  # РАзмеры маркеров

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b']  # Цвета графиков
    line_styles = [
        '-',
        '-',
        '-',
        '-',
        '-',
        '-',
        '--',
        '--']  # Стили линий графиков

    if (len(data_dict['ydata']) -
            1) // 8 > 0:  # Если кол-во кривых больше 8, то списки расширяются
        n = ((len(data_dict['ydata']) - 1) // 8) + 1
        markers = markers * n
        m_size = m_size * n
        colors = colors * n
        line_styles = line_styles * n

    i = 0
    # для каждого списка данных, рисование кривой
    for ydata in data_dict['ydata']:
        ax.plot(data_dict['xdata'],
                ydata,
                linewidth=0.8,
                label=data_dict['data_lables'][i],
                color=colors[i],
                linestyle=line_styles[i],
                marker=markers[i],
                markevery=(i * (mark // len(data_dict['data_lables'])),
                           mark),
                ms=m_size[i])
        i += 1

    # Создание легенды
    ax.legend(
        bbox_to_anchor=bbox,
        bbox_transform=plt.gcf().transFigure,
        loc='upper left',
        borderaxespad=0,
        mode='expand',
        ncol=legend_col,
        framealpha=None,
        facecolor='w',
        edgecolor='black',
        prop=font_std_11)


def df_plot(
        df: pd.DataFrame,
        lables_dict: dict,
        plot_size: str = 'A5',
        legend_col: int = 2):
    """
    Строит графики по столбцам Таблицы пандас. Преобразует Таблицу \
        в список и передает его \
    функции строяшщей графики. Возвращает объект figure matplotlib.

    Parameters
    ----------
    df : pd.DataFrame
        Таблица для построения графика.
    lables_dict : dict
        Словарь, содержащий заголовки и подписи осей.
    plot_size : str, optional
        Размер графика . The default is 'A5'.
    legend_col : int, optional
        Количество столбцов в легенде. The default is 2.

    Returns
    -------
    fig : figure matplotlib
        Объект области построенного графика.
    """
    # Преобразование Таблиц в списки
    xdata = df.index.tolist()  # Данные по оси Х
    ydata = df.T.values.tolist()  # Данные по оси Y
    # Добавление в словарь элементов - списков с данными по оси х и у
    lables_dict['xdata'] = xdata  # Список данных по х
    lables_dict['ydata'] = ydata  # Список данных по y

    fig, ax = figure_cre(plot_size)  # Создание пустого графика

    if plot_size == 'A4':
        plot_curves(ax, lables_dict, legend_col, bbox=(
            0.05, 0.0, 0.85, 0.05))  # Построение кривых
    else:
        plot_curves(ax, lables_dict, legend_col)

    return fig


def plot_mg_loads(mg_data: [pd.DataFrame], gear_res: dict, path: pathlib.Path):
    """
    Строит графики для нагрузок на шасси ООШ. \
    Здесь задаются подписи осей графиков.

    Parameters
    ----------
    mg_data : [pd.DataFrame]
        Список Таблиц, содержащих данные результатов Адамс.
    gear_res : {[pd.Dataframe]}
        Словарь с Таблицами Результатов расчета программы.
    path : pathlib.Path
        Путь к папке с результатами.

    Returns
    -------
    None.

    """
    path.mkdir()  # создание дирректории для записи рисунков
    # Словарь, содержащий параметры для создания графика
    plot_dict = {
        'data_lables': [
            r'$P_x^{ПООШ}$',
            r'$P_y^{ПООШ}$',
            r'$P_x^{ЗООШ}$',
            r'$P_y^{ЗООШ}$'],
        'fig_title': 'Действующие нагрузки на ООШ',
        'ax_x': 'Время, [с]',
        'ax_y': r'Нагрузка на опору $P_x$, $P_y$, [тс]'}

    # Изначальное значение заголовка, для изменения его в цикле
    title = plot_dict['fig_title']

    i = 0  # Нумератор
    for df in mg_data:
        # Определение номера случая в соответствии с таблицей
        case = int(gear_res['MG_max'][0]['number']['none'][i])

        plot_dict['fig_title'] = title + ', случай №' + \
            str(case)  # Добавление номера случая к заголовку
        # построение графика
        fig = df_plot(mg_data[0].loc[slice(None), (['f_wh'], [3, 5], [
                      'x', 'y'])], plot_dict, 'A5', legend_col=4)
        if len(plot_dict['data_lables']) > 2:  # Изменение его высоты
            fig.set_figheight(fig.get_figheight() - 0.15 *
                              (len(plot_dict['data_lables']) // 2))

        # Сохрангение картинки
        jpeg_name = 'mg_n' + str(case) + '.jpeg'
        fig.savefig(
            path / jpeg_name,
            edgecolor='black',
            dpi=300,
            bbox_inches='tight')
        (path / jpeg_name).chmod(0o777)
        i += 1


def plot_fg_loads(fg_data: [pd.DataFrame], gear_res: dict, path: pathlib.Path):
    """
    Строит графики для нагрузок на шасси ПОШ. \
    Здесь задаются подписи осей графиков.

    Parameters
    ----------
    fg_data : [pd.DataFrame]
        Список Таблиц, содержащих данные результатов Адамс.
    gear_res : dict
        Словарь с Таблицами Результатов расчета программы.
    path : pathlib.Path
        Путь к папке с результатами.

    Returns
    -------
    None.

    """
    path.mkdir()  # создание дирректории для записи рисунков
    # Словарь, содержащий параметры для создания графика
    plot_dict = {
        'data_lables': [r'$P_x^{ПОШ}$', r'$P_y^{ПОШ}$'],
        'fig_title': 'Действующие нагрузки на ПОШ',
        'ax_x': 'Время, [с]',
        'ax_y': r'Нагрузка на опору $P_x$, $P_y$, [тс]'
    }

    # Изначальное значение заголовка, для изменения его в цикле
    title = plot_dict['fig_title']

    i = 0  # Нумератор
    for df in fg_data:
        # Определение номера случая в соответствии с таблицей
        case = int(gear_res['FG'][0]['number']['none'][i])

        plot_dict['fig_title'] = title + ', случай №' + \
            str(case)  # Добавление номера случая к заголовку
        # построение графика
        fig = df_plot(fg_data[0].loc[slice(None), (['f_wh'], [1], [
                      'x', 'y'])], plot_dict, 'A5', legend_col=4)
        if len(plot_dict['data_lables']) > 2:  # Изменение его высоты
            fig.set_figheight(fig.get_figheight() - 0.15 *
                              (len(plot_dict['data_lables']) // 2))

        # Сохрангение картинки
        jpeg_name = 'mg_n' + str(case) + '.jpeg'
        fig.savefig(
            path / jpeg_name,
            edgecolor='black',
            dpi=300,
            bbox_inches='tight')
        (path / jpeg_name).chmod(0o777)
        i += 1


def get_overload_lables(df1: pd.DataFrame):
    """
    Возвращает список названий кривых, определенный по таблице пандас.

    Parameters
    ----------
    df1 : pd.DataFrame
        Таблица пандас, для которой планируется построить графики.

    Returns
    -------
    lables : list
        Список названий кривых.

    """
    lables = []
    for lbl in df1.index:
        if lbl == 'MAX' or lbl == 'MIN':
            lables.append('Огибающая')
        else:
            lables.append('Случай №' + str(int(df1['number'][lbl])))
    return lables


def plot_overloads(dict1: dict, path: pathlib.Path, c_amount: int = 8):
    """
    Основная программа для построения графиков перегрузок.

    Parameters
    ----------
    dict1 : dict
        Словарь с результатами по перегрузкам.
    c_amount : int
        Количество кривых на графике.
    path : pathlib.Path
        Путь к папке с результатами.

    Returns
    -------
    None.

    """
    path.mkdir()  # создание дирректории для записи рисунков

    for key in dict1:
        if '_l_' not in key and '_r_' not in key:
            key_path = path / key
            os.mkdir(key_path)  # создание дирректории для записи рисунков
            if 'kr_' in key:
                plot_dict = {
                    'data_lables': '',
                    'fig_title': 'Распределение перегрузок по размаху \
                        полукрыла. Расчетные значения',
                    'ax_x': 'Координата по оси 0X, [м]',
                    'ax_y': r'Перегрузки $n_x$ / $n_y$'}
            elif 'go_' in key:
                plot_dict = {
                    'data_lables': '',
                    'fig_title': 'Распределение перегрузок по полуразмаху \
                        ГО. Расчетные значения',
                    'ax_x': 'Координата по оси 0Z, [м]',
                    'ax_y': r'Перегрузки $n_x$ / $n_y$'}
            elif 'fuz_' in key:
                plot_dict = {
                    'data_lables': '',
                    'fig_title': 'Распределение перегрузок по \
                        фюзеляжу. Расчетные значения',
                    'ax_x': 'Координата по оси 0Z, [м]',
                    'ax_y': r'Перегрузки $n_y$'}
            else:
                continue

            n = c_amount
            for i in range((len(dict1[key].index.tolist()) // n) + 1):
                start = 0 + n * i
                stop = (start + n
                        if start + n < (len(dict1[key].index.tolist()) - 1)
                        else (len(dict1[key].index.tolist()) - 1))

                df_to_plot = dict1[key].iloc[start: stop]
                df_to_plot = df_to_plot.append(
                    dict1[key].iloc[(len(dict1[key].index.tolist()) - 1)])

                # Создание списка подписей данных
                plot_dict['data_lables'] = get_overload_lables(df_to_plot)
                del df_to_plot['number']

                fig = df_plot(df_to_plot.T, plot_dict, 'A4', legend_col=4)
                if len(plot_dict['data_lables']) > 4:  # Изменение его высоты
                    fig.set_figheight(fig.get_figheight(
                    ) - 0.15 * (len(plot_dict['data_lables']) // 4))

                # Сохранение картинки
                jpeg_name = 'acc_'+str(i + 1)+'.jpeg'
                fig.savefig(key_path / jpeg_name,
                            edgecolor='black',
                            dpi=300,
                            bbox_inches='tight')
                (key_path / jpeg_name).chmod(0o777)


def plot_dv_overloads(dv_data: [pd.DataFrame],
                      res_dv: dict,
                      path: pathlib.Path):
    """
    Строит графики для перегрузок двигателей. \
    Здесь задаются подписи осей графиков.

    Parameters
    ----------
    dv_data : [pd.DataFrame]
        Список Таблиц, содержащих данные результатов Адамс.
    res_dv : dict
        Словарь с Таблицами Результатов расчета программы.
    path : str
        Путь к папке с результатами.

    Returns
    -------
    None.

    """
    path.mkdir()  # создание дирректории для записи рисунков
    # Словарь, содержащий параметры для создания графика
    plot_dict = {
        'data_lables': [
            r'$n_x^{ц.т. СУ}$',
            r'$n_y^{ц.т. СУ}$',
            r'$n_z^{ц.т. СУ}$'],
        'fig_title': 'Действующие перегрузки на СУ',
        'ax_x': 'Время, [с]',
        'ax_y': r'Перегрузка $n_x$, $n_y$, $n_z$'}

    # Изначальное значение заголовка, для изменения его в цикле
    title = plot_dict['fig_title']

    i = 0
    for df in dv_data:
        # Определение номера случая в соответствии с таблицей
        case = int(res_dv[[*res_dv][0]]['number']['none'][i])

        # построение графика
        col_used_0 = df.columns.remove_unused_levels()  # Используемые колонки
        # Цикл по используемым инменам колонок на 0 уровне
        for lvl_0 in col_used_0.levels[0]:
            # Используемые колонки
            col_used_1 = df[lvl_0].columns.remove_unused_levels()
            # Цикл по используемым колонкам на первом уровне
            for lvl_1 in col_used_1.levels[0]:

                # Добавление номера случая к заголовку
                plot_dict['fig_title'] = title + ' №' + \
                    str(int(lvl_1)) + ', случай №' + str(case)
                fig = df_plot(df[lvl_0][lvl_1], plot_dict, 'A5', legend_col=4)
                if len(plot_dict['data_lables']) > 2:  # Изменение его высоты
                    fig.set_figheight(fig.get_figheight(
                    ) - 0.15 * (len(plot_dict['data_lables']) // 2))

                # Сохрангение картинки
                jpeg_name = 'dv_n'+str(case)+'_'+str(int(lvl_1))+'.jpeg'
                fig.savefig(
                    path / jpeg_name,
                    edgecolor='black',
                    dpi=300,
                    bbox_inches='tight')
                (path / jpeg_name).chmod(0o777)
                del(fig)
        i += 1

# %%% Запись в Excel


def df_style(df1: pd.DataFrame):
    """
    Применяет стандартный стиль к Таблице, возвращает стилизованную таблицу.

    Parameters
    ----------
    df1 : pd.DataFrame
        Таблица пандас.

    Returns
    -------
    df_styled: styled pd.DataFrame

    """
    prop_data = [
        ('font-size', '14px'),
        ('text-align', 'right'),
        ('border-style', 'solid'),
        ('border-width', '1px'),
        ('background-color', 'none')]
    prop_names = [
        ('font-size', '14px'),
        ('font-weight', 'bold'),
        ('text-align', 'center'),
        ('border-style', 'solid'),
        ('border-width', '3px')]
    styles = [
        dict(selector="th", props=prop_names),  # Заголовки
        dict(selector="td", props=prop_data)  # Данные
    ]

    df_styled = df1.style\
        .highlight_max(color='#fea993')\
        .highlight_min(color='#a2cffe')

    # df_styled = df1.style\
    #     .set_table_styles(styles)\
    #     .highlight_max(color='#fea993')\
    #     .highlight_min(color='#a2cffe')
#        .format(format_dict(df1,{'P':'{:.2f}', 'S':'{:.1f}'},level = 1))\

    return df_styled


def df_to_excel(writer, df1: pd.DataFrame, sheet: str, row: int,):
    """
    Записывает таблицу на лист excel начиная с заданной строки.

    Parameters
    ----------
    writer : xlsxwriter writer
        Запущенный процесс записи в эксель???.
    df : pd.DataFrame
        Таблица для записи.
    sheet : str
        Имя листа эксель.
    row : int
        Номер строки с которой начнется запись.

    Returns
    -------
    None.

    """
    # engine='xlsxwriter'
    # engine='openpyxl'
    df_style(df1).to_excel(
        writer,
        sheet_name=sheet,
        startrow=row,
        float_format='%.2f')


def to_excel_list(writer, tables_list: [pd.DataFrame], sheet_name: str):
    """
    Записывает на один лист документа Excel Таблицы из списка.

    Parameters
    ----------
    writer : xlsxwriter writer
        Запущенный процесс записи в эксель???.
    tables_list : [pd.DataFrame]
        Список таблиц для записи.
    sheet_name : str
        Имя листа в документе эксель.

    Returns
    -------
    None.

    """
    row_n = 0
    for df1 in tables_list:
        df_to_excel(writer, df1, sheet_name, row=row_n)
        row_n += (df1.shape[0] + 5)


# %%% Диалоговые окна

import tkinter as tk
from tkinter import filedialog

def dialog_open_file() -> str:
    """
    Открывает диалоговое окно открытия файла, возвращает имя файла.

    Returns
    -------
    str
        Имя файла для открытия.

    """
    root = tk.Tk()  # Объект окна верхнего уровня
    root.withdraw()  # Скрываем окно верхнего уровня
    root.wm_attributes('-topmost', 1)  # Фокус на объекте верхнего уровня

    # Запись в переменную имени файла
    file_path = filedialog.askopenfilename(parent=root)

    # Удаление окна верхнего уровня
    root.destroy()

    return file_path

# %% Не отсортированные функции

def find_df_columns(df: DataFrame,
                    search_string: AnyStr,
                    level: Any = 'max') -> List[Any]:
    """
    Возвращает список кортежей (путь для срезки колонок) для колонок,
    в названии которых на выбранном уровне содержится искомая стока.
    Если уровень не выбран, то поиск производится по максимальному (самому
    нижнему) уровню.

    Parameters
    ----------
    df: Таблица пандас
    search_string: искомая строка
    level: уровень поиска - целое число или строка.

    Returns
    -------
    Список кортежей с путями для среза по колонкам.
    """
    # Максимально возможный уровень в именах колонок
    max_level = len(df.columns.values[0])-1
    # Уровень поиска.
    level = max_level if (level == 'max' or level > max_level) else level
    return [tpl
            for tpl in df.columns.values
            if search_string in tpl[level]]


def df_column_multiply(df: DataFrame,
                       search_string: AnyStr,
                       multiplier: Any,
                       level: Any = 'max') -> NoReturn:
    """
    Домножает столбцы таблицы пандас, в названии которых на заданном
    уровне содержится искомая строка, на множитель используя метод .mul().
    При Этом изменяется исходная таблица!
    Parameters
    ----------
    df: таблица пандас.
    search_string: искомая строка.
    multiplier: множитель - что угодно (число, массив, таблица ...).
    level: уровень поиска.
    """
    # Определение колонок, значения которых необходимо домножить
    columns_to_multiply = find_df_columns(df, search_string, level)
    # Домножение:
    for column_path in columns_to_multiply:
        df.loc[:, column_path] = df.loc[:, column_path].mul(multiplier)