# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 15:11:16 2020.

@author: ulyanovas
"""
import shutil
from typing import Dict, Any, List, Union, Tuple, NoReturn

from pandas import DataFrame, Series

import my_decorators as mdec
import dl_functions as dlf
import pandas as pd
import pathlib
import matplotlib.pyplot as plt

def main(params: dict):
    # %% Исходные данные
    init_data_file_name = params['dl_path']
    order = params['butfilt_order']
    lowcut = params['butfilt_cuttoff']
    landing_type = ('sym' if params['simetric'] else 'asim')

    workdir_path = pathlib.Path(init_data_file_name).parent.absolute()

    assert landing_type == 'sym', 'Only symmetric landing supported'

    # %% Считывание файлов исходных данных и перевод в СИ

    with open(init_data_file_name, 'r') as tfile:
        text = tfile.read()

    text = text.split('\n')
    text = [x for x in text if x != '']
    text = [x for x in text if x[0] != '#']

    # Считывание из текста таблицы исходных данных в Таблицу Пандас, перевод в СИ
    df_init = dlf.df_to_SI(dlf.df_from_text(text, '[INITIAL_DATA_TABLE]'))
    # Считывание из текста таблицы соответствия измерителей в Таблицу Пандас,
    # перевод в СИ
    df_init_ad = dlf.df_to_SI(dlf.df_from_text(text, '[ADAMS_DATA_TABLE]'))
    # Корректировка от точки названий измерителей
    df_init_ad['me_name'] = [dlf.check_content(
        x, '.', True) for x in df_init_ad['me_name']]
    # Формирование левой части для ТАблиз с результатами
    # Список с колонками левой части
    lst_left = dlf.list_from_text(text, '[LEFT_OUTPUT_TABLE]')
    # Таблица в итоговой системе единиц измерения
    df_left = dlf.units_from_SI(df_init[lst_left[1]], lst_left[0])

    # Формирование списка с таблицами обработанных (отфильтрованных) результатов
    all_adams_res = []  # Пустой список с таблицами обработанных результатов
    # df_init.index:
    for i in range(df_init.index.start, 4, df_init.index.step):  # !!!
        # Импорт файла результатов Adams
        data_file_path = workdir_path / df_init['data_file_name'][i]
        adams_res = dlf.import_adss_as_df(data_file_path)
        # Корректировка от точки названий столбцов
        adams_res.columns = [
            dlf.check_content(
                x, '.', True) for x in adams_res.columns]
        # Фильтрация импортированного файла для соответсвия названия его столбцов
        # именам измерителей в файле исходных данных
        adams_res = dlf.ad_res_filter(df_init_ad['me_name'], adams_res)
        # Перевод в систему СИ
        data = adams_res.to_numpy()
        units = list(df_init_ad['units'])
        df_adams_res_SI = pd.DataFrame(
            dlf.units_to_SI(units, data),
            columns=adams_res.columns)  # Перевод в Pandas
        # фильтрация данных
        dlf.df_butfilt(df_adams_res_SI, order, lowcut, 2)

        # Задание индексов для рядов значений (индексы - моменты времени)
        df_adams_res_SI = df_adams_res_SI.set_index(df_adams_res_SI.columns[0])

        all_adams_res.append(df_adams_res_SI)  # Наполнение списка

    # %% Таблица с полным Мультииндексом

    df_init_ad = df_init_ad[
        ~df_init_ad['type'].isin(['time'])]  # Реверс изин ~

    # Формирование массива с индексами (мультииндекс)
    df_col = pd.MultiIndex.from_frame(df_init_ad)

    all_full_data = []
    for df_res in all_adams_res:
        associated_data = df_res[df_init_ad['me_name']]
        associated_data.columns = df_col
        all_full_data.append(associated_data)

    # %% Перегрузки в двигателях

    # noinspection PyTupleAssignmentBalance
    all_dv_data, res_dv = engine_ovlds_calculation(all_full_data, df_init)

    # raise SystemExit
    # %%  Распределение перегрузок по фюзеляжу

    ltype = landing_type
    assert ltype == 'sym', 'Поддерживается только симметричный вариант посадки'

    # Расчет перегрузок для симметричного случая
    res_ovlds = dlf.calc_overloads(all_adams_res, df_init_ad, df_init)

    # %% Нагрузки на шасси

    # Расчет нагрузок на шасси
    all_fg_data, all_mg_data, res_gear = gear_loads_calculation(
        all_full_data, df_init)

    # %%%

    # Перевод в СК для отчета
    convert_from_si(all_fg_data, all_mg_data, res_gear)
    # Добавление левой части таблицы
    add_left_to_tables(df_left, res_dv, res_gear)

    # raise SystemExit
    # %% Вывод результатов

    # Создание дирректории для результатов
    results_path = create_results_dir(workdir_path)
    # %%% создание графиков

    plot_diagrams(all_dv_data, all_fg_data, all_mg_data,
                  res_dv, res_gear, res_ovlds,
                  results_path)

    # raise SystemExit#!!!
    # %%% вывод в excel
    write_results_to_excel(res_gear, res_dv, res_ovlds,
                           results_path / 'Dl_results.xlsx')


@mdec.logtime('Создание полной таблицы с нагрузками на шасси')
def all_lg_data_cre(all_adams_res, df_init_ad, landing_type):
    ltype = landing_type
    assert ltype == 'sym', 'Поддерживается только симметричный вариант посадки'
    # Формирование блока данных с результатами для шасси и мультииндексом
    # Фильтр массива исходных данных по агрегату landing_gear
    df_lnd_gear_init = df_init_ad[
        df_init_ad['aggregate'].isin(['landing_gear'])]
    df_lnd_gear_init = df_lnd_gear_init[[
        'type', 'agr_number', 'component',
        'me_name']]  # Удаление лишних столбцов
    # Формирование массива с индексами (мультииндекс)
    df_col_multiindex = pd.MultiIndex.from_frame(df_lnd_gear_init)
    # Пустой список с таблицами результатов по шасси и мультииндексом
    all_lndgear_data = []
    for df_res in all_adams_res:
        # Выборка столбцов с данными для шасси
        df_lndgear_data = df_res[df_lnd_gear_init['me_name']]
        # добавление мультииндекса к данными
        df_lndgear_data.columns = df_col_multiindex
        df_lndgear_data = df_lndgear_data.sort_index(
            axis=1)  # Сортировка по названию столбцов
        all_lndgear_data.append(df_lndgear_data)  # Наполнение списка
    return all_lndgear_data


@mdec.logtime('Добавление левой части к таблицам')
def add_left_to_tables(df_left: DataFrame,
                       res_dv: Dict[Any, DataFrame],
                       res_gear: Dict[Any, Any]) -> NoReturn:
    """
    Добавление левой части (левой шапки) к таблицам с результатами.
    Функция изменяет входящие в нее объекты.
    Parameters
    ----------
    df_left: Таблица с левой частью.
    res_dv: Словарь с результатами по СУ.
    res_gear: Словарь с результатами по шасси.
    """
    for key in res_dv:
        # Добавление левой части Таблицам
        res_dv[key] = pd.concat([df_left, res_dv[key]], axis=1, join='inner')
    for key in res_gear:  # Для всех ключей в словаре с результатами
        lst = res_gear[key]  # Более короткая ссылка на список Таблиц
        for i in range(len(lst)):
            # Добавление левой части Таблицам
            lst[i] = pd.concat([df_left, lst[i]], axis=1, join='inner')


@mdec.logtime('Конвертация единиц измерения из СИ в рабочую')
def convert_from_si(all_fg_data: List[Any],
                    all_mg_data: List[Union[DataFrame, Any]],
                    res_gear: Dict[Any, Any]) -> NoReturn:
    """
    Конвертация из СИ в расчетную систему единиц измерения.
    Функция изменяет входящие объекты.
    Parameters
    ----------
    all_fg_data: Список таблиц с переходными процессами ПОШ.
    all_mg_data: Список таблиц с переходными процессами ООШ.
    res_gear: Словарь с таблицами с результатами по нагрузкам на шасси.
    """
    dlf.gear_data_to_SI(all_mg_data)  # Перевод в СИ
    dlf.gear_data_to_SI(all_fg_data)  # Перевод в СИ
    for key in res_gear:  # Для всех ключей в словаре с результатами
        lst = res_gear[key]  # Более короткая ссылка на список Таблиц
        for i in range(len(lst)):  # Для всех таблиц в списке
            # Список единиц измерения
            units = dlf.units_from_names(
                list(lst[i].columns.get_level_values(1)))
            # Переход к эдиницам из списка
            lst[i] = dlf.units_from_SI(lst[i], units)


@mdec.logtime('Расчет перегрузок по агрегатам')
def engine_ovlds_calculation(
        all_full_data: List[Union[Union[Series, DataFrame, None], Any]],
        df_init: DataFrame,
        ) -> Tuple[List[Union[DataFrame, Series]], Dict[Any, DataFrame]]:
    """
    Расчет перегрузок в ц.т. СУ
    Parameters
    ----------
    all_full_data: список Таблиц с полным мультииндексом, в которых
    содержаться данные расчета Адамс (переходные процессы).
    df_init: Таблица пандас, содержащая исходные данные.

    Returns
    -------
    all_dv_data: Список таблиц, содержащих переходные процессы по СУ,
    для дальнейшей их отрисовки.
    res_dv: Словарь с таблицами результатов для двигателей.
    """
    all_dv_data = []  # Список таблиц с данными по двигателям
    for df_res in all_full_data:
        dv_data = pd.concat([df_res.xs('dv_l',
                                       level='aggregate',
                                       axis=1,
                                       drop_level=False),
                             df_res.xs('dv_r',
                                       level='aggregate',
                                       axis=1,
                                       drop_level=False)],
                            axis=1)
        dv_data.columns = dv_data.columns.droplevel(
            ['me_name', 'units', 'me_loc_x', 'me_loc_y', 'type'])
        dv_data = dv_data.reorder_levels(
            ['aggregate', 'agr_number', 'component', 'me_loc_z'], axis=1)
        all_dv_data.append(dv_data)
    # Вычисление комбинаций по двигателям
    res_dv = dlf.calc_dv_overloads(all_dv_data, df_init)
    return all_dv_data, res_dv


@mdec.logtime('Расчет нагрузок на шасси')
def gear_loads_calculation(
        all_full_data: List[Union[Union[Series, None, DataFrame], Any]],
        df_init: DataFrame
        ) -> Tuple[List[Any], List[Union[DataFrame, Any]], Dict[Any, Any]]:
    """
    Расчет нагрузок на шасси.
    Parameters
    ----------
    all_full_data: переходные процессы (данные Адамс) для всех
    расчетных случаев.
    df_init: Таблица пандас с исходными данными.

    Returns
    -------
    all_fg_data: переходные процессы для передней опоры шасси
    all_mg_data: переходные процессы для основных опор шасси
    res_gear: Словарь с таблицами результатов по всем опорам шасси.
    """
    all_lndgear_data = []
    for df_res in all_full_data:
        lg_data = df_res.xs('landing_gear', level='aggregate',
                            axis=1, drop_level=False)
        lg_data.columns = lg_data.columns.droplevel(
            ['units', 'me_loc_x', 'me_loc_y', 'me_loc_z', 'aggregate',])
        lg_data = lg_data.reorder_levels(
            ['type', 'agr_number', 'component', 'me_name'], axis=1)
        all_lndgear_data.append(lg_data)

    # Словарь со списками таблиц, содержащих результаты расчета нагрузок
    res_gear = {}

    # Расчет нагрузок ПОШ
    all_fg_data = dlf.calc_fg_loads(all_lndgear_data,
                                               df_init,
                                               res_gear)

    # Расчет нагрузок ООШ
    if all_lndgear_data[0].columns.get_level_values(
            'agr_number').max() == 5:  # 4 основные опоры шасси
        all_mg_data = dlf.calc_mg_loads_4(
            all_lndgear_data,
            df_init,
            res_gear)  # Расчет нагрузок для 4 ООШ

    # 2 основные опоры шасси
    elif all_lndgear_data[0].columns.get_level_values('agr_number').max() == 3:
        raise AssertionError('Расчет для двух основных опор не доделан')

    else:  # Хз сколько основных опор шасси
        raise AssertionError('Неподдерживаемое количество опор')
    # Домножение таблиц на коэффициент безопасности
    for key in res_gear:
        for i in range(len(res_gear[key])):
            # Множитель:
            multiplier_slice = df_init['safety_factor'].loc[
                res_gear[key][i].index]
            # Умножение:
            dlf.df_column_multiply(res_gear[key][i], 'P', multiplier_slice)
    return all_fg_data, all_mg_data, res_gear


@mdec.logtime('Запись результатов в Excel')
def write_results_to_excel(results_gears: Dict[Any, Any],
                           results_engine: Dict[Any, DataFrame],
                           results_overloads: Dict[str, DataFrame],
                           file_name: pathlib.Path) -> None:
    """
    Запись Таблиц с результатами в файл excel.
    Parameters
    ----------
    results_gears: Таблицы с результатами для опор шасси.
    results_engine: Таблицы с результатами для СУ.
    results_overloads: Таблицы с результатами для перегрузок по агрегатам.
    file_name: Полное имя файла, в который будет производиться запись.
    """
    wr = pd.ExcelWriter(file_name, engine='xlsxwriter')
    # Нагрузки на ПОШ
    dlf.to_excel_list(wr, results_gears['FG'], 'FG')
    # Нагрузки на ООШ
    dlf.to_excel_list(wr, results_gears['MG_max'], 'MG_max')
    # Комбинации ООШ
    dlf.to_excel_list(wr, results_gears['MG_comb'], 'MG_comb')
    # Совместные комбинации ООШ
    dlf.to_excel_list(wr, results_gears['MG_cross_comb'], 'MG_crosscomb')
    # Перегрузки по двигателю
    dlf.to_excel_list(wr, [results_engine[key]
                           for key in [*results_engine]], 'n_dv')
    # Перегрузки по фюзеляжу
    dlf.to_excel_list(wr, [results_overloads[key]
                           for key in [*results_overloads]
                           if 'fuz' in key], 'n_fuz')
    # Перегрузки по ГО
    dlf.to_excel_list(wr, [results_overloads[key]
                           for key in [*results_overloads]
                           if 'go' in key], 'n_go')
    # Перегрузки по крылу
    dlf.to_excel_list(wr, [results_overloads[key]
                           for key in [*results_overloads]
                           if 'kr' in key], 'n_kr')
    wr.save()


@mdec.logtime('Создание диаграмм')
def plot_diagrams(all_engine_data: [[pd.DataFrame]],
                  all_front_gear_data: List[Any],
                  all_main_gear_data: [[pd.DataFrame]],
                  results_engine: Dict[Any, DataFrame],
                  results_gears: dict,
                  results_overloads: Dict[str, pd.DataFrame],
                  results_path: pathlib.Path) -> None:
    """
    Построение и отрисовка диаграмм (графиков) для полученных результатов.

    Parameters
    ----------
    all_engine_data: результаты Адамс по всем случаям для двигателей
    all_front_gear_data: результаты Адамс по всем случаям для ПОШ
    all_main_gear_data: результаты Адамс по всем случаям для ООШ
    results_engine: Таблицы обработанных результатов для СУ
    results_gears: Таблицы обработанных результа тов для опор шасси
    results_overloads: Таблицы обработанных результатов для перегрузок
    results_path: Путь к папке с результатами.
    """
    # Диаграммы перегрузок двигателей
    dlf.plot_dv_overloads(all_engine_data,
                          results_engine,
                          results_path / 'dv_overloads')
    # Диаграммы для перегрузок
    dlf.plot_overloads(results_overloads, results_path / 'Overloads', 8)
    # Графики ООШ и ПОШ
    dlf.plot_mg_loads(all_main_gear_data, results_gears,
                      results_path / 'MG_loads')
    dlf.plot_fg_loads(all_front_gear_data, results_gears,
                      results_path / 'FG_loads')
    plt.close('all')  # Удаление всех созданных фигур


@mdec.logtime('Создание дирректории с результатами')
def create_results_dir(workdir_path: pathlib.Path,
                       res_dir: str = '_RES_') -> pathlib.Path:
    """
    Создает пустую директорию для записи результатов. Сперва проверяет не
        создана ли директория с подобным именем, если создана, то удаляет ее.

    Parameters
    ----------
    workdir_path: Путь к рабочей директории
    res_dir: Имя директории с результатами

    Returns
    -------
    Путь к директории с результатами
    """
    for x in workdir_path.iterdir():  # Проверка наличия дирректории
        if workdir_path / res_dir == x:
            shutil.rmtree(workdir_path / res_dir)
    (workdir_path / res_dir).mkdir()  # Создание дирректории

    return workdir_path / res_dir


if __name__ == '__main__':
    params = {
        'dl_path': 'D:\Work\Python\DL\DL_data\Innit_data.txt',
        'butfilt_order': 3,
        'butfilt_cuttoff': 30.0,
        'simetric': True,
        'design_case_start_number': 1,
        'design_case_end_number': 4,
    }

    main(params)
    fin_time = mdec.LogTime.get_time('norm')