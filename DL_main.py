# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 15:11:16 2020.

@author: ulyanovas
"""
import shutil
import os
import dl_functions as dlf
import pandas as pd
import numpy as np
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
    my_time = dlf.get_time()

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

    my_time('Read init files')  # Время выполнения программы

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

    my_time('Read and butfilt files')  # Время выполнения программы
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

    my_time('Full Multiindex Tables')  # Время выполнения программы
    # %% Перегрузки в двигателях

    all_dv_data = []  # Список таблиц с данными по двигателям
    for df_res in all_full_data:
        dv_data = pd.concat([df_res.xs('dv_l', level='aggregate',
                                       axis=1, drop_level=False),
                             df_res.xs('dv_r', level='aggregate',
                                       axis=1, drop_level=False)], axis=1)
        dv_data.columns = dv_data.columns.droplevel(
            ['me_name', 'units', 'me_loc_x', 'me_loc_y', 'type'])
        dv_data = dv_data.reorder_levels(
            ['aggregate', 'agr_number', 'component', 'me_loc_z'], axis=1)
        all_dv_data.append(dv_data)

    # Вычисление комбинаций по двигателям
    res_dv = dlf.calc_dv_overloads(all_dv_data, df_init)

    my_time('Calculate ENG overloads')  # Время выполнения программы
    # raise SystemExit
    # %%  Распределение перегрузок по фюзеляжу

    ltype = landing_type
    assert ltype == 'sym', 'Поддерживается только симметричный вариант посадки'

    # Расчет перегрузок для симметричного случая
    res_ovlds = dlf.calc_overloads(all_adams_res, df_init_ad, df_init)

    my_time('Calculate FUZ and WNG overloads')  # Время выполнения программы
    # %% Нагрузки на шасси

    ltype = landing_type
    assert ltype == 'sym', 'Поддерживается только симметричный вариант посадки'

    # Словарь со списками таблиц, содержащих результаты расчета нагрузок
    res_gear = {}

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
    # !!! Пример: df_tmp1 = df_lndgear_data.xs('y', level = 'component',axis = 1)

    my_time('Create Gears tables')  # Время выполнения программы
    # %%% ПОШ

    # Расчет нагрузок ПОШ
    all_fg_data = dlf.calc_fg_loads(all_lndgear_data, df_init, res_gear)

    my_time('Calculate FG loads')  # Время выполнения программы
    # %%% ООШ

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

    my_time('Calculate MG loads')  # Время выполнения программы
    # %%% Перевод в СК для отчета, Добавление левой части таблицы

    for key in res_dv:
        # Добавление левой части Таблицам
        res_dv[key] = pd.concat([df_left, res_dv[key]], axis=1, join='inner')

    for key in res_gear:  # Для всех ключей в словаре с результатами
        lst = res_gear[key]  # Более короткая ссылка на список Таблиц
        # Для всех таблиц в списке переходим к отчетной системе единиц измерения
        for i in range(len(lst)):
            lst[i] = dlf.units_from_SI(lst[i], dlf.units_from_names
            (list(lst[i].columns.get_level_values(1))))
            # Добавление левой части Таблицам
            lst[i] = pd.concat([df_left, lst[i]], axis=1, join='inner')

    dlf.gear_data_to_SI(all_mg_data)  # Перевод в СИ
    dlf.gear_data_to_SI(all_fg_data)  # Перевод в СИ

    my_time('Transform Units')  # Время выполнения программы
    # raise SystemExit
    # %% Вывод результатов

    # Создание дирректории для результатов
    results_path = create_results_dir(workdir_path)
    my_time('Rewrite RES Dir')  # Время выполнения программы
    # %%% создание графиков

    plot_diagrams(all_dv_data,
                  all_fg_data,
                  all_mg_data,
                  res_dv,
                  res_gear,
                  res_ovlds,
                  results_path)

    my_time('Plot Diagrams')  # Время выполнения программы
    # raise SystemExit#!!!
    # %%% вывод в excel
    # f_name = 'RES__/Dl_results.xlsx'
    #
    # wr = pd.ExcelWriter(f_name, engine='xlsxwriter')
    #
    # dlf.to_excel_list(wr, res_gear['FG'], 'FG')  # Нагрузки на ПОШ
    # dlf.to_excel_list(wr, res_gear['MG_max'], 'MG_max')  # Нагрузки на ООШ
    # dlf.to_excel_list(wr, res_gear['MG_comb'], 'MG_comb')  # Комбинации ООШ
    # dlf.to_excel_list(
    #     wr,
    #     res_gear['MG_cross_comb'],
    #     'MG_crosscomb')  # Совместные комбинации ООШ
    #
    # dlf.to_excel_list(wr, [res_dv[key] for key in [*res_dv]],
    #                   'n_dv')  # Перегрузки по двигателю
    #
    # dlf.to_excel_list(
    #     wr, [
    #         res_ovlds[key] for key in [
    #             *res_ovlds] if 'fuz' in key],
    #     'n_fuz')  # Перегрузки по фюзеляжу
    # dlf.to_excel_list(
    #     wr, [
    #         res_ovlds[key] for key in [
    #             *res_ovlds] if 'go' in key], 'n_go')  # Перегрузки по ГО
    # dlf.to_excel_list(
    #     wr, [
    #         res_ovlds[key] for key in [
    #             *res_ovlds] if 'kr' in key], 'n_kr')  # Перегрузки по крылу
    #
    # wr.save()
    #
    # my_time('Write to EXCEL')  # Время выполнения программы

    return my_time('END')  # Время выполнения программы


def plot_diagrams(all_dv_data: [[pd.DataFrame]],
                  all_fg_data: list,
                  all_mg_data: [[pd.DataFrame]],
                  res_dv: object,
                  res_gear: object,
                  res_ovlds: object,
                  results_path: object) -> object:
    # Диаграммы перегрузок двигателей
    dlf.plot_dv_overloads(all_dv_data,
                          res_dv,
                          results_path / 'dv_overloads')
    # Диаграммы для перегрузок
    dlf.plot_overloads(res_ovlds, results_path / 'Overloads', 8)
    # Графики ООШ и ПОШ
    dlf.plot_mg_loads(all_mg_data, res_gear, results_path / 'MG_loads')
    dlf.plot_fg_loads(all_fg_data, res_gear, results_path / 'FG_loads')
    plt.close('all')  # Удаление всех созданных фигур


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

    fin_time = main(params)