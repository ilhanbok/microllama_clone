from os import listdir
from os.path import isfile, join

from io import StringIO

import pandas as pd
import traceback

import sys
import re
import numpy as np
import csv

import matplotlib.pyplot as plt
csv.field_size_limit(sys.maxsize)

def remove_multiline_comments(code):
    # Remove """ docstrings/comments
    code = re.sub(r'""".*?"""', 'pass', code, flags=re.DOTALL)
    # Remove ''' docstrings/comments
    code = re.sub(r"'''.*?'''", 'pass', code, flags=re.DOTALL)
    code =   re.split(r'# example usage', code, flags=re.IGNORECASE)[0]
    return code

def dataframe_to_dict(df, orient='dict'):
    """
    Convert pandas DataFrame to dictionary. Returns original input if not a DataFrame.

    Parameters:
    df: Input to convert (if DataFrame) or return as-is
    orient (str): Determines the type of the values of the dictionary.
                  Options: 'dict', 'list', 'series', 'split', 'tight', 'records', 'index'

    Returns:
    dict or original input: Dictionary representation if DataFrame, otherwise original input
    """
    # Return original input if not a DataFrame
    if not isinstance(df, pd.DataFrame):
        return df

    return df.to_dict(orient=orient)

# Use this function to parse and sanitize TRACK/CENTER raw data strings
def clean_and_cast(string,replacement=[0,1,2,3,4]):
    s = re.sub(r',{2,}',',',string.replace('NA','').strip().rstrip('>>>').strip().rstrip(',').strip())
    if len(s) == 0:
        return replacement
    result_final = []
    try:
        result_final = eval(s)
        if isinstance(result_final,tuple):
            result_final = list(result_final)
        if not isinstance(result_final,list):
            result_final = [result_final]
    except Exception as e:
        result_data = s.split(',')
        for item in result_data:
            try:
                if item != '':
                    result_final.append(eval(item))
            except Exception as e:
                result_final.append(item)
    return result_final

def cast(array):
    result_final = []
    for item in array:
        try:
            result_final.append(eval(item))
        except Exception as e:
            result_final.append(item)
    return result_final

if __name__ == "__main__":
  class DictToClass:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)
  with open('autoeval_output.out','a') as o:
    base_path_TRACK = '/home/xibok/Documents/TRACK EXAMPLE UPLOAD 07162024/DATA/tracktbi csv dump processed/'
    #o.write(f'TRACK_Varname,CENTER_Varname,Output,Status\n')
    base_subfolder_path = sys.argv[1]
    base_vars_and_path = sys.argv[2]
    python_list = [base_subfolder_path]#f for f in sorted(listdir(base_subfolder_path)) if isfile(join(base_subfolder_path, f
    out_list = [base_vars_and_path]#f for f in sorted(listdir(base_vars_and_path)) if isfile(join(base_vars_and_path, f))]
    b = 0
    for file in python_list:
        plt.ioff()
        print(file)
        with open(base_vars_and_path) as bf:
            TRACK_Varname = bf.readline().strip()
            bf.readline()
            TRACK_Tablename = 'noNA_sampled_' + TRACK_Varname.split('.')[0] + '_scrambled.csv.csv'
            TRACK_Fieldname = TRACK_Varname.split('.')[1]
            TRACK_data = []
            # If possible, read from the original TRACK file
            with open(base_path_TRACK + TRACK_Tablename) as trt:
                csv_reader = csv.reader(trt,delimiter=',')
                first = True
                for row in csv_reader:
                    if first:
                        #print(row)
                        print(TRACK_Fieldname)
                        if TRACK_Fieldname in row:
                          index = row.index(TRACK_Fieldname)
                        else:
                          break
                        first = False
                    else:
                        TRACK_data.append(row[index])
            TRACK_data = cast(TRACK_data)

            CENTER_Varname = bf.readline().strip()
            CENTER_data_str = re.sub(r',{2,}',',',bf.readline().strip().rstrip('>>>').strip().rstrip(',').strip())
            CENTER_Fieldname = CENTER_Varname.split('.')[1]
            if len(TRACK_data) == 0 and (len(CENTER_data_str) == 0):
                TRACK_data = [0,1,2,3,4]
                CENTER_data = [0,1,2,3,4]
            elif (len(CENTER_data_str) == 0):
                CENTER_data = TRACK_data
            else:
                CENTER_data = clean_and_cast(CENTER_data_str)
                if (len(TRACK_data) == 0):
                    TRACK_data = CENTER_data

            desired_length = max([len(TRACK_data),len(CENTER_data)])

            print(TRACK_data)
            print(CENTER_data)

            try:
                TRACK_data = np.tile(TRACK_data,20)[:desired_length]
                CENTER_data = np.tile(CENTER_data,20)[:desired_length]
            except ValueError as e:
                tb = traceback.format_exc()
                print(tb)

            print(TRACK_data)
            print(CENTER_data)

        b += 1
        y1 = None
        y2 = None

        dict_data = {TRACK_Fieldname : TRACK_data, CENTER_Fieldname : CENTER_data}
        dict_data2 = {TRACK_Fieldname : TRACK_data, CENTER_Fieldname : CENTER_data}
        class_data = DictToClass(dict_data)
        class_data2 = DictToClass(dict_data2)

        arr_data = []
        for val in TRACK_data:
            arr_data.append({TRACK_Fieldname : val})

        arr_data2 = []
        for val in CENTER_data:
            arr_data2.append({CENTER_Fieldname : val})

        str_e = open(base_subfolder_path).read()
        l = []
        ls_prev = locals().copy()

        #exec(str_e)
        #exec('\n\n'.join(open(base_subfolder_path).read().replace('from __future__ import print_function','').split('\n\n')[0:-1]))
        #
        # ls = locals().copy()
        #
        # # Find new or changed variables (excluding functions and built-ins)
        # for key, value in ls.items():
        #     if key not in ls_prev or ls_prev[key] != value:
        #         if not callable(value) and not key.startswith("__"):
        #             l.append((key, value))
        # print(l)
        # print(str_e)
        # y1 = eval(l[-1][0]) if l and l[-1][0] != 'ls_prev' else None
        # print(y1)
        # print('done with y1')
        if y1 is None:
            str_e = remove_multiline_comments(open(base_subfolder_path).read())
            if '"""' in str_e:
                str_e = '\ndef '.join(str_e.split('\ndef ')[0:-1])
            str_f = str_e
            if str_e.count('\ndef ') > 1:
                str_e = '\ndef '.join(str_e.split('\ndef ')[0:2])
                str_f = '\ndef '+str_e.split('\ndef ')[-1]
            str_e = re.sub(r'\ndef [^(]*\(','\ndef g(',str_e)
            try:
                exec(str_f)
            except Exception as e:
                tb = traceback.format_exc()
                print(tb)
            try:
                exec(str_e)
            except Exception as e:
                tb = traceback.format_exc()
                print(tb)
            print(str_e)
            if y1 is None:
                try:
                    y1 = g()
                except Exception as e:
                    tb = traceback.format_exc()
                    print(tb)
            if y1 is None:
                try:
                    y1,y2 = g(TRACK_data,CENTER_data)
                except Exception as e:
                    tb = traceback.format_exc()
                    print(tb)
            if y1 is None:
                try:
                    y1 = g(TRACK_data[0])
                    y2 = g(CENTER_data[0])
                except Exception as e:
                    tb = traceback.format_exc()
                    print(tb)
            if y1 is None:
                try:
                    y1 = g(TRACK_data)
                    y2 = g(CENTER_data)
                except Exception as e:
                    tb = traceback.format_exc()
                    print(tb)
            if y1 is None:
                try:
                    y1 = g(TRACK_data.astype(str))
                    y2 = g(CENTER_data.astype(str))
                except Exception as e:
                    tb = traceback.format_exc()
                    print(tb)
            if y1 is None:
                try:
                    y1 = g([TRACK_data,TRACK_data])
                    y2 = g([CENTER_data,CENTER_data])
                except Exception as e:
                    tb = traceback.format_exc()
                    print(tb)
            for _ in range(5):
                if y1 is None:
                    print(_)
                    print('\n\n\n')
                    try:
                        y1 = g(dict_data)
                        y2 = g(dict_data2)
                    except KeyError as e:
                        the_key = str(e).replace('KeyError: ','')[1:-1]
                        dict_data[the_key] = dict_data[TRACK_Fieldname]
                        dict_data2[the_key] = dict_data2[CENTER_Fieldname]
                        try:
                            the_key_int = int(the_key)
                            dict_data[the_key_int] = dict_data[TRACK_Fieldname]
                            dict_data2[the_key_int] = dict_data2[CENTER_Fieldname]
                        except Exception as e:
                            pass
                        for i in range(len(arr_data)):
                            arr_data[i][the_key] = arr_data[i][TRACK_Fieldname]
                        for i in range(len(arr_data2)):
                            arr_data2[i][the_key] = arr_data2[i][CENTER_Fieldname]
                        tb = traceback.format_exc()
                        print(tb)
                        print(dict_data)
                        print(dict_data2)
                    except Exception as e:
                        tb = traceback.format_exc()
                        print(tb)
            for _ in range(5):
                if y1 is None:
                    print(_)
                    print('\n\n\n')
                    try:
                        y1 = g(arr_data)
                        y2 = g(arr_data2)
                    except KeyError as e:
                        the_key = str(e).replace('KeyError: ','')[1:-1]
                        dict_data[the_key] = dict_data[TRACK_Fieldname]
                        dict_data2[the_key] = dict_data2[CENTER_Fieldname]
                        try:
                            the_key_int = int(the_key)
                            dict_data[the_key_int] = dict_data[TRACK_Fieldname]
                            dict_data2[the_key_int] = dict_data2[CENTER_Fieldname]
                        except Exception as e:
                            pass
                        for i in range(len(arr_data)):
                            arr_data[i][the_key] = arr_data[i][TRACK_Fieldname]
                        for i in range(len(arr_data2)):
                            arr_data2[i][the_key] = arr_data2[i][CENTER_Fieldname]
                        tb = traceback.format_exc()
                        print(tb)
                        print(dict_data)
                        print(dict_data2)
                    except Exception as e:
                        tb = traceback.format_exc()
                        print(tb)
            for _ in range(5):
                if y1 is None:
                    print(_)
                    print('\n\n\n')
                    try:
                        y1 = g(arr_data,arr_data2)
                    except KeyError as e:
                        the_key = str(e).replace('KeyError: ','')[1:-1]
                        dict_data[the_key] = dict_data[TRACK_Fieldname]
                        dict_data2[the_key] = dict_data2[CENTER_Fieldname]
                        try:
                            the_key_int = int(the_key)
                            dict_data[the_key_int] = dict_data[TRACK_Fieldname]
                            dict_data2[the_key_int] = dict_data2[CENTER_Fieldname]
                        except Exception as e:
                            pass
                        for i in range(len(arr_data)):
                            arr_data[i][the_key] = arr_data[i][TRACK_Fieldname]
                        for i in range(len(arr_data2)):
                            arr_data2[i][the_key] = arr_data2[i][CENTER_Fieldname]
                        tb = traceback.format_exc()
                        print(tb)
                        print(dict_data)
                        print(dict_data2)
                    except Exception as e:
                        tb = traceback.format_exc()
                        print(tb)
            for _ in range(5):
                if y1 is None:
                    print(_)
                    print('\n\n\n')
                    try:
                        y1 = g(dict_data,dict_data2)
                    except KeyError as e:
                        the_key = str(e).replace('KeyError: ','')[1:-1]
                        dict_data[the_key] = dict_data[TRACK_Fieldname]
                        dict_data2[the_key] = dict_data2[CENTER_Fieldname]
                        try:
                            the_key_int = int(the_key)
                            dict_data[the_key_int] = dict_data[TRACK_Fieldname]
                            dict_data2[the_key_int] = dict_data2[CENTER_Fieldname]
                        except Exception as e:
                            pass
                        for i in range(len(arr_data)):
                            arr_data[i][the_key] = arr_data[i][TRACK_Fieldname]
                        for i in range(len(arr_data2)):
                            arr_data2[i][the_key] = arr_data2[i][CENTER_Fieldname]
                        tb = traceback.format_exc()
                        print(tb)
                        print(dict_data)
                        print(dict_data2)
                    except Exception as e:
                        tb = traceback.format_exc()
                        print(tb)
            for _ in range(5):
                if y1 is None:
                    print(_)
                    print('\n\n\n')
                    try:
                        y1 = g([dict_data,dict_data],[dict_data2,dict_data2])
                    except KeyError as e:
                        the_key = str(e).replace('KeyError: ','')[1:-1]
                        dict_data[the_key] = dict_data[TRACK_Fieldname]
                        dict_data2[the_key] = dict_data2[CENTER_Fieldname]
                        try:
                            the_key_int = int(the_key)
                            dict_data[the_key_int] = dict_data[TRACK_Fieldname]
                            dict_data2[the_key_int] = dict_data2[CENTER_Fieldname]
                        except Exception as e:
                            pass
                        for i in range(len(arr_data)):
                            arr_data[i][the_key] = arr_data[i][TRACK_Fieldname]
                        for i in range(len(arr_data2)):
                            arr_data2[i][the_key] = arr_data2[i][CENTER_Fieldname]
                        tb = traceback.format_exc()
                        print(tb)
                        print(dict_data)
                        print(dict_data2)
                    except Exception as e:
                        tb = traceback.format_exc()
                        print(tb)
            for key in dict_data.keys():
                dict_data[key] = {}
            for key in dict_data2.keys():
                dict_data2[key] = {}
            for _ in range(5):
                if y1 is None:
                    print(_)
                    print('\n\n\n')
                    try:
                        y1 = g(dict_data)
                        y2 = g(dict_data2)
                    except KeyError as e:
                        the_key_2 = str(e).replace('KeyError: ','')[1:-1]
                        for key in dict_data.keys():
                            dict_data[key][the_key_2] = dict_data[TRACK_Fieldname]
                        for key in dict_data2.keys():
                            dict_data2[key][the_key_2] = dict_data2[CENTER_Fieldname]
                        tb = traceback.format_exc()
                        print(tb)
                        print(dict_data)
                        print(dict_data2)
                    except Exception as e:
                        tb = traceback.format_exc()
                        print(tb)
            for _ in range(5):
                if y1 is None:
                    print(_)
                    print('\n\n\n')
                    try:
                        y1 = g([dict_data,dict_data])
                        y2 = g([dict_data2,dict_data2])
                    except KeyError as e:
                        the_key_2 = str(e).replace('KeyError: ','')[1:-1]
                        for key in dict_data.keys():
                            dict_data[key][the_key_2] = dict_data[TRACK_Fieldname]
                        for key in dict_data2.keys():
                            dict_data2[key][the_key_2] = dict_data2[CENTER_Fieldname]
                        tb = traceback.format_exc()
                        print(tb)
                        print(dict_data)
                        print(dict_data2)
                    except Exception as e:
                        tb = traceback.format_exc()
                        print(tb)
            for _ in range(5):
                if y1 is None:
                    print(_)
                    print('\n\n\n')
                    try:
                        y1 = g(dict_data,dict_data2)
                    except KeyError as e:
                        the_key_2 = str(e).replace('KeyError: ','')[1:-1]
                        for key in dict_data.keys():
                            dict_data[key][the_key_2] = dict_data[TRACK_Fieldname]
                        for key in dict_data2.keys():
                            dict_data2[key][the_key_2] = dict_data2[CENTER_Fieldname]
                        tb = traceback.format_exc()
                        print(tb)
                        print(dict_data)
                        print(dict_data2)
                    except Exception as e:
                        tb = traceback.format_exc()
                        print(tb)
            for _ in range(5):
                if y1 is None:
                    print(_)
                    print('\n\n\n')
                    try:
                        y1 = g([dict_data,dict_data],[dict_data2,dict_data2])
                    except KeyError as e:
                        the_key_2 = str(e).replace('KeyError: ','')[1:-1]
                        for key in dict_data.keys():
                            dict_data[key][the_key_2] = dict_data[TRACK_Fieldname]
                        for key in dict_data2.keys():
                            dict_data2[key][the_key_2] = dict_data2[CENTER_Fieldname]
                        tb = traceback.format_exc()
                        print(tb)
                        print(dict_data)
                        print(dict_data2)
                    except Exception as e:
                        tb = traceback.format_exc()
                        print(tb)
            if y1 is None:
                try:
                    y1 = g(class_data)
                    y2 = g(class_data2)
                except Exception as e:
                    tb = traceback.format_exc()
                    print(tb)
            if y1 is None:
                try:
                    y1 = g(pd.DataFrame(dict_data))
                    y2 = g(pd.DataFrame(dict_data2))
                except Exception as e:
                    tb = traceback.format_exc()
                    print(tb)
                    print(dict_data)
                    print(dict_data2)
            if y1 is None:
                try:
                    y1 = g(TRACK_data,CENTER_data)
                except Exception as e:
                    tb = traceback.format_exc()
                    print(tb)
            if y1 is None:
                try:
                    y1 = g(TRACK_data[0],CENTER_data[0])
                except Exception as e:
                    tb = traceback.format_exc()
                    print(tb)
            if y1 is None:
                try:
                    y1 = g([TRACK_data,TRACK_data],[CENTER_data,CENTER_data])
                except Exception as e:
                    tb = traceback.format_exc()
                    print(tb)
            if y1 is None:
                try:
                    y1 = g(dict_data,dict_data2)
                except KeyError as e:
                    dict_data[str(e).replace('KeyError: ','')[1:-1]] = dict_data[TRACK_Fieldname]
                    dict_data2[str(e).replace('KeyError: ','')[1:-1]] = dict_data2[CENTER_Fieldname]
                    print(dict_data)
                    class_data = DictToClass(dict_data)
                    class_data2 = DictToClass(dict_data2)
                    tb = traceback.format_exc()
                    print(tb)
                except Exception as e:
                    try:
                        y1 = g([dict_data,dict_data],[dict_data2,dict_data2])
                    except KeyError as e:
                        dict_data[str(e).replace('KeyError: ','')[1:-1]] = dict_data[TRACK_Fieldname]
                        dict_data2[str(e).replace('KeyError: ','')[1:-1]] = dict_data2[CENTER_Fieldname]
                        print(dict_data)
                        class_data = DictToClass(dict_data)
                        class_data2 = DictToClass(dict_data2)
                        tb = traceback.format_exc()
                        print(tb)
                        try:
                            y1 = g([dict_data,dict_data],[dict_data2,dict_data2])
                        except Exception as e:
                            tb = traceback.format_exc()
                            print(tb)
                    except Exception as e:
                        tb = traceback.format_exc()
                        print(tb)
            if y1 is None:
                try:
                    y1 = g(dict_data,dict_data2)
                except KeyError as e:
                    dict_data[str(e).replace('KeyError: ','')[1:-1]] = dict_data[TRACK_Fieldname]
                    dict_data2[str(e).replace('KeyError: ','')[1:-1]] = dict_data2[CENTER_Fieldname]
                    print(dict_data)
                    class_data = DictToClass(dict_data)
                    class_data2 = DictToClass(dict_data2)
                    tb = traceback.format_exc()
                    print(tb)
                except Exception as e:
                    tb = traceback.format_exc()
                    print(tb)
            if y1 is None:
                try:
                    y1 = g(dict_data,dict_data2)
                except Exception as e:
                    tb = traceback.format_exc()
                    print(tb)
            if y1 is None:
                try:
                    y1 = g(pd.DataFrame(dict_data),pd.DataFrame(dict_data2))
                except Exception as e:
                    tb = traceback.format_exc()
                    print(tb)
        y1 = str(dataframe_to_dict(y1,'list')).replace('\n',' ')
        y2 = str(dataframe_to_dict(y2,'list')).replace('\n',' ')
        print('y1: {0}, y2: {1}'.format(y1,y2))
        o.write(f'{TRACK_Varname}☺{CENTER_Varname}☺{file}☺{y1}☺{y2}\n')
