import glob
import os
import pandas as pd
import numpy as np
import locale
from datetime import date, timedelta, datetime 
from dateutil.relativedelta import relativedelta
import re
import uuid
pd.options.mode.chained_assignment = None

def delta_months(periodo, delta=-12):
    """
    Suma o resta un número de meses a una fecha en formato 'YYYYMM' y devuelve el resultado
    en el mismo formato.
    
    Args:
        periodo (int): Fecha en formato 'YYYYMM' a la que se le sumarán o restarán los meses.
        delta (int, optional): Número de meses que se sumarán o restarán a la fecha. Por defecto es -12.
    
    Returns:
        str: Fecha resultante en formato 'YYYYMM'.
    """
    
    # Convertir la fecha en formato 'YYYYMM' a un objeto datetime
    periodo_dt = datetime.strptime(str(periodo), '%Y%m')
    
    # Agregar el número de meses especificado por el parámetro delta
    resultado_dt = periodo_dt + relativedelta(months=delta)
    
    # Convertir la fecha resultante en formato 'YYYYMM'
    resultado_str = resultado_dt.strftime('%Y%m')
    
    # Devolver el resultado
    return resultado_str

def download_files(fecha, path_files):
    """
    Descarga los archivos correspondientes a un periodo determinado y los procesa para generar
    un diccionario que contiene los valores en pesos de cada métrica para cada banco.
    
    
    
    Args:
        fecha (str): Periodo que se desea descargar en formato 'YYYYMM'.
        path_files (str): Ruta donde se encuentran los archivos. Debe incluir un backslash y un 
        espacio al final. Ej: r'C:\InfCMF\ '
    
    Returns:
        dict: Diccionario con los bancos como claves y un diccionario con las métricas como claves y los
              valores en pesos como valores para cada banco.
    """
    
    # Formatear la ruta de los archivos y obtener la lista de todos los archivos correspondientes al periodo
    path = os.path.join(path_files, fecha)
    path = path.replace(" ", "", 1)
    all_files = glob.glob(os.path.join(path, "*.txt"))
    
    # Descartar los archivos que contengan las palabras "plan", "b2" o "c2" en su nombre
    stopwords = ['plan', 'b2', 'c2']
    for word in stopwords:
        all_files = [file for file in all_files if word not in file.lower()]
    
    # Crear una lista vacía donde se guardarán los datos de cada archivo
    li = []
    
    # Procesar cada archivo
    for filename in all_files:
        # Leer el nombre del banco desde la primera línea del archivo
        name_bank = pd.read_csv(filename, sep='\t', nrows=1, header=None, decimal=",", encoding='latin-1')
        name_bank = name_bank.iloc[0, 1]
        
        # Leer los datos de las métricas del archivo
        df = pd.read_csv(filename, sep='\t', skiprows=[0], header=None, decimal=",", encoding='latin-1')
        
        # Si el archivo corresponde al tipo "b1", sumar las columnas 2 a 4 para obtener el total de la cuenta
        if ("b1" in filename):
            df[1] = df[1] + df[2] + df[3] + df[4]
        if ("r1" in filename and int(fecha)<202201):
            df[1] = df[1] + df[2] + df[3] + df[4]
        
        # Seleccionar solo las columnas con los códigos de las métricas y los valores en pesos
        df = df.iloc[:, :2]
        
        # Agregar columnas con el código del banco y el nombre del banco
        df['cod_bank'] = ((filename.split('/')[-1]).split('.')[0])[-3:]
        df['Institucion'] = name_bank
        
        # Agregar el dataframe a la lista de dataframes
        li.append(df)
        
    # Unir todos los dataframes en uno solo
    frame = pd.concat(li, axis=0, ignore_index=True)
    
    # Renombrar las columnas del dataframe resultante
    frame = frame.rename(columns={0: "codigo_cuenta", 1: "moneda_total"})
    
    # Calcular la suma de los valores en pesos para cada métrica y banco
    reporte = pd.pivot_table(frame, values='moneda_total', index=['codigo_cuenta'],
                    columns=['Institucion'], aggfunc=np.sum)
    if (fecha[:4]<"2022"):
        for col in (reporte.columns):
            reporte[col]=reporte[col]*1000000
    # Convertir el dataframe resultante en un diccionario
    reporte_dict = reporte.to_dict()
    
    # Devolver el dic
    return reporte_dict

def calculate_metric(df,dict_report):
    """
     La función calculate_metric se encarga de calcular una serie de métricas a 
     partir de un diccionario de datos df y otro diccionario que especifica las
     operaciones a realizar en cada una de las métricas (dict_report).
     
     Argumentos:

        df: diccionario con los códigos de cuenta y sus respectivos valores.
        
        dict_report: diccionario que especifica las operaciones a realizar en cada
        una de las métricas. Las claves corresponden a los nombres de las métricas,
        mientras que los valores corresponden a las operaciones a realizar. Las
        operaciones se especifican como strings que deben ser evaluados para obtener 
        el valor de la métrica. Las variables a utilizar en las operaciones son las 
        llaves del diccionario df, por lo que el formato debe ser df[nombre_codigo].
    
    Retorno:
        bank_list: lista de valores que corresponden a las métricas calculadas.
    """     
    bank_list = []
    for metric, equation in dict_report.items():
        try:
            metric_value =eval(equation)
        except ZeroDivisionError:
            metric_value=0
        bank_list.append(metric_value)
    return bank_list


def get_report(dict_report, data_report_dict):
    """
    Obtiene el reporte para todos los bancos para 1 periodo.

    Args:
        dict_report (dict): Diccionario con los nombres de las métricas (str) como claves y las operaciones de los códigos de cuenta como valores.
                            Ejemplo: {'cartera_consumo_cuotas': "df[1305100]", 
                                      'cartera_consumo': "df[148000100]-df[148000100]", 
                                      'cartera': "df[148000100]/df[148000100]"}
        data_report_dict (dict): Diccionario con los códigos de cuenta (str) como claves y sus respectivos valores como valores.

    Returns:
        pandas.DataFrame: DataFrame con las métricas solicitadas en el dict_report para todos los bancos. Las columnas del DataFrame
        corresponden a los bancos y las filas a las métricas.

    """
    df_final = pd.DataFrame()
    df_final['metrica'] = pd.Series(list(dict_report.keys()))
    for bank, cod_cuenta in data_report_dict.items():
        a = (calculate_metric(cod_cuenta, dict_report))    
        df_final[bank] = pd.Series(a)
    return df_final


def deltas_new_inf_ms(df1,column_values_name):
    grupos_por_banco = df1.groupby('banco')
    dfinal=pd.DataFrame()
    for banco, grupo in grupos_por_banco:
            test=grupos_por_banco.get_group(banco)

            df=test
            values = list(df[column_values_name])
            periods = list(df['fecha'])
            metric_name = list(df['metrica'])
            bancos = list(df['banco'])
            metric_zip = zip(values, periods, metric_name)
            metrica_actual = metric_name[0]
            banco_actual = bancos[0]
            delta_aa=[]
            delta_ma=[]
            moneda_total_1_month_ago=[]
            moneda_total_12_months_ago=[]
            moneda_total_24_months_ago=[]
            moneda_total_36_months_ago=[]
            metric_dict={}

            for value,period,metric,banco in zip(values,periods,metric_name,bancos):
                if metrica_actual != metric or banco_actual != banco:

                    delta_aa.extend(calculate_delta(metric_dict,q_months=12))
                    delta_ma.extend(calculate_delta(metric_dict,q_months=1))
                    moneda_total_1_month_ago.extend(get_previous_value(metric_dict,q_months=1))
                    moneda_total_12_months_ago.extend(get_previous_value(metric_dict,q_months=12))
                    moneda_total_24_months_ago.extend(get_previous_value(metric_dict,q_months=24))
                    moneda_total_36_months_ago.extend(get_previous_value(metric_dict,q_months=36))
                    metric_dict = {}

                metric_dict[str(period)]=value
                metrica_actual = metric
                banco_actual = banco

            delta_aa.extend(calculate_delta(metric_dict,q_months=12))
            delta_ma.extend(calculate_delta(metric_dict,q_months=1))
            moneda_total_1_month_ago.extend(get_previous_value(metric_dict,q_months=1))
            moneda_total_12_months_ago.extend(get_previous_value(metric_dict,q_months=12))
            moneda_total_24_months_ago.extend(get_previous_value(metric_dict,q_months=24))
            moneda_total_36_months_ago.extend(get_previous_value(metric_dict,q_months=36))

            df['ms_delta_aa'.format('name_table')]=delta_aa
            df['ms_delta_ma'.format('name_table')]=delta_ma
            df['ms_moneda_total_1_month_ago']=moneda_total_1_month_ago
            df['ms_moneda_total_12_months_ago']=moneda_total_12_months_ago
            df['ms_moneda_total_24_months_ago']=moneda_total_24_months_ago
            df['ms_moneda_total_36_months_ago']=moneda_total_36_months_ago


            result=pd.concat([df,dfinal])
            dfinal=result


    return dfinal


def deltas_new_inf(df1,column_values_name):

    grupos_por_banco = df1.groupby('banco')
    dfinal=pd.DataFrame()
    for banco, grupo in grupos_por_banco:
            test=grupos_por_banco.get_group(banco)

            df=test
            values = list(df[column_values_name])
            periods = list(df['fecha'])
            metric_name = list(df['metrica'])
            bancos = list(df['banco'])
            metric_zip = zip(values, periods, metric_name)

            metrica_actual = metric_name[0]
            banco_actual = bancos[0]
            delta_aa=[]
            delta_ma=[]
            moneda_total_1_month_ago=[]
            moneda_total_12_months_ago=[]
            moneda_total_24_months_ago=[]
            moneda_total_36_months_ago=[]
            metric_dict={}

            for value,period,metric,banco in zip(values,periods,metric_name,bancos):
                if metrica_actual != metric or banco_actual != banco:

                    delta_aa.extend(calculate_delta(metric_dict,q_months=12))
                    delta_ma.extend(calculate_delta(metric_dict,q_months=1))
                    moneda_total_1_month_ago.extend(get_previous_value(metric_dict,q_months=1))
                    moneda_total_12_months_ago.extend(get_previous_value(metric_dict,q_months=12))
                    moneda_total_24_months_ago.extend(get_previous_value(metric_dict,q_months=24))
                    moneda_total_36_months_ago.extend(get_previous_value(metric_dict,q_months=36))
                    metric_dict = {}

                metric_dict[str(period)]=value
                metrica_actual = metric
                banco_actual = banco

            delta_aa.extend(calculate_delta(metric_dict,q_months=12))
            delta_ma.extend(calculate_delta(metric_dict,q_months=1))
            moneda_total_1_month_ago.extend(get_previous_value(metric_dict,q_months=1))
            moneda_total_12_months_ago.extend(get_previous_value(metric_dict,q_months=12))
            moneda_total_24_months_ago.extend(get_previous_value(metric_dict,q_months=24))
            moneda_total_36_months_ago.extend(get_previous_value(metric_dict,q_months=36))

            df['delta_aa'.format('name_table')]=delta_aa
            df['delta_ma'.format('name_table')]=delta_ma
            df['moneda_total_1_month_ago']=moneda_total_1_month_ago
            df['moneda_total_12_months_ago']=moneda_total_12_months_ago
            df['moneda_total_24_months_ago']=moneda_total_24_months_ago
            df['moneda_total_36_months_ago']=moneda_total_36_months_ago


            result=pd.concat([df,dfinal])
            dfinal=result


    return dfinal



def get_report_year(dict_report, year, info_directory):
    """
    Se obtiene el reporte para el año entregado.

    Args:
        dict_report (dict): Diccionario con los nombres de las métricas (str) como claves y las operaciones de los códigos de cuenta como valores.
                            Ejemplo: {'cartera_consumo_cuotas': "df[1305100]", 
                                      'cartera_consumo': "df[148000100]-df[148000100]", 
                                      'cartera': "df[148000100]/df[148000100]"}
        year (str): Año para el cual se quiere realizar el reporte en formato 'yyyy01'.
        info_directory (str): Ruta donde se encuentra la información para los reportes.

    Returns:
        tuple of two pandas.DataFrame:
        - El primer DataFrame es el reporte sin pivotear, con los bancos como columnas y las métricas como filas.
        - El segundo DataFrame es el reporte pivoteado, con las métricas como columnas y las fechas como filas.
    """
    final_report = pd.DataFrame()
    final_report_with_pivot = pd.DataFrame()
    for i in range (0,12):
        dateaux = delta_months(year, i)
        try:
            data_report_dict = download_files(dateaux, info_directory)
            report_finish = get_report(dict_report, data_report_dict)
            report_finish2 = report_finish
            report_finish2['fecha'] = dateaux

            report_finish = pd.pivot_table(report_finish, columns=['metrica'])
            report_finish['fecha'] = dateaux

            final_report = pd.concat([final_report, report_finish], axis=0)
            final_report_with_pivot = pd.concat([final_report_with_pivot, report_finish2], axis=0)
        except ValueError:
            fecha_fin=delta_months(dateaux,-1)
            print(f"Archivo creado hasta: {fecha_fin}".format(fecha_fin))
            print(f"No existen datos para {dateaux} y el resto del año".format(dateaux))
            return final_report, final_report_with_pivot
    print(f"Archivo de {dateaux} creado con éxito".format(dateaux))
    return final_report, final_report_with_pivot



def get_report_years(info_directory, years_list, dict_report_v_2021=None, dict_report_v_2022=None):
    """
    Crea el reporte para la lista de años que se entregue.

    Args:
        info_directory (str): Path donde se encuentran los archivos.
            El formato del path_files es una barra invertida al final seguida de un espacio en blanco:
            ej. r'C:\InfCMF\ ' con "\ ".
        years_list (list): Lista de años que se quieren obtener.
            El formato es ['2021', '2022',...].
        dict_report_v_2021, dict_report_v_2022 (dict): Diccionarios con las métricas a calcular por año.
            key -> Nombres de la métrica (str)
            values -> Operaciones de los códigos (str), ej. df['codigo'] + df['codigo2'].

    Returns:
        2 Dataframe:
        - Uno sin pivotear, el cual tiene los bancos como columnas.
        - Uno pivoteado, el cual tiene las métricas como columnas.
    """
    final_report = pd.DataFrame()
    final_report_no_pivot = pd.DataFrame()
    for year in years_list:
        if year < "2022":
            dict_report = dict_report_v_2021
        else:
            dict_report = dict_report_v_2022
            
        year = year + '01'
        dataset, dataset_without_pivot = get_report_year(dict_report, year, info_directory)
        final_report = pd.concat([final_report, dataset], axis=0)
        final_report_no_pivot = pd.concat([final_report_no_pivot, dataset_without_pivot], axis=0)
        
    return final_report, final_report_no_pivot



def calculate_delta(metric_dict, q_months):
    """
    Calcula la variación porcentual de una métrica a lo largo de q_months meses.
    
    Args:
    - metric_dict (dict): Diccionario con la métrica y sus valores para cada periodo.
    - q_months (int): Número de meses que se quieren comparar.
    
    Returns:
    - delta_array (list): Lista con la variación porcentual de la métrica a lo largo de q_months meses.
                          Si no se puede calcular la variación para un periodo, se devuelve None.
    """
    delta_array = []
    for period in metric_dict:
        prev_period = delta_months(period, delta=-q_months)
        try:
            delta_value = (metric_dict[period] / metric_dict[prev_period]) - 1
        except (KeyError, ZeroDivisionError):
            delta_value = None
        delta_array.append(delta_value)
    return delta_array



def get_previous_value(metric_dict,q_months):

    """
    Función que retorna los valores previos de una métrica para los periodos
    presentes en el diccionario dado.
    
    Args:
        metric_dict (dict): diccionario con los valores de una métrica para cada periodo.
        q_months (int): número de meses hacia atrás que se desea obtener el valor previo.
    
    Returns:
        list: una lista con los valores previos de la métrica para cada periodo en el diccionario.
    """
    _array = []
    for period in metric_dict:
        # Se calcula el periodo previo
        prev_period = delta_months(period,delta=-q_months)
        try:
            # Se busca el valor previo de la métrica
            p_value = metric_dict[prev_period]
        except KeyError:
            # Si no se encuentra un valor previo para ese periodo, se pone como None
            p_value = None
        _array.append(p_value)
    return _array




def add_deltas_report_by_columns(df, fecha1, fecha2, nombre):
    """
    Añade una columna al dataframe con la diferencia porcentual entre dos columnas de fechas específicas.
    
    Args:
        df (pandas.DataFrame): El dataframe que se modificará.
        fecha1 (str): El nombre de la primera columna de fecha.
        fecha2 (str): El nombre de la segunda columna de fecha.
        nombre (str): El nombre de la nueva columna que contendrá las diferencias porcentuales.
        
    Returns:
        pandas.DataFrame: El dataframe original con la columna agregada.
    """
    subset = df.loc[:, (fecha1, fecha2)]
    
    subset[fecha1] = subset[fecha1]
    subset[fecha2] = subset[fecha2]
    
    

    numerador = subset[fecha1]
    denominador = subset[fecha2]
    
    df[nombre] = ((numerador / denominador) - 1) * 100
    df[nombre] = df[nombre].round(1)

    return df



def get_percentages(dataset_without_pivot):
    """
    Toma un conjunto de datos y calcula los porcentajes para cada columna en el conjunto de datos, excepto para las
    columnas "metrica", "fecha" y "SISTEMA FINANCIERO". Los porcentajes se calculan dividiendo cada valor en las columnas
    seleccionadas por el valor en la columna "SISTEMA FINANCIERO" y multiplicando por 100.
    
    Args:
        dataset_without_pivot (pd.DataFrame): Conjunto de datos en el que se van a calcular los porcentajes.
        
    Returns:
        pd.DataFrame: Conjunto de datos con los porcentajes calculados.
    """
    for col in dataset_without_pivot.columns:
        if (col!= 'metrica' and col!='fecha' and col!='SISTEMA FINANCIERO'):
            dataset_without_pivot[col]=(dataset_without_pivot[col]/dataset_without_pivot['SISTEMA FINANCIERO'])*100

    return dataset_without_pivot

def get_pivot_percentages(metrica,data):
    """
    Toma una métrica específica y un conjunto de datos como entrada y devuelve una tabla dinámica de los datos filtrados por
    la métrica, agrupados por fecha y mostrando los valores de las otras columnas para cada fecha. Los valores se muestran en
    formato de punto flotante con 3 decimales.
    
    Args:
        metrica (str): Métrica a filtrar en los datos.
        data (pd.DataFrame): Conjunto de datos que contiene los datos a partir de los cuales se va a crear la tabla dinámica.
        
    Returns:
        pd.DataFrame: Tabla dinámica de los datos filtrados por la métrica, agrupados por fecha y mostrando los valores de
        las otras columnas para cada fecha.
    """
    filter=data['metrica']==metrica
    dataset = data[filter]
    dataset=pd.pivot_table(dataset,columns=['fecha'])
    pd.set_option('display.float_format', '{:.3f}'.format)
    return dataset


def crear_carpeta(nombre_carpeta):
    # Verificar si la carpeta ya existe
    if not os.path.exists(nombre_carpeta):
        # Crear la carpeta en el directorio actual
        os.mkdir(nombre_carpeta)
        print(f"La carpeta '{nombre_carpeta}' se ha creado con éxito.")
    else:
        print(f"La carpeta '{nombre_carpeta}' ya existe.")
        
        
        
        
def get_reporte_n_1(data, dict_report,folder_name):
    """
    Función que genera reportes en formato CSV a partir de un dataframe.

    Args:
    data (pandas.DataFrame): Dataframe con los datos a reportar.
    dict_report (dict): Diccionario con los nombres de las columnas a reportar como llaves y los nombres de los archivos CSV como valores.

    Returns:
    None
    """
    
    crear_carpeta(folder_name)
    
    data.index.name = 'banco'
    data = data.reset_index()

    for key, value in dict_report.items():
        subconjunto = data.loc[:, ["banco", key, "fecha"]]
        subconjunto = subconjunto.pivot(index='banco', columns='fecha', values=key)
        subconjunto = subconjunto.drop(subconjunto.index[(subconjunto != "0").any(axis=1)==False], axis=0)

        last_date = subconjunto.columns[-1]

        delta_names = ["DELTA MA", "DELTA AA", "DELTA AA2", "DELTA AA3"]
        deltas = [36, 24, 12, 1]
        
        # Eliminar las filas con valores faltantes en las columnas "last_date" y "delta_months(last_date,-2)"
        subconjunto[[last_date, delta_months(last_date,-2)]] = subconjunto[[last_date, delta_months(last_date,-2)]].replace(0, np.nan)
        subconjunto = subconjunto.dropna(subset=[last_date, delta_months(last_date,-2)])

        # Agregar las columnas de diferencias
        for i in deltas:
            
            try:      
                delta = delta_months(last_date, -i)
                subconjunto = add_deltas_report_by_columns(subconjunto, last_date, delta, delta_names.pop())
            except KeyError:
                continue
                
                
        num_cols = subconjunto.filter(regex='^.*DELTA.*$').columns
        subconjunto[num_cols] = subconjunto[num_cols].applymap(lambda x: (x / 100))
        
        subconjunto.to_csv(f'{folder_name}/{key}.csv'.format(folder_name,key), index=True)
        
    print("Done")

    
    
def get_marketshare_n_1(data_nopivot, dict_report,folder_name):
    '''
    Función que genera los reportes de Market Share de cada uno de los bancos en el 
    sistema financiero en formato csv.

    Parámetros:
    - data_nopivot: DataFrame con los datos sin pivoteo.
    - dict_report: Diccionario con los nombres de los reportes y las columnas que 
                    deben ser utilizadas para cada uno.

    Retorna:
    - None
    '''
    crear_carpeta(folder_name)
    
    data = get_percentages(data_nopivot)

    # Para cada reporte en el diccionario
    for key, value in dict_report.items():
        
        # Obtenemos la tabla pivot con los porcentajes correspondientes
        table = get_pivot_percentages(key, data)
        
        # Obtenemos la última fecha
        last_date = table.columns[-1]

        delta_names = ["DELTA MA", "DELTA AA", "DELTA AA2", "DELTA AA3"]
        deltas = [36, 24, 12, 1]
        
        # Llenamos los valores nulos con 0
        table = table.fillna(0)
        
        # Eliminamos las filas con valores 0 en la columna de porcentaje
        table = table.loc[table[table.columns[2]] != 0]

        # Agregamos las columnas delta
        for i in deltas:
            try:            
                delta = delta_months(last_date, -i)
                table[delta_names.pop()] = ((table[last_date] - table[delta]) * 100).round(0)
            except KeyError:
                delta_names.pop()
                continue
        
        # Creamos la función que elimina los ".000" de los valores
        elimina_ceros = lambda x: str(x).replace('.000', '')
        
        
        num_cols = table.filter(regex='^(?!.*20).*$').columns
        table[num_cols] = table[num_cols].applymap(elimina_ceros)

        table.index.name = 'banco'
        table = table.reset_index()
            
        table = table.drop(table[table['banco'] == 'SISTEMA FINANCIERO'].index)
        # Guardamos la tabla como un archivo CSV
        
        num_cols = table.filter(like='20').columns
        table[num_cols] = table[num_cols].round(3)

        table[num_cols] = table[num_cols].apply(lambda x: round(x / 100, 4)) 
        table.to_csv(f'{folder_name}/{key}_BPP.csv'.format(folder_name,key), index=True)
        
    print("Done")

    
    
def create_table(data,dict_report):

    data.index.name = 'banco'
    
    data = data.reset_index()
    df1=pd.DataFrame()
    
    for key, value in dict_report.items():
        subconjunto = data.loc[:, ["banco", key, "fecha"]]
        subconjunto['metrica']=key
        subconjunto = subconjunto.rename(columns={key: 'moneda_total'})
        result=pd.concat([subconjunto,df1])
        df1=result
    
    
    return deltas_new_inf(df1,'moneda_total')






def create_ms_table(data,dict_report):
    data.index.name = 'banco'
    data = data.reset_index()

    dataset=data
    df1=pd.DataFrame()

    for key, value in dict_report.items():
        subconjunto = dataset.loc[:, ["banco", key, "fecha"]]
        subconjunto['metrica']=key
        subconjunto = subconjunto.rename(columns={key: 'moneda_total'})

        grupos_por_banco = subconjunto.groupby('fecha')
        df_percentages=pd.DataFrame()
        for banco, grupo in grupos_por_banco:
            test=grupos_por_banco.get_group(banco)
            sistema_financiero_total = test.loc[test["banco"] == "SISTEMA FINANCIERO", "moneda_total"].sum()
            try :
                test["moneda_total"] = test["moneda_total"] / sistema_financiero_total

            except KeyError:
                test["moneda_total"]=0
            df_percentages=pd.concat([test,df_percentages])
        subconjunto=df_percentages
        result=pd.concat([subconjunto,df1])
        df1=result


    
    dfinal = deltas_new_inf_ms(df1,'moneda_total')
    dfinal = dfinal.rename(columns={'moneda_total': 'ms_moneda_total'})
    return dfinal






def create_database(info_directory,dict_report_v_2022,dict_report_v_2021):
    _info_directory= info_directory
    _info_directory = _info_directory.replace("\ ", "")
    files = os.listdir(_info_directory)
    #filtrar por archivos solo con formato YYYYMM
    pattern = r'^\d{6}$'
    files.sort()
    # Obtener el nombre del último archivo
    matching_files = [file for file in os.listdir(_info_directory) if re.search(pattern, file)]
    matching_files.sort()

    ##adding last 2 periods if they are diferent else just the last one
    latest_file = matching_files[-1]

    ultima_fecha_downloaded_files=latest_file
    
    years_list=[]
    
    value=2016
    if(dict_report_v_2021==None):
        value=2022
    ##    
    for i in range(value,(int(ultima_fecha_downloaded_files[:4])+1)):
        years_list.append(str(i))
        
    data,data_nopivot=get_report_years(info_directory,years_list,dict_report_v_2021,
                                   dict_report_v_2022)

    TABLA1=create_table(data,dict_report_v_2022)
    TABLA1MS=create_ms_table(data,dict_report_v_2022)
    tablemaster= pd.merge(TABLA1, TABLA1MS, on=['fecha','banco','metrica'], how='outer')
    
    return tablemaster


def create_all_data(info_directory,dict_report_v_2022_ ,dict_report_v_2021_ ,Dict_Reporte_Carteras2022_ ,Dict_Reporte_Carteras2021_, file_name ):
        Tabla1=create_database(info_directory,dict_report_v_2022_,dict_report_v_2021_)
        Tabla2=create_database(info_directory,Dict_Reporte_Carteras2022_,Dict_Reporte_Carteras2021_)

        df =pd.concat([Tabla1,Tabla2]) 
        df = df.drop_duplicates(['banco','metrica','fecha','moneda_total','ms_moneda_total'])

        df = df.rename_axis(None, axis=1)
        df['fecha_date']=pd.to_datetime(df['fecha'].astype(str), format='%Y%m')


        df['banco'] = df['banco'].astype(str)
        df['metrica'] = df['metrica'].astype(str)
        df['fecha'] = df['fecha'].astype('int64')
        df['delta_aa'] = df['delta_aa'].astype('float64')
        df['delta_ma'] = df['delta_ma'].astype('float64')
        df['ms_delta_aa'] = df['ms_delta_aa'].astype('float64')
        df['ms_delta_ma'] = df['ms_delta_ma'].astype('float64')
        df['moneda_total_1_month_ago'] = df['moneda_total_1_month_ago'].astype("float64")
        df['moneda_total_12_months_ago'] = df['moneda_total_12_months_ago'].astype('float64')
        df['moneda_total_24_months_ago'] = df['moneda_total_24_months_ago'].astype('float64')
        df['moneda_total_36_months_ago'] = df['moneda_total_36_months_ago'].astype('float64')
        df['ms_moneda_total_1_month_ago'] = df['ms_moneda_total_1_month_ago'].astype('float64')
        df['ms_moneda_total_12_months_ago'] = df['ms_moneda_total_12_months_ago'].astype('float64')
        df['ms_moneda_total_24_months_ago'] = df['ms_moneda_total_24_months_ago'].astype('float64')
        df['ms_moneda_total_36_months_ago'] = df['ms_moneda_total_36_months_ago'].astype('float64')
        #df2['fecha_date'] = df2['fecha_date'].astype('datetime64')
        #df2['banco'] = df2['banco'].astype(str)
        
        df=df.fillna(0)

        df.to_csv(file_name, index=False ,sep=";")
        
        return df


def get_historical_table(info_directory,output_file):
    dict_report_v_2022={
            'ctra_consumo_cuotas':"df[148000100]",
             'ctra_consumo_tarjetas':"df[148000300]",
             'ctra_consumo_total':"df[148000100]+df[148000200]+df[148000300]+df[148000400]+df[148000900]",
             'ctra_consumo_ndet':"df[148000100]+df[148000200]+df[148000300]+df[148000400]+df[148000900]-df[811400000]",
             'ctra_consumo_det':"df[811400000]",
             'ctra_deudores_en_cc':"df[148000200]"}
                    

    dict_report_v_2021={'ctra_consumo_cuotas':"df[1305100]",
             'ctra_consumo_tarjetas':"df[1305400]",
             'ctra_consumo_total':"df[1305100]+df[1305300]+df[1305400]+df[1305800]+df[1305900]",
             'ctra_consumo_ndet':"df[1305100]+df[1305300]+df[1305400]+df[1305800]+df[1305900]-df[8115000]",
             'ctra_consumo_det':"df[8115000]",
             'ctra_deudores_en_cc':"df[1305300]"}
    

    Dict_Reporte_Carteras2022={'ctra_creditos_por_tc':"df[148000301]",
                        'ctra_utilizaciones_de_tc_por_cobrar':"df[148000302]",
                        'ctra_consumo_cuotas_ctra_det':"df[811400100]",
                        'ctra_consumo_cuotas_ctra_ndet':"df[148000100]-df[811400100]",
                        'ctra_deudores_cc':"df[148000200]",
                        'ctra_deudores_cc_ctra_det':"df[811400200]",
                        'ctra_deudores_cc_ctra_ndet':"df[148000200]-df[811400200]",
                        'ctra_deudores_tc_ctra_det':"df[811400300]",
                        'ctra_deudores_tc_ctra_ndet':"df[148000300]-df[811400300]",
                        'ctra_consumo_cuotas':"df[148000100]",
                        'ctra_consumo_tarjetas':"df[148000300]",
                        'ctra_consumo_total':"df[148000100]+df[148000200]+df[148000300]+df[148000400]+df[148000900]",
                        'ctra_consumo_ndet':"df[148000100]+df[148000200]+df[148000300]+df[148000400]+df[148000900]-df[811400000]",
                        'ctra_consumo_det':"df[811400000]",
                        'ctra_deudores_en_cc':"df[148000200]"
                        }
    Dict_Reporte_Carteras2021=None

    '''Directorio con la información de get_cmf_info'''
    try: 
        tabla=create_all_data(info_directory,dict_report_v_2022,dict_report_v_2021,Dict_Reporte_Carteras2022,Dict_Reporte_Carteras2021,output_file)
        return tabla
    except FileNotFoundError:
        print("No se encuentra la información con la ruta especificada")
        return


if __name__ == '__main__':
    info_directory=r'InfCMF'
    output_file = f'database.csv'
    tabla = get_historical_table(info_directory,output_file)

    


    
    