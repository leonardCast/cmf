import os
import re
import shutil
from datetime import datetime
from shutil import rmtree

import io
import requests
import zipfile
from dateutil.relativedelta import relativedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
import time

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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


def string_to_date(element):
    dic_months = {'enero': '01', 'diciembre': '12', 'noviembre': '11', 'octubre': '10', 'septiembre': '09',
                  'agosto': '08', 'julio': '07', 'junio': '06', 'mayo': '05', 'abril': '04', 'marzo': '03',
                  'febrero': '02'}

    month_year_list = element.lower().replace("ir a", "").rstrip().lstrip().split(" ")

    try:
        year = month_year_list[1]
        month = dic_months[month_year_list[0]]
    except KeyError:
        return ("Formato erróneo")

    patron_month = "^0[1-9]|1[0-2]$"
    patron_year = "^(200\d|20[1-9]\d|2100)$"

    if re.match(patron_month, month):
        verify_month = 1
    else:
        verify_month = 0

    if re.match(patron_year, year):
        verify_year = 1
    else:
        verify_year = 0

    if (verify_month and verify_year):
        period = f"{month_year_list[1]}{dic_months[month_year_list[0]]}"
        return period

    return "Formato erróneo"


def get_links_cmf(url):
    ##path of chromedirver##
    browser = webdriver.Chrome(executable_path=r"C:\cmf-data-pipeline\dchrome\chromedriver.exe")
    #########################

    browser.get(url)
    names_and_links = dict()

    elements = browser.find_elements(By.TAG_NAME, "a")
    start=0

    for elem in reversed(elements):
        if ("enero 2009" in elem.get_attribute("title")and start!=1):
            start=1
            date_aux=200901
        
        if ("zip" in elem.get_attribute("href") and start==1):
            names_and_links[str(date_aux)] = elem.get_attribute("href")
            date_aux=delta_months(date_aux,1)

    browser.close()
    return names_and_links

def download_InfoCMF(names_and_links, directory_to_save_inf, date_to_download=None):
    for date_archive, link_archive in names_and_links.items():

        # Verificar si la fecha de este archivo es la que queremos descargar
        if date_to_download and date_archive != date_to_download:
            continue  # Saltar este archivo si no es la fecha deseada

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
        }
        new_directory_name = directory_to_save_inf + '\ ' + date_archive
        new_directory_name = new_directory_name.replace(" ", "", 1)
        
        
        
        session = requests.Session()
        retry = Retry(connect=3, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        response = session.get(link_archive, headers=headers, stream=True)
        
        

        z = zipfile.ZipFile(io.BytesIO(response.content))
        temp_directory_path = directory_to_save_inf + '\\temp'
        z.extractall(temp_directory_path)

        ##know if is just 1 archive
        archivos_despues = os.listdir(temp_directory_path)


        aux_count=1
        if (len(archivos_despues) > 1):

            for archivo in archivos_despues:
                if(archivo=="__MACOSX"):
                    shutil.rmtree(temp_directory_path+"\\__MACOSX")
                    aux_count=2
            


            if(aux_count==1): 

                nueva_carpeta = os.path.join(temp_directory_path, date_archive)
                os.mkdir(nueva_carpeta)
                for archivo in archivos_despues:
                    ruta_archivo = os.path.join(temp_directory_path, archivo)
                    if os.path.isfile(ruta_archivo):
                        shutil.move(ruta_archivo, nueva_carpeta)

        new_directory_path = os.path.join(os.path.dirname(directory_to_save_inf), new_directory_name)
        time.sleep(5)
        os.rename(os.path.join(temp_directory_path, os.listdir(temp_directory_path)[0]), new_directory_path)

        try:
            os.rmdir(temp_directory_path)

        except OSError:
            rmtree(temp_directory_path)


def download_last_months(directory_to_save_inf,date_to=None):
    """
    Descarga los últimos meses de la informacion de la CMF que no exista en el repositorio.

    Args:
        directory_to_save_inf(str): en formato r'C:\{directory}', carpeta que se
        guardan los archivos.

        date_to(str): en formato "YYYYMM", si se entrega una fecha solo descargará dicha fecha

    Returns:
        No retorna nada pero agrega a la carpeta entregada los meses faltantes de info desde la
        ultima fecha en el repositorio de información.
    """
    names_and_links = get_links_cmf('https://www.cmfchile.cl/portal/estadisticas/617/w3-propertyvalue-32901.html')
    
    
    if(date_to):
        try:
            download_InfoCMF(names_and_links, directory_to_save_inf, date_to)
            return
        except FileExistsError:
            print("El archivo ya existe")
            try:
                # Borrar la carpeta "temp" y su contenido
                carpeta_a_borrar = os.path.join(directory_to_save_inf, 'temp')
                shutil.rmtree(carpeta_a_borrar)
                carpeta_a_borrar = os.path.join(directory_to_save_inf, date_to)
                shutil.rmtree(carpeta_a_borrar)
                download_InfoCMF(names_and_links, directory_to_save_inf, date_to)
                print(f"Se sobreescribió el periodo: {date_to}")
                return
            except Exception as e:
                print(f'Error al borrar la carpeta: {e}')
                return
            

    
    files = os.listdir(directory_to_save_inf)

    # filtrar por archivos solo con formato YYYYMM
    pattern = r'^\d{6}$'
    files.sort()

    # Obtener el nombre del último archivo
    matching_files = [file for file in files if re.search(pattern, file)]

    if matching_files:
        matching_files.sort()
        latest_file = matching_files[-1]
        date_to_download = delta_months(latest_file, delta=1)
        
    else:
        date_to_download = None
        print('Se inicia descarga de todos los archivos')
        get_historical_data(directory_to_save_inf)
        return


    try: 
        download_InfoCMF(names_and_links, directory_to_save_inf, date_to_download)


    except FileExistsError:
        print("El archivo ya existe, no se ha creado uno nuevo")
        try:
            # Borrar la carpeta "temp" y su contenido
            carpeta_a_borrar = os.path.join(directory_to_save_inf, 'temp')
            shutil.rmtree(carpeta_a_borrar)
        except Exception as e:
            print(f'Error al borrar la carpeta: {e}')
        else:
            print('La carpeta se ha borrado correctamente')

    fecha_break=delta_months(date_to_download, delta=1)        
    fechas_CMF=list(names_and_links.keys())
    if(fecha_break in fechas_CMF):
        download_last_months(directory_to_save_inf)
    else:
        print("Todos los archivos faltantes ya fueron actualizados")
        done="actualizado"
        return done


def chek_folder(_folder):
    if not os.path.exists(_folder):
        os.makedirs(_folder)
    return


def get_historical_data(directory_to_save_inf='InfCMF'):
    chek_folder(directory_to_save_inf)
    if os.listdir(directory_to_save_inf):
        return

    names_and_links = get_links_cmf('https://www.cmfchile.cl/portal/estadisticas/617/w3-propertyvalue-32901.html')
    try:

        download_InfoCMF(names_and_links, directory_to_save_inf)
        return 
    
    except:
        print("Error en la descarga de la informacion")
        return 


if __name__ == '__main__':
    
    get_historical_data(directory_to_save_inf='InfCMF')
    #download_last_months('InfCMF')
    #download_last_months('InfCMF', "202305")

