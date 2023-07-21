from get_cmf_info import download_last_months
from update_table import all_update

if __name__ == '__main__':
    files = 'InfCMF'
    tabla_no_actualizada = 'database.csv'
    try:
        descarga=download_last_months(files)

    except:
        
        print("Error en la descarga de los datos")


    try:
        tabla,file_name=all_update(files,tabla_no_actualizada)
        ##Tabla -> con fecha + id
        tabla.to_csv(file_name, index=False,sep=";")

    except TypeError:
        print("No se puede acceder a la base de datos no actualizada")


    ##Para descarga de un solo mes solo se debe entregar la fecha a descargar
    #download_last_months('InfCMF', "202305")