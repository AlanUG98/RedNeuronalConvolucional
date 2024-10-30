import os
import shutil

# Ruta de la carpeta de origen
carpeta_origen = r'C:\Users\xatla\Desktop\TTv2\MaterialDataSet\DataSetCrudo\pruebas\gesto3'

# Ruta de la carpeta de destino
carpeta_destino = r'C:\Users\xatla\Desktop\TTv2\datasetFInal\pruebasexternas\gesto3'

# Obtener la lista de archivos en la carpeta de origen
archivos = os.listdir(carpeta_origen)

# Contador para los nombres de destino
contador = 1

# Iterar a través de los archivos en la carpeta de origen
for archivo in archivos:
    # Construir el nuevo nombre de archivo
    nuevo_nombre = f'g3_{contador}.jpg'

    # Ruta completa de origen
    ruta_origen = os.path.join(carpeta_origen, archivo)

    # Ruta completa de destino
    ruta_destino = os.path.join(carpeta_destino, nuevo_nombre)

    # Mover y renombrar el archivo
    shutil.move(ruta_origen, ruta_destino)

    # Incrementar el contador
    contador += 1

print('Proceso completado.')
