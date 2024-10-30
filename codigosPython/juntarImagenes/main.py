import os
import shutil

# Ruta de origen
ruta_origen = r'C:\Users\xatla\Desktop\TTv2\DataSetJunto'

# Ruta de destino
ruta_destino = r'C:\Users\xatla\Desktop\TTv2\DataSetFinal'

# Carpetas de origen y destino
carpetas_origen = ['A_g1', 'A_g2', 'A_g3', 'B_g1', 'B_g2', 'B_g3', 'C_g1', 'C_g2', 'C_g3', 'D_g1', 'D_g2', 'D_g3', 'E_g1', 'E_g2', 'E_g3']
carpetas_destino = ['gesto1', 'gesto2', 'gesto3']

# Crear carpetas de destino si no existen
for carpeta in carpetas_destino:
    ruta_carpeta_destino = os.path.join(ruta_destino, carpeta)
    if not os.path.exists(ruta_carpeta_destino):
        os.makedirs(ruta_carpeta_destino)

# Recorrer las carpetas de origen
for carpeta_origen in carpetas_origen:
    # Comprobar si la carpeta de origen termina en g1, g2 o g3
    if carpeta_origen.endswith('g1'):
        indice_destino = 0
    elif carpeta_origen.endswith('g2'):
        indice_destino = 1
    elif carpeta_origen.endswith('g3'):
        indice_destino = 2
    else:
        continue  # Ignorar carpetas que no cumplan con el patrón

    ruta_carpeta_origen = os.path.join(ruta_origen, carpeta_origen)
    imagenes = os.listdir(ruta_carpeta_origen)
    carpeta_destino = os.path.join(ruta_destino, carpetas_destino[indice_destino])

    # Copiar y renombrar imágenes al destino en orden
    for i, imagen in enumerate(imagenes):
        nombre_destino = f"{i + 1}_{carpetas_destino[indice_destino]}.jpg"  # Cambia la extensión si es necesario
        ruta_origen_imagen = os.path.join(ruta_carpeta_origen, imagen)
        ruta_destino_imagen = os.path.join(carpeta_destino, nombre_destino)
        shutil.copy(ruta_origen_imagen, ruta_destino_imagen)

print("Proceso completado.")
