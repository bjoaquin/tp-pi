from leer_dados_en_video import leer_dados_en_video
from pathlib import Path

# Listar todos los archivos .mp4 en la carpeta "input"
filenames = list(Path("./input").glob("*.mp4"))

for filename in filenames:
    print("Procesando:", filename)
    leer_dados_en_video(str(filename))