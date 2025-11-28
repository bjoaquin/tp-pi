import cv2
from leer_dados_en_imagen import leer_dados_en_imagen

def leer_dados_en_video(filename_in):
    # Cargar video de entrada
    video = cv2.VideoCapture(filename_in)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))      # Ancho (cantidad de columnas)
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))    # Alto (cantidad de filas)
    fps = int(video.get(cv2.CAP_PROP_FPS))                # Velocidad del video (fps: frame per second)

    # Generar el contenedor del video de salida
    filename_out = filename_in.replace('input', 'output').replace('.mp4', '_procesado.mp4')
    video_out = cv2.VideoWriter(filename_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))

    # Recorrer el video de entrada, analizar, modificar y grabar en el video de salida
    while (video.isOpened()):
        ret, frame = video.read()  # Obtener un frame
        if ret:    # ret==True si la lectura fue exitosa
            # Procesar frame
            new_frame = leer_dados_en_imagen(frame)
            new_frame_rgb = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)

            # Grabar frame en el video de salida
            video_out.write(new_frame_rgb)
        else:
            break  # Abortar si hubo problemas al leer el frame.

    video.release()
    video_out.release()
    cv2.destroyAllWindows()
   
   

# Ejemplo de uso
if __name__ == "__main__":
    filename_in = 'tirada_6.mp4'
    leer_dados_en_video('./input/' + filename_in)