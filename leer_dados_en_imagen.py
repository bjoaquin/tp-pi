import cv2
import numpy as np

def leer_dados_en_imagen(img):
    # Convertir imagen de BGR a RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Recortar imagen para enfocarse en el area de los dados
    x_start, y_start = 30, 0
    x_end, y_end = 1050, 1600
    cropped_image = rgb_img[y_start:y_end, x_start:x_end]

    # Separar canales de color (R, G, B)
    r, g, b = cv2.split(cropped_image)

    # Umbralizar canal rojo
    _, thresh_binary = cv2.threshold(r, 60, 255, cv2.THRESH_BINARY)

    # Funcion para rellenar (Ref: PDI_U7_Descriptores_ej1_Monedas_y_Dados.py)
    def fillhole(input_image):
        im_flood_fill = input_image.copy()
        h, w = input_image.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        im_flood_fill = im_flood_fill.astype("uint8")
        cv2.floodFill(im_flood_fill, mask, (0, 0), 255)
        im_flood_fill_inv = cv2.bitwise_not(im_flood_fill)
        img_out = input_image | im_flood_fill_inv
        return img_out

    # Rellenar huecos en la imagen binaria
    img_rellena = fillhole(thresh_binary)

    # Apertura para suavizar bordes
    img_clean = cv2.morphologyEx(img_rellena, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30,30)))

    # Deteccion de componentes (idealmente 6: el fondo + 5 dados)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img_clean)

    # Si no se detectan dados, retornar imagen original
    if num_labels == 1:
        return rgb_img

    # Inicializar mascara donde figuren solamente los dados
    mask_dados = np.zeros_like(img_clean)

    for i in range(1, num_labels):

        # Obtener contorno del i-esimo objeto
        obj = (labels == i).astype(np.uint8) * 255
        contour, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Obtener rectangulo correspondiente al i-esimo objeto (potencialmente, un dado)
        (x, y), (w, h), _ = cv2.minAreaRect(contour[0])
        aspect_ratio = w/h
        comp_to_box_ratio = stats[i][4] / (w*h)

        # Retener dados y descartar lo demas
        # [Regla: si (2000 < area < 6000) y (0.8 < aspect_ratio < 1.15), es dado]
        #if 2000 < stats[i][4] < 6000 and 0.8 < aspect_ratio < 1.15:
        if 2000 < stats[i][4] < 6000 and comp_to_box_ratio > 0.85:
            mask_dados += obj

    # Aplicar mascara de dados sobre el canal azul
    img_dados = cv2.bitwise_and(b, b, mask=mask_dados)

    # Umbralizar para obtener pips (valores) de los dados
    _, img_dados_th = cv2.threshold(img_dados, 175, 255, cv2.THRESH_BINARY)

    # Clausura para rellenar los pips
    img_valores_close = cv2.morphologyEx(img_dados_th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10)))
    img_valores_open = cv2.morphologyEx(img_valores_close, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)))

    # Eliminar pips laterales (pequeños)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(img_valores_open)
    img_valores_clean = img_valores_open.copy()
    for i in range(1, num_labels):
        area = stats[i][4]
        if(area < 30):
            img_valores_clean[labels == i] = 0

    # Contar pips en cada dado y etiquetarlos en la imagen original
    # Para ello, primero se hallan las componentes conectadas de los dados y se recorre cada objeto detectado.
    # Luego, dentro del dado, se vuelven a hallar las componentes conectadas y se cuentan.
    img_dados_con_valores = rgb_img.copy()
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_dados)

    for i in range(1, num_labels): # i = 0 es el fondo (se omite)
        mask = (labels == i).astype(np.uint8)

        # Obtener contornos del i-esimo dado
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) == 0:
            continue
        contour = cnts[0]

        # Contar pips dentro del i-esimo dado
        vals = img_valores_clean * mask
        num_vals, _ = cv2.connectedComponents(vals)
        val_dado = num_vals - 1

        if val_dado > 0:
          # Rectangulo de area minima (bounding box del dado)
          rect = cv2.minAreaRect(contour)
          box = cv2.boxPoints(rect)
          box = np.intp(box)

          # Dibujar bounding box azul
          shift = np.array([x_start, 0], dtype=int) # Se desplaza para compensar el cropping
          box_shifted = box + shift
          cv2.drawContours(img_dados_con_valores, [box_shifted], 0, (0,0,255), 2)

          # Escribir etiqueta (numero de pips en el dado)
          x = stats[i, cv2.CC_STAT_LEFT] - 10
          y = stats[i, cv2.CC_STAT_TOP] - 10
          cv2.putText(img_dados_con_valores, str(val_dado), (x, y), cv2.FONT_HERSHEY_DUPLEX, 4, (0,0,255), 6)

    return img_dados_con_valores



# Ejemplo de uso
if __name__ == "__main__":
    filename_in = 'screenshot_1.png'
    filename_out = filename_in.replace('.png', '_procesado.png')
    
    # Leer imagen de prueba
    frame = cv2.imread('input/' + filename_in)
    # Procesar imagen
    frame_modificado = leer_dados_en_imagen(frame)
    # Reordenar canales de color
    frame_modificado_rgb = cv2.cvtColor(frame_modificado, cv2.COLOR_BGR2RGB)
    
    # Mostrar resultado
    cv2.imshow('Prueba', frame_modificado_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('output/' + filename_out, frame_modificado_rgb)