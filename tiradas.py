import cv2
import numpy as np
import os 
import glob 

# ==========================================================
# AGREGAMOS LA SIGUIENTE SECCIÓN PARA ACTIVAR EL MODO VER PASO A PASO
# A NIVEL FUNCIONAL, EL VIDEO SIN EL PASO A PASO CUMPLE CON LAS CONSIGNAS
# PERO EN CASO QUE DESEEN VER TAMBIEN COMO ACTUA LA MASCARA SOBRE EL VIDEO ORIGINAL
# Y COMO APLICAN LOS BOUNDING BOXES, PUEDEN ACTIVARLO DESCOMENTANDO LA LINEA DE ABAJO Y VER MAS VIDEOS.
# ==========================================================
ver_mas = " "
# DESCOMENTAR LA LINEA DE ABAJO PARA 2 VIDEOS QUE AGREGAN DETALLE
# ver_mas = "QUIERO VER EL PASO A PASO" 

# ==========================================================

#Configuraciones Globales
PATRON_VIDEO = "frames/tirada_*.mp4" # Patrón para buscar videos
CARPETA_SALIDA = "tiradas_salida"

ANCHO_MAX_DISPLAY = 450
AREA_MIN_DADO = 100         # Área mínima MUY PEQUEÑA para descartar menores
AREA_MIN_PUNTO = 5           # Área mínima (Píxeles) del punto.
UMBRAL_MOVIMIENTO = 15000    # Píxeles de diferencia máximos entre Frames para considerar el frame como estático   
CONTEO_FRAME_ESTATICO = 10     # Frames consecutivos necesarios para confirmar que los dados están quietos
TOLERANCIA_ASPECT_RATIO = 0.3  # Tolerancia para la relación de aspecto es decir +- 30%

# Umbrales HSV fijos (inferidos previamente con Display):
H_MIN = 30
H_MAX = 179
S_MIN = 125
S_MAX = 255
V_MIN = 0
V_MAX = 255

# Nombres de los dados
NOMBRES_DADOS = [
    "Dado A", "Dado B", "Dado C", "Dado D", "Dado E", "Dado F", 
    "Dado G", "Dado H", "Dado I", "Dado J", "Dado K"]


# Funcion para Reproducción
def redimensionar_para_mostrar(img, ancho_max=ANCHO_MAX_DISPLAY):
    """Redimensiona la imagen para ajustarla al ancho máximo de visualización."""
    h, w = img.shape[:2]
    if w <= ancho_max:
        return img
    escala = ancho_max / float(w)
    nuevo_w = int(w * escala)
    nuevo_h = int(h * escala)
    return cv2.resize(img, (nuevo_w, nuevo_h), interpolation=cv2.INTER_AREA)


#FUNCIONES DE DETECCIÓN Y ANÁLISIS
def obtener_mascara_cuerpo_dado(frame_hsv):
    """Obtiene la máscara para el cuerpo de los dados (el color rojo/no-fondo) usando valores fijos."""
    
    lower = np.array([H_MIN, S_MIN, V_MIN], dtype=np.uint8) #Piso HSV
    upper = np.array([H_MAX, S_MAX, V_MAX], dtype=np.uint8) #Techo HSV
    
    mascara_original = cv2.inRange(frame_hsv, lower, upper)
    mascara = cv2.bitwise_not(mascara_original) #Invertimos

    kernel = np.ones((5, 5), np.uint8) #Habíamos probado con Kernel de 40x40, pero decidimos aumentar el H_MIN
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel) 
    
    kernel_pequeño = np.ones((3, 3), np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel_pequeño) 
    
    return mascara


def obtener_mascara_puntos(frame_hsv):
    """Máscara para los puntos blancos de los dados."""
    lower_white = np.array([0, 0, 200], dtype=np.uint8)
    upper_white = np.array([179, 50, 255], dtype=np.uint8)
    
    mascara_puntos = cv2.inRange(frame_hsv, lower_white, upper_white)
    
    return mascara_puntos

def analizar_dado(mascara_puntos_roi):
    """Cuenta el número de puntos dentro de la ROI de un dado."""
    
    _, thresh_puntos = cv2.threshold(mascara_puntos_roi, 100, 255, cv2.THRESH_BINARY)
    contornos_puntos, _ = cv2.findContours(thresh_puntos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Al igual que en el de monedas, usamos external
    
    conteo_puntos = 0
    for contorno_punto in contornos_puntos:
        if cv2.contourArea(contorno_punto) > AREA_MIN_PUNTO:
            conteo_puntos += 1
            
    return conteo_puntos


def analizar_frame_estatico(frame, mascara_dados, frame_hsv):
    """
    Función principal de análisis para detectar dados, contar el valor y asignar nombres.
    """
    H, W = frame.shape[:2]
    
    contornos_dados, _ = cv2.findContours(mascara_dados, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    resultados_dados = []
    
    # MODIFICACIÓN 2: Usamos esta máscara aquí para la detección de puntos, 
    # y la pasamos para la visualización de debug más adelante.
    mascara_puntos_completa = obtener_mascara_puntos(frame_hsv) 

    for contorno in contornos_dados:
        area = cv2.contourArea(contorno)
        x, y, w, h = cv2.boundingRect(contorno)
        
        descartado = False
        razon_descarte = "" 

        # Filtros de descarte
        if area < AREA_MIN_DADO:
            descartado = True
            razon_descarte = "AREA"
        aspect_ratio = w / h
        if not (1.0 - TOLERANCIA_ASPECT_RATIO <= aspect_ratio <= 1.0 + TOLERANCIA_ASPECT_RATIO):
            descartado = True
            razon_descarte = "ASPECTO"
        if w > W * 0.95 or h > H * 0.95 or x < 5 or y < 5 or (x + w) > W - 5 or (y + h) > H - 5:
            descartado = True
            razon_descarte = "BORDE/TAMAÑO"
        if descartado:
            continue 

        # Si pasa los filtros: Analizar puntos
        mascara_puntos_roi = mascara_puntos_completa[y:y+h, x:x+w]
        valor_dado = analizar_dado(mascara_puntos_roi)

        resultados_dados.append({
            'bbox': (x, y, w, h),
            'valor': valor_dado,
            'aspect_ratio': aspect_ratio,
        })

    # Ordenar los dados por su posición x para asignar nombres consistentes (izq a der)
    resultados_dados.sort(key=lambda d: d['bbox'][0])
    
    # Asignar nombres
    for i, resultado in enumerate(resultados_dados):
        if i < len(NOMBRES_DADOS):
            resultado['nombre'] = NOMBRES_DADOS[i]
        else:
            resultado['nombre'] = f"Dado {i+1}" # Fallback
    
    #Devolvemos también la máscara de puntos para el paso a paso
    return resultados_dados, mascara_puntos_completa


# MAIN LOOP (AUTOMATICO)
def main():
    # Obtener videos "frame_n.mp4"
    archivos_video = sorted(glob.glob(PATRON_VIDEO))
    if not archivos_video:
        print(f"ERROR: No se encontraron videos que coincidan con el patrón {PATRON_VIDEO}.")
        return
    print(f"Se encontraron {len(archivos_video)} videos para procesar. Iniciando procesamiento automático...")
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    # Iteramos sobre cada video
    for path_video in archivos_video:
        print(f"\n--- Procesando video: {path_video} ---")
        
        # Inicialización de recursos por video
        cap = cv2.VideoCapture(path_video)

        if not cap.isOpened():
            print(f"ERROR: No se pudo abrir el video: {path_video}. Saltando al siguiente.")
            continue
            
        ancho_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        alto_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Grabamos la salida con la velocidad del video Original
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        fps_a_usar = round(original_fps) 
        if fps_a_usar <= 0:
            fps_a_usar = 30

        nombre_base_video = os.path.basename(path_video).split('.')[0]
        nombre_archivo_salida = os.path.join(CARPETA_SALIDA, f"{nombre_base_video}_resultado.mp4")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(nombre_archivo_salida, fourcc, fps_a_usar, (ancho_frame, alto_frame))
        print(f"Grabando salida en: {nombre_archivo_salida}")

        # Variables de estado por video
        mascara_previa = None
        contador_estatico = 0
        datos_dados_finales = [] 
        resultados_impresos = False 
        
        # Variables para DEBUG
        mascara_dados_debug = None
        mascara_puntos_debug = None

        # Bucle de frames para el video actual
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_para_salida = frame.copy() 
            frame_para_mostrar = frame.copy() 
            indice_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

            # Procesamiento
            frame_hsv = cv2.cvtColor(frame_para_salida, cv2.COLOR_BGR2HSV)
            mascara = obtener_mascara_cuerpo_dado(frame_hsv) 
            
            # Guardamos la máscara para el debug (MODIFICACIÓN 5)
            mascara_dados_debug = mascara.copy() 
            
            # Detección de Movimiento/Estatico
            DADOS_ESTATICOS = False
            texto_estado = "MOVIMIENTO"
            
            if mascara_previa is not None:
                #Detección de cambio entre frames
                mascara_diff = cv2.absdiff(mascara, mascara_previa)
                movimiento = np.sum(mascara_diff) / 255.0

                if movimiento < UMBRAL_MOVIMIENTO:
                    contador_estatico += 1
                else:
                    contador_estatico = 0
                    
                DADOS_ESTATICOS = contador_estatico >= CONTEO_FRAME_ESTATICO
                
                if DADOS_ESTATICOS and indice_frame > 15: #Dejamos margen para estabilización inicial
                     texto_estado = "ESTATICO"
                     
                     if not datos_dados_finales:
                         # MODIFICACIÓN 6: Actualizamos la llamada para obtener también la máscara de puntos
                         datos_dados_finales, mascara_puntos_debug = analizar_frame_estatico(frame_para_salida, mascara, frame_hsv)
                         
                     # ... [Lógica de impresión en consola original] ...
                     if datos_dados_finales and not resultados_impresos:
                         valor_total_imprimir = sum(d['valor'] for d in datos_dados_finales)
                         print("\n[DETECCIÓN] --- ¡TIRADA ESTATICA! ---")
                         for resultado in datos_dados_finales:
                             print(f"          {resultado['nombre']}: {resultado['valor']}")
                         print(f"          Total tirada: {valor_total_imprimir}")
                         print("-----------------------------------")
                         resultados_impresos = True
                         
                else:
                     datos_dados_finales = []
                     resultados_impresos = False
                     # Reseteamos la máscara de puntos si no estamos estáticos
                     if ver_mas == "QUIERO VER EL PASO A PASO":
                         mascara_puntos_debug = np.zeros_like(mascara)


            mascara_previa = mascara.copy()

            # PREPARACIÓN PARA VISUALIZACIÓN "ver_mas"
            mascara_con_bbox_debug = None
            if ver_mas == "QUIERO VER EL PASO A PASO":
                # Creamos la ventana de Máscara de Dados con Bounding Box
                mascara_con_bbox_debug = cv2.cvtColor(mascara_dados_debug, cv2.COLOR_GRAY2BGR) #Canal BGR Solo para dibujar lineas
                
            #Dibujar Bounding Boxes en frame_para_salida
            valor_total_salida = 0
            for resultado in datos_dados_finales:
                x, y, w, h = resultado['bbox']
                valor = resultado['valor']
                nombre = resultado['nombre']
                valor_total_salida += valor
                
                # Dibujar Bounding Box ACEPTADO (contorno verde) en Frame Salida
                cv2.rectangle(frame_para_salida, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Dibujar Bounding Box ACEPTADO (contorno verde) en Frame de Pasos
                if ver_mas == "QUIERO VER EL PASO A PASO" and mascara_con_bbox_debug is not None:
                     cv2.rectangle(mascara_con_bbox_debug, (x, y), (x + w, y + h), (0, 0, 255), 2) # Rojo en máscara

                # Poner el nombre y valor
                texto_nombre_valor = f"{nombre}: {valor}"
                cv2.putText(frame_para_salida, texto_nombre_valor, (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2) 
            
            # Información de Estado y Frame
            texto_info_salida = f"Frame: {indice_frame} | Estado: {texto_estado} | Total: {valor_total_salida}"
            cv2.putText(frame_para_salida, texto_info_salida, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Escribir el frame en el video de salida
            out.write(frame_para_salida)

            # Información de Estado y Frame EN EL FRAME DE VISUALIZACIÓN
            valor_total_mostrar = sum(d['valor'] for d in datos_dados_finales) if datos_dados_finales else 0
            num_dados_mostrar = len(datos_dados_finales)
            
            texto_estado_mostrar = texto_estado
            if num_dados_mostrar > 0 and DADOS_ESTATICOS:
                 texto_estado_mostrar = f"{num_dados_mostrar} Dados detectados"

            texto_info_mostrar = f"Frame: {indice_frame} | Estado: {texto_estado_mostrar} | Total: {valor_total_mostrar}"
            cv2.putText(frame_para_mostrar, texto_info_mostrar, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Mostrar UNICAMENTE el video original (solo texto resumen) - ORIGINAL
            frame_disp = redimensionar_para_mostrar(frame_para_mostrar)
            cv2.imshow("Video Original (Deteccion)", frame_disp)
            
            # Visualización Extendida (Con "ver_mas" activado) 
            if ver_mas == "QUIERO VER EL PASO A PASO":
                
                # El video con la mascara de los puntos
                if mascara_puntos_debug is not None:
                     mascara_puntos_disp = redimensionar_para_mostrar(mascara_puntos_debug)
                     cv2.imshow('Mascara de Puntos', mascara_puntos_disp)

                # El video de la mascara aplicada CON bounding box
                if mascara_con_bbox_debug is not None:
                     mascara_bbox_disp = redimensionar_para_mostrar(mascara_con_bbox_debug)
                     cv2.imshow('Mascara con Bounding Box', mascara_bbox_disp)
                     
            # Espera de tecla (Original)
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                 break 
            
        #Liberamos Recursos
        cap.release()
        out.release() 
        # Aseguramos que se cierren todas las ventanas
        cv2.destroyAllWindows()
        if cv2.waitKey(1) & 0xFF == ord('q'): #escape para salir
             break

    print("\nProcesamiento completo de todos los videos.")

if __name__ == "__main__":
    main()