import cv2
import os
import time
import sys

# --- CONFIGURAÇÕES ---
RECOGNIZER_PATH = 'trainer.yml'
CLASSIFICADOR_PATH = 'haarcascade_frontalface_default.xml'
KNOWN_FACES_DIR = 'known_faces'
CONFIANCA_MINIMA = 90 # Limite para considerar "Conhecido"

# --- CONFIGURAÇÕES DE RASTREAMENTO (ANTI-VOLATILIDADE) ---
DISTANCIA_MAXIMA_RASTRO = 75
FRAMES_PARA_APRENDER = 20 # Nº de frames para "confirmar" um rosto novo

# --- CARREGAR NOMES DAS PESSOAS (LENDO PASTAS) ---
known_faces_names = []
label_map = {} 
current_id = 0

print("Carregando nomes das pastas de pessoas...")
for name in sorted(os.listdir(KNOWN_FACES_DIR)):
    person_path = os.path.join(KNOWN_FACES_DIR, name)
    if os.path.isdir(person_path):
        label_map[name] = current_id
        known_faces_names.append(name) 
        current_id += 1
        
print(f"Pessoas conhecidas carregadas: {known_faces_names}")

# --- INICIALIZAR RECONHECEDOR E DETECTOR ---
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
except AttributeError:
    recognizer = cv2.createLBPHFaceRecognizer()
    
if not os.path.exists(RECOGNIZER_PATH):
    print(f"AVISO: Arquivo '{RECOGNIZER_PATH}' não encontrado.")
    print("Iniciando em modo de 'aprendizado'.")
else:
    recognizer.read(RECOGNIZER_PATH) 
    
face_detector = cv2.CascadeClassifier(CLASSIFICADOR_PATH)

# --- VARIÁVEIS DE RASTREAMENTO ---
# Lista de: [ (centro), best_id, best_conf, frames_desconhecido, roi_para_salvar ]
rostos_rastreados = []

def calcular_centro(x, y, w, h):
    return (x + w // 2, y + h // 2)

def calcular_distancia(centro1, centro2):
    return ((centro1[0] - centro2[0])**2 + (centro1[1] - centro2[1])**2)**0.5

# --- LOOP PRINCIPAL DA CÂMERA ---
print(">>> Iniciando câmera...")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detectadas = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    novos_rostos_rastreados = []
    rostos_detectados_indices = list(range(len(faces_detectadas)))

    # ETAPA 1. RASTREAR ROSTOS EXISTENTES
    for (centro_antigo, id_antigo, conf_antiga, frames_desconhecido, roi_antigo) in rostos_rastreados:
        melhor_distancia = float('inf')
        melhor_indice = -1

        for i in rostos_detectados_indices:
            (x, y, w, h) = faces_detectadas[i]
            centro_novo = calcular_centro(x, y, w, h)
            dist = calcular_distancia(centro_antigo, centro_novo)

            if dist < DISTANCIA_MAXIMA_RASTRO and dist < melhor_distancia:
                melhor_distancia = dist
                melhor_indice = i
        
        # SE ENCONTRAMOS O MESMO ROSTO
        if melhor_indice != -1:
            (x, y, w, h) = faces_detectadas[melhor_indice]
            roi_gray = gray[y:y+h, x:x+w]
            
            try:
                id_, confidence = recognizer.predict(roi_gray)
            except cv2.error:
                id_ = -1
                confidence = 100
            
            nome_atual = "Desconhecido"
            frames_desconhecido_novo = 0
            
            if confidence < CONFIANCA_MINIMA:
                # É um rosto conhecido
                nome_atual = known_faces_names[id_]
                label = nome_atual
                color = (0, 255, 0) # Verde
            else:
                # É um rosto desconhecido (ou falha)
                frames_desconhecido_novo = frames_desconhecido + 1
                label = f"Novo Rosto? ({frames_desconhecido_novo}/{FRAMES_PARA_APRENDER})"
                color = (0, 165, 255) # Laranja
            
            # Adiciona à lista de rastreamento, guardando o MELHOR ID (mesmo que falhe)
            novos_rostos_rastreados.append([calcular_centro(x,y,w,h), id_, confidence, frames_desconhecido_novo, roi_gray])
            rostos_detectados_indices.remove(melhor_indice)
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # ETAPA 2. PROCESSAR ROSTOS NOVOS
    for i in rostos_detectados_indices:
        (x, y, w, h) = faces_detectadas[i]
        roi_gray = gray[y:y+h, x:x+w]
        
        try:
            id_, confidence = recognizer.predict(roi_gray)
        except cv2.error:
            id_ = -1
            confidence = 100
        
        nome = "Desconhecido"
        frames_desconhecido_novo = 0
        if confidence < CONFIANCA_MINIMA:
            nome = known_faces_names[id_]
            label = nome
            color = (0, 255, 0)
        else:
            frames_desconhecido_novo = 1
            label = f"Novo Rosto? (1/{FRAMES_PARA_APRENDER})"
            color = (0, 165, 255)
            
        novos_rostos_rastreados.append([calcular_centro(x,y,w,h), id_, confidence, frames_desconhecido_novo, roi_gray])
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    rostos_rastreados = novos_rostos_rastreados
    
    # --- ETAPA 3: VERIFICAÇÃO DE GATILHO (A LÓGICA ANTI-DUPLICATA) ---
    for (centro, best_id, best_conf, frames_desconhecido, roi_para_salvar) in rostos_rastreados:
        
        # Se o rosto foi estável o suficiente...
        if frames_desconhecido >= FRAMES_PARA_APRENDER:
            
            # DECISÃO: É uma duplicata ou uma pessoa nova?
            # 'best_id' é a melhor suposição do reconhecedor, mesmo que tenha falhado no limite de confiança.
            
            if best_id != -1 and best_id < len(known_faces_names):
                # CASO A: É UMA DUPLICATA
                # A melhor suposição é uma pessoa existente (ex: 'daniel', ID 0)
                person_name = known_faces_names[best_id]
                folder_path = os.path.join(KNOWN_FACES_DIR, person_name)
                print(f"[AUTO-IMPROVE] Rosto instável de '{person_name}' detectado.")
                print(f"    Adicionando foto à pasta '{person_name}' para reforçar o treino.")
                
            else:
                # CASO B: É UMA PESSOA NOVA
                # A melhor suposição foi -1 (ninguém parecido)
                person_name = f"Pessoa_{int(time.time())}"
                folder_path = os.path.join(KNOWN_FACES_DIR, person_name)
                print(f"[AUTO-SAVE] Rosto 'Desconhecido' estável. Criando nova pasta: {person_name}")
                os.makedirs(folder_path)
            
            # Ação de salvar (comum aos dois casos)
            filename = f"{int(time.time())}.jpg"
            filepath = os.path.join(folder_path, filename)
            cv2.imwrite(filepath, roi_para_salvar)
            
            print(f"[AUTO-SAVE] Imagem salva em: {filepath}")
            print("[AUTO-SAVE] Sinalizando para o gerenciador re-treinar...")
            
            cap.release()
            cv2.destroyAllWindows()
            sys.exit(10) # Sinaliza para o manager.py

    cv2.imshow('Reconhecimento Automatico Estavel - Pressione "q" para sair', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

print("Encerrando...")
cap.release()
cv2.destroyAllWindows()
# Se sair por 'q', sys.exit(0) é chamado implicitamente
