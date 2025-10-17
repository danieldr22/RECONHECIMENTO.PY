import cv2
import os
import time
import sys # <--- NOVO: Para o script poder "sair"

# --- CONFIGURAÇÕES ---
RECOGNIZER_PATH = 'trainer.yml'
CLASSIFICADOR_PATH = 'haarcascade_frontalface_default.xml'
KNOWN_FACES_DIR = 'known_faces'
CONFIANCA_MINIMA = 90 

# --- CARREGAR NOMES DAS PESSOAS ---
known_faces_names = []
label_map = {} 
current_id = 0

# A lógica de carregar nomes deve ser robusta
for name in sorted(os.listdir(KNOWN_FACES_DIR)):
    if name.startswith('.'): 
        continue
    # Pega o nome base (ex: 'daniel' de 'daniel_123.jpg')
    label_name = os.path.splitext(name)[0].split('_')[0] 
    if label_name not in label_map:
        label_map[label_name] = current_id
        known_faces_names.append(label_name) 
        current_id += 1
        
print(f"Pessoas conhecidas carregadas (ID -> Nome):")
for i, name in enumerate(known_faces_names):
    print(f"{i} -> {name}")

# --- INICIALIZAR RECONHECEDOR E DETECTOR ---
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
except AttributeError:
    recognizer = cv2.createLBPHFaceRecognizer()
    
# Verifica se o trainer.yml existe antes de ler
if not os.path.exists(RECOGNIZER_PATH):
    print(f"AVISO: Arquivo '{RECOGNIZER_PATH}' não encontrado.")
    print("Rostos 'Desconhecidos' serão salvos para o primeiro treino.")
else:
    recognizer.read(RECOGNIZER_PATH) 
    
face_detector = cv2.CascadeClassifier(CLASSIFICADOR_PATH)

# --- LOOP PRINCIPAL DA CÂMERA ---
print(">>> Iniciando câmera...")
print(">>> Procurando por novos rostos para aprender...")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]

        # Tenta prever o rosto. Se o trainer.yml não existir,
        # 'confidence' será muito alta, caindo direto no 'else'.
        try:
            id_, confidence = recognizer.predict(roi_gray)
        except cv2.error:
            # Acontece se o 'trainer.yml' estiver vazio ou for o primeiro uso
            id_ = -1
            confidence = 100 # Força a ser "Desconhecido"

        if id_ != -1 and confidence < CONFIANCA_MINIMA: 
            if id_ < len(known_faces_names):
                name = known_faces_names[id_]
                label = f"{name} ({confidence:.1f})"
            else:
                name = "ID Fora do Alcance"
                label = name
            color = (0, 255, 0) # Verde
        else:
            label = "Desconhecido"
            color = (0, 0, 255) # Vermelho
            
            # --- LÓGICA DE AUTO-SALVAR E SAIR ---
            # Se um "Desconhecido" é visto, salva a imagem e sai.
            
            print(f"[AUTO-SAVE] Novo rosto '{label}' detectado. Salvando...")
            
            # Cria um nome de arquivo único para 'Desconhecido'
            filename = f"Desconhecido_{int(time.time())}.jpg"
            filepath = os.path.join(KNOWN_FACES_DIR, filename)
            
            # Salva a imagem do rosto (em escala de cinza)
            cv2.imwrite(filepath, roi_gray)
            
            print(f"[AUTO-SAVE] Salvo como: {filepath}")
            print("[AUTO-SAVE] Sinalizando para o gerenciador re-treinar...")
            
            # Libera a câmera e fecha as janelas
            cap.release()
            cv2.destroyAllWindows()
            
            # Sai do script com um código especial (10)
            # O 'manager.py' vai detectar esse código.
            sys.exit(10)
            # --- FIM DA LÓGICA ---

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow('Reconhecimento Facial - Pressione "q" para sair', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break # Sai com código 0 (padrão), sinalizando encerramento normal

print("Encerrando...")
cap.release()
cv2.destroyAllWindows()
# Se sair por 'q', sys.exit(0) é chamado implicitamente