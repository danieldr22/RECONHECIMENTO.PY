import cv2
import os
import numpy as np
from PIL import Image

# Caminho para o classificador de rostos do OpenCV
CLASSIFICADOR_PATH = 'haarcascade_frontalface_default.xml'
# Caminho para a pasta com os rostos conhecidos
FACES_DIR = 'known_faces'

print(">>> Verificando arquivos...") # <--- ESTA LINHA VAI EXECUTAR PRIMEIRO

# --- Verificações Iniciais ---
if not os.path.exists(CLASSIFICADOR_PATH):
    print(f"ERRO: Não foi possível encontrar o arquivo classificador:")
    print(f"{CLASSIFICADOR_PATH}")
    print("Por favor, baixe este arquivo e coloque na mesma pasta do script.")
    exit()

if not os.path.exists(FACES_DIR):
    print(f"ERRO: A pasta '{FACES_DIR}' não foi encontrada.")
    print("Por favor, crie esta pasta e adicione as fotos de rosto nela.")
    exit()

print(">>> Arquivos encontrados. Inicializando...")

# Inicializa o detector de rostos
face_detector = cv2.CascadeClassifier(CLASSIFICADOR_PATH)

# Inicializa o reconhecedor LBPH
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
except AttributeError:
    print("AVISO: Sua versão do OpenCV não tem 'cv2.face'.")
    print("Tentando inicializar com a versão antiga 'cv2.createLBPHFaceRecognizer()'")
    try:
        # Esta é a versão para OpenCV 2 ou 3
        recognizer = cv2.createLBPHFaceRecognizer()
    except Exception as e:
        print(f"ERRO FATAL: Falha ao inicializar o reconhecedor do OpenCV: {e}")
        print("Tente instalar o pacote 'opencv-contrib-python' com: pip install opencv-contrib-python")
        exit()

def get_image_data(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.startswith('.')] # Ignora arquivos ocultos
    face_samples = []
    ids = []
    
    current_id = 0
    label_ids = {} 

    print(f"\n>>> Lendo imagens da pasta '{FACES_DIR}'...")
    
    for image_path in image_paths:
        label_name = os.path.basename(image_path).split('.')[0]
        
        if not label_name: # Pula se o nome do arquivo for algo como ".jpg"
            continue
            
        if label_name not in label_ids:
            label_ids[label_name] = current_id
            current_id += 1
            
        person_id = label_ids[label_name]

        try:
            pil_image = Image.open(image_path).convert('L') # 'L' = Escala de cinza
            image_array = np.array(pil_image, 'uint8')
        except Exception as e:
            print(f"AVISO: Não foi possível ler a imagem {image_path}. Pulando. (Erro: {e})")
            continue

        faces = face_detector.detectMultiScale(image_array, scaleFactor=1.1, minNeighbors=5)

        if not len(faces):
             print(f"AVISO: Nenhum rosto detectado em {image_path}. Pulando.")

        for (x, y, w, h) in faces:
            roi = image_array[y:y+h, x:x+w]
            face_samples.append(roi)
            ids.append(person_id)
            
    print(f"Pessoas encontradas e mapeadas: {label_ids}")
    return face_samples, ids, label_ids

print(">>> Iniciando treinamento...")
faces, ids, labels_map = get_image_data(FACES_DIR)

if not faces:
    print(f"\nERRO: O treinamento falhou.")
    print(f"Nenhum rosto foi aprendido. Verifique sua pasta '{FACES_DIR}':")
    print("1. Ela está vazia?")
    print("2. As fotos estão nítidas e com o rosto de frente?")
else:
    print("\n>>> Treinando o modelo...")
    recognizer.train(faces, np.array(ids))
    recognizer.save('trainer.yml')
    print(f"\n>>> SUCESSO! Modelo treinado com {len(faces)} imagens e salvo como 'trainer.yml'")
    print(">>> Mapa de IDs:", labels_map)