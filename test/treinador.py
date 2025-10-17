import cv2
import os
import numpy as np
from PIL import Image

CLASSIFICADOR_PATH = 'haarcascade_frontalface_default.xml'
FACES_DIR = 'known_faces' # Esta é a pasta base

print(">>> Verificando arquivos...")

if not os.path.exists(CLASSIFICADOR_PATH):
    print(f"ERRO: Não foi possível encontrar o arquivo classificador: {CLASSIFICADOR_PATH}")
    exit()

if not os.path.exists(FACES_DIR):
    print(f"ERRO: A pasta '{FACES_DIR}' não foi encontrada.")
    exit()

print(">>> Arquivos encontrados. Inicializando...")

face_detector = cv2.CascadeClassifier(CLASSIFICADOR_PATH)
try:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
except AttributeError:
    print("AVISO: Usando 'cv2.createLBPHFaceRecognizer()'")
    recognizer = cv2.createLBPHFaceRecognizer()

def get_image_data(base_path):
    """
    Lê as subpastas de 'base_path'.
    Cada subpasta é tratada como uma pessoa diferente.
    """
    face_samples = []
    ids = []
    label_ids = {} # Mapeia nome da pessoa (string) -> id (int)
    current_id = 0

    print(f"\n>>> Lendo pastas de pessoas em '{base_path}'...")
    
    # Itera sobre cada item na pasta base (ex: 'daniel', 'Pessoa_123')
    for person_name in os.listdir(base_path):
        person_path = os.path.join(base_path, person_name)
        
        # Se não for um diretório (pasta), pula
        if not os.path.isdir(person_path):
            continue
        
        # Se é um diretório, é uma pessoa. Atribui um ID.
        if person_name not in label_ids:
            label_ids[person_name] = current_id
            current_id += 1
            
        person_id = label_ids[person_name]
        
        print(f"--- Processando '{person_name}' (ID: {person_id})")
        
        # Agora, lê todas as imagens dentro da pasta da pessoa
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            
            try:
                pil_image = Image.open(image_path).convert('L') # Converte para cinza
                image_array = np.array(pil_image, 'uint8')
            except Exception as e:
                print(f"  AVISO: Não foi possível ler {image_path}. Pulando. (Erro: {e})")
                continue

            faces = face_detector.detectMultiScale(image_array, scaleFactor=1.1, minNeighbors=5)

            if not len(faces):
                 print(f"  AVISO: Nenhum rosto detectado em {image_path}. Pulando.")

            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]
                face_samples.append(roi)
                ids.append(person_id)
                
    print(f"\nPessoas encontradas e mapeadas: {label_ids}")
    return face_samples, ids, label_ids

print(">>> Iniciando treinamento...")
faces, ids, labels_map = get_image_data(FACES_DIR)

if not faces:
    print(f"\nAVISO: O treinamento falhou (ou não há rostos para treinar).")
    print(f"Nenhum rosto foi aprendido. A pasta '{FACES_DIR}' pode estar vazia.")
    print("O arquivo 'trainer.yml' não será criado/atualizado.")
else:
    print("\n>>> Treinando o modelo...")
    recognizer.train(faces, np.array(ids))
    recognizer.save('trainer.yml')
    print(f"\n>>> SUCESSO! Modelo treinado com {len(faces)} imagens e salvo como 'trainer.yml'")
    print(">>> Mapa de IDs:", labels_map)
