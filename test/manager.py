mport subprocess
import time
import os

print("--- INICIANDO GERENCIADOR DE RECONHECIMENTO ---")
print("Este script irá reiniciar o programa de câmera sempre que")
print("um novo rosto 'Desconhecido' for aprendido.")
print("Pressione Ctrl+C neste terminal para parar tudo.")
print("="*50)

# Garante que os caminhos para os scripts estão corretos
RCF_SCRIPT = 'RCF.py'
TRAINER_SCRIPT = 'treinador.py'

# Verifica se os scripts existem
if not os.path.exists(RCF_SCRIPT) or not os.path.exists(TRAINER_SCRIPT):
    print(f"ERRO: Não foi possível encontrar '{RCF_SCRIPT}' ou '{TRAINER_SCRIPT}'.")
    print("Certifique-se de que todos os 3 scripts estão na mesma pasta.")
    exit()

# Loop de execução principal
while True:
    print(f"\n[MANAGER] Iniciando '{RCF_SCRIPT}' (câmera)...")
    
    # Executa o RCF.py e espera ele terminar.
    # Usamos 'python' para garantir que ele seja executado pelo interpretador
    result = subprocess.run(['python', RCF_SCRIPT])
    
    # RCF.py foi modificado para sair com código 10 se um novo rosto for salvo
    if result.returncode == 10:
        print(f"\n[MANAGER] '{RCF_SCRIPT}' sinalizou um novo rosto.")
        print(f"[MANAGER] Iniciando treinamento ('{TRAINER_SCRIPT}')...")
        
        # Roda o script de treinamento
        try:
            subprocess.run(['python', TRAINER_SCRIPT], check=True)
            print("[MANAGER] Treinamento concluído com sucesso.")
        except subprocess.CalledProcessError:
            print("[MANAGER] ERRO: Falha durante o treinamento.")
            print("Verifique o script 'treinador.py' e as imagens.")
            break # Interrompe o loop em caso de erro de treino

        print("[MANAGER] Reiniciando a câmera em 3 segundos...")
        time.sleep(3)
    
    else:
        # Se o RCF.py saiu por outro motivo (ex: usuário apertou 'q')
        print(f"\n[MANAGER] '{RCF_SCRIPT}' foi encerrado pelo usuário (código {result.returncode}).")
        print("Finalizando o gerenciador.")
        break # Sai do loop while True e encerra o programa

print("="*50)
print("Gerenciador finalizado.")
