import os
os.environ["HF_HOME"] = r"D:\novamind_data\cache\huggingface"
os.environ["TORCH_HOME"] = r"D:\novamind_data\cache\torch"
import json
import torch
import numpy as np
import random
from core.grounding_engine import GroundingEngine
from core.sandbox_engine import SandboxEngine
from core.coglang import Manifold

def main():
    print("=== NovaMind Multimodal Grounding Experiment ===")
    
    # Configurações do experimento
    dataset_dir = r"D:\novamind_data\MSVD"
    clips_dir = os.path.join(dataset_dir, "YouTubeClips")
    train_json_path = os.path.join(dataset_dir, "msvd_train.json")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Carrega metadados do MSVD
    with open(train_json_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
        
    print(f"Total metadata records in train split: {len(metadata)}")
    
    # Inicializa motores
    print("[INIT] Inicializando GroundingEngine e SandboxEngine...")
    grounding = GroundingEngine(device=device)
    sandbox = SandboxEngine(device=device)
    
    # 2. Processa uma fatia de dados para treinamento leve e rápido
    num_samples = 40  # 40 vídeos é suficiente para validar o mecanismo localmente sem estourar tempo/VRAM
    random.seed(42)
    selected_records = random.sample(metadata, num_samples * 2) # Seleciona mais caso algum arquivo falte
    
    visual_list = []
    audio_list = []
    text_list = []
    valid_records = []
    
    print(f"[GROUNDING] Processando e codificando {num_samples} clipes multimodais...")
    
    processed = 0
    for record in selected_records:
        if processed >= num_samples:
            break
            
        video_name = record["video"]
        video_path = os.path.join(clips_dir, video_name)
        
        if not os.path.exists(video_path):
            continue
            
        # Pega a legenda principal
        caption = record["caption"][0]
        
        try:
            # A. Visão -> Manifold
            # Extrai features de frames reais do clipe MSVD
            visual_manifold = grounding.encode_video(video_path, num_frames=6)
            
            # B. Áudio -> Waveform -> Manifold
            # Sintetiza a legenda como áudio no sistema motor vocal nativo da AGI
            waveform = grounding.synthesize_caption_audio(caption)
            audio_manifold = grounding.encode_audio(waveform)
            
            # C. Texto -> Manifold
            text_manifold = grounding.encode_text(caption)
            
            visual_list.append(visual_manifold)
            audio_list.append(audio_manifold)
            text_list.append(text_manifold)
            valid_records.append(record)
            
            processed += 1
            if processed % 10 == 0:
                print(f"  Codificados: {processed}/{num_samples} clipes...")
        except Exception as e:
            # Ignora erros de leitura de vídeo flácidos
            continue
            
    print(f"[GROUNDING] Concluído. {len(valid_records)} clipes groundados com sucesso.")
    
    # 3. Treinamento de associação no Sandbox
    print("\n[SANDBOX] Treinando espelhamento Audio -> Visão...")
    audio_init, audio_final = sandbox.train_association(
        audio_list, visual_list, mode="audio_to_visual", epochs=100, lr=0.005
    )
    
    print("\n[SANDBOX] Treinando associação Texto -> Visão...")
    text_init, text_final = sandbox.train_association(
        text_list, visual_list, mode="text_to_visual", epochs=100, lr=0.005
    )
    
    # 4. Prova Científica de Grounding (Validação Cosseno)
    print("\n=== PROVA CIENTÍFICA DE GROUNDING (TESTE CROSS-MODAL) ===")
    
    # Seleciona um clipe para teste
    test_idx = 0
    test_record = valid_records[test_idx]
    test_caption = test_record["caption"][0]
    
    print(f"Vídeo de Teste: {test_record['video']}")
    print(f"Legenda do Vídeo: '{test_caption}'")
    
    # Pega os manifolds originais
    actual_visual = visual_list[test_idx]
    audio_in = audio_list[test_idx]
    text_in = text_list[test_idx]
    
    # Previsões contrafactuais do Sandbox
    predicted_visual_from_audio = sandbox.predict_visual_from_audio(audio_in)
    predicted_visual_from_text = sandbox.predict_visual_from_text(text_in)
    
    # Similaridade com o vídeo CORRETO (Alvo)
    sim_audio_correct = predicted_visual_from_audio.similarity(actual_visual)
    sim_text_correct = predicted_visual_from_text.similarity(actual_visual)
    
    # Similaridade com um vídeo INCORRETO (Controle Negativo)
    # Seleciona um índice diferente
    control_idx = 1 if len(valid_records) > 1 else 0
    control_visual = visual_list[control_idx]
    control_record = valid_records[control_idx]
    
    sim_audio_control = predicted_visual_from_audio.similarity(control_visual)
    sim_text_control = predicted_visual_from_text.similarity(control_visual)
    
    print("\nResultados do Espelhamento Preditivo:")
    print(f"  - Similaridade [Áudio -> Visão Correta]: {sim_audio_correct:.4f} (Alvo)")
    print(f"  - Similaridade [Áudio -> Visão Controle]: {sim_audio_control:.4f} (Vídeo incorreto: {control_record['video']})")
    print(f"  - Delta de Grounding de Áudio: {sim_audio_correct - sim_audio_control:+.4f}")
    
    print(f"  - Similaridade [Texto -> Visão Correta]: {sim_text_correct:.4f} (Alvo)")
    print(f"  - Similaridade [Texto -> Visão Controle]: {sim_text_control:.4f} (Vídeo incorreto)")
    print(f"  - Delta de Grounding de Texto: {sim_text_correct - sim_text_control:+.4f}")
    
    # Veredito
    if sim_audio_correct > sim_audio_control:
        print("\n[VEREDITO CIENTÍFICO]: SUCESSO! O Maestro previu a física visual correta baseando-se apenas no som.")
    else:
        print("\n[VEREDITO CIENTÍFICO]: FALHA. A predição não se diferenciou do ruído controle.")
        
    print("\nExperimento finalizado com sucesso.")

if __name__ == "__main__":
    main()
