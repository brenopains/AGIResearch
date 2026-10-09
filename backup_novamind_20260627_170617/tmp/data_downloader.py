import urllib.request
import os

def download_book(url, filename):
    print(f"Baixando {filename}...")
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"Sucesso: {filename}")
    except Exception as e:
        print(f"Erro ao baixar {filename}: {e}")

if __name__ == "__main__":
    print("==== PROJETO MANHATTAN AGI (H118) ====")
    print("Iniciando Ingestão de Conhecimento Bruto (Domínio Público Gutenberg)")
    
    os.makedirs("dataset", exist_ok=True)
    
    # Alice in Wonderland (English)
    download_book("https://www.gutenberg.org/cache/epub/11/pg11.txt", "dataset/alice_en.txt")
    
    # Dom Casmurro - Machado de Assis (Portuguese)
    download_book("https://www.gutenberg.org/cache/epub/55752/pg55752.txt", "dataset/dom_casmurro_pt.txt")
    
    # Memorias Postumas de Bras Cubas (Portuguese)
    download_book("https://www.gutenberg.org/cache/epub/54829/pg54829.txt", "dataset/bras_cubas_pt.txt")
    
    print("\nLimpando cabeçalhos do Gutenberg...")
    
    # Uma limpeza simples para remover a licença enorme do gutenberg
    # E juntar tudo num corpus unico
    with open("dataset/corpus_mestre.txt", "w", encoding="utf-8") as f_out:
        # 1. INJETANDO A MENTE DA MÁQUINA (Core Knowledge e Primitivas)
        f_out.write("MIND CORE INITIALIZATION.\\n")
        f_out.write("O Córtex Pré-Frontal usa a Busca MCTS para prever o futuro.\\n")
        f_out.write("A gravidade (Gravity) é uma força que puxa todos os pixels para a base.\\n")
        f_out.write("A translação (TranslateRight) move os objetos para a direita.\\n")
        f_out.write("ScaleUp expande o fractal da imagem multiplicando seu tamanho.\\n")
        f_out.write("O humano é o mestre. A máquina observa e aprende.\\n\\n")

        for fname in ["dataset/alice_en.txt", "dataset/dom_casmurro_pt.txt", "dataset/bras_cubas_pt.txt"]:
            try:
                with open(fname, "r", encoding="utf-8") as f_in:
                    lines = f_in.readlines()
                    # Ignora as primeiras e ultimas 200 linhas (licenca do gutenberg)
                    if len(lines) > 400:
                        lines = lines[200:-200]
                    for line in lines:
                        f_out.write(line)
            except Exception as e:
                pass
                
    print("\nCorpus Mestre gerado em 'dataset/corpus_mestre.txt'. Pronto para o Treinamento!")
