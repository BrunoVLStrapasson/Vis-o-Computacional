
import cv2
import numpy as np
import os
import glob

# ============================================================
# CONFIGURACOES
# ============================================================

PASTA_IMAGENS = "imagens"
PASTA_RESULTADOS = "resultados"

LARGURA_VISUALIZACAO = 1000

# Proporcao aproximada de uma folha A4:
# largura / altura = 210 / 297
LARGURA_SAIDA = 840
ALTURA_SAIDA = 1188

os.makedirs(PASTA_RESULTADOS, exist_ok=True)

# Pontos selecionados pelo usuario
pontos = []
imagem_visualizacao = None
imagem_base = None


# ============================================================
# SELECAO DOS QUATRO CANTOS
# ============================================================

def selecionar_ponto(event, x, y, flags, param):
    global pontos, imagem_visualizacao

    if event == cv2.EVENT_LBUTTONDOWN:

        if len(pontos) >= 4:
            return

        pontos.append([x, y])

        cv2.circle(
            imagem_visualizacao,
            (x, y),
            6,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            imagem_visualizacao,
            str(len(pontos)),
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        cv2.imshow(
            "Selecionar quatro cantos",
            imagem_visualizacao
        )


# ============================================================
# ORDENAR PONTOS
# ============================================================

def ordenar_pontos(pts):
    """
    Retorna os pontos na ordem:
    superior esquerdo,
    superior direito,
    inferior direito,
    inferior esquerdo.
    """

    pts = np.asarray(pts, dtype=np.float32)

    # Para documentos aproximadamente verticais,
    # separar em pares superiores e inferiores.
    indices = np.argsort(pts[:, 1])

    superiores = pts[indices[:2]]
    inferiores = pts[indices[2:]]

    superiores = superiores[
        np.argsort(superiores[:, 0])
    ]

    inferiores = inferiores[
        np.argsort(inferiores[:, 0])
    ]

    superior_esquerdo = superiores[0]
    superior_direito = superiores[1]

    inferior_esquerdo = inferiores[0]
    inferior_direito = inferiores[1]

    return np.array(
        [
            superior_esquerdo,
            superior_direito,
            inferior_direito,
            inferior_esquerdo
        ],
        dtype=np.float32
    )


# ============================================================
# TRANSFORMACAO PERSPECTIVA
# ============================================================

def transformar_imagem(imagem, pontos_origem):

    pontos_origem = ordenar_pontos(pontos_origem)

    pontos_destino = np.array(
        [
            [0, 0],
            [LARGURA_SAIDA - 1, 0],
            [LARGURA_SAIDA - 1, ALTURA_SAIDA - 1],
            [0, ALTURA_SAIDA - 1]
        ],
        dtype=np.float32
    )

    # Calcular matriz de homografia
    H = cv2.getPerspectiveTransform(
        pontos_origem,
        pontos_destino
    )

    # Aplicar transformacao
    imagem_corrigida = cv2.warpPerspective(
        imagem,
        H,
        (LARGURA_SAIDA, ALTURA_SAIDA)
    )

    return imagem_corrigida, H, pontos_origem


# ============================================================
# PROCESSAMENTO DAS IMAGENS
# ============================================================

def processar_imagem(caminho):
    global pontos
    global imagem_visualizacao
    global imagem_base

    nome = os.path.basename(caminho)
    nome_base = os.path.splitext(nome)[0]

    print("\n" + "=" * 65)
    print(f"Processando: {nome}")
    print("=" * 65)

    imagem = cv2.imread(caminho)

    if imagem is None:
        print("ERRO: nao foi possivel abrir a imagem.")
        return

    altura, largura = imagem.shape[:2]

    # Reduzir apenas para exibicao
    escala = min(
        1.0,
        LARGURA_VISUALIZACAO / largura
    )

    nova_largura = int(largura * escala)
    nova_altura = int(altura * escala)

    imagem_base = cv2.resize(
        imagem,
        (nova_largura, nova_altura),
        interpolation=cv2.INTER_AREA
    )

    pontos = []
    imagem_visualizacao = imagem_base.copy()

    nome_janela = "Selecionar quatro cantos"

    cv2.namedWindow(
        nome_janela,
        cv2.WINDOW_NORMAL
    )

    cv2.setMouseCallback(
        nome_janela,
        selecionar_ponto
    )

    print("\nSelecione os quatro cantos do documento.")
    print("A ordem dos cliques e livre.")
    print("ENTER: confirmar")
    print("R: reiniciar selecao")
    print("ESC: pular imagem")

    while True:

        cv2.imshow(
            nome_janela,
            imagem_visualizacao
        )

        tecla = cv2.waitKey(20) & 0xFF

        if tecla in (13, 10):

            if len(pontos) == 4:
                break

            print(
                "Selecione exatamente quatro pontos."
            )

        elif tecla == ord("r"):

            pontos = []
            imagem_visualizacao = imagem_base.copy()

            print("Selecao reiniciada.")

        elif tecla == 27:

            cv2.destroyWindow(nome_janela)
            print("Imagem ignorada.")
            return

    cv2.destroyWindow(nome_janela)

    # Converter coordenadas para resolucao original
    pontos_originais = (
        np.asarray(pontos, dtype=np.float32) / escala
    )

    # Aplicar transformacao
    corrigida, H, pontos_ordenados = transformar_imagem(
        imagem,
        pontos_originais
    )

    # ========================================================
    # SALVAR IMAGEM COM OS CANTOS
    # ========================================================

    imagem_pontos = imagem.copy()

    pontos_int = np.round(
        pontos_ordenados
    ).astype(int)

    cv2.polylines(
        imagem_pontos,
        [pontos_int],
        True,
        (0, 255, 0),
        5
    )

    for i, (x, y) in enumerate(pontos_int):

        cv2.circle(
            imagem_pontos,
            (x, y),
            12,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            imagem_pontos,
            str(i + 1),
            (x + 15, y - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3
        )

    caminho_pontos = os.path.join(
        PASTA_RESULTADOS,
        f"{nome_base}_pontos.jpg"
    )

    cv2.imwrite(
        caminho_pontos,
        imagem_pontos
    )

    # ========================================================
    # SALVAR IMAGEM CORRIGIDA
    # ========================================================

    caminho_corrigida = os.path.join(
        PASTA_RESULTADOS,
        f"{nome_base}_corrigida.jpg"
    )

    cv2.imwrite(
        caminho_corrigida,
        corrigida
    )

    # ========================================================
    # VERSAO EM ESCALA DE CINZA
    # ========================================================

    cinza = cv2.cvtColor(
        corrigida,
        cv2.COLOR_BGR2GRAY
    )

    caminho_cinza = os.path.join(
        PASTA_RESULTADOS,
        f"{nome_base}_cinza.jpg"
    )

    cv2.imwrite(
        caminho_cinza,
        cinza
    )

    # ========================================================
    # SALVAR MATRIZ DE HOMOGRAFIA
    # ========================================================

    caminho_matriz = os.path.join(
        PASTA_RESULTADOS,
        f"{nome_base}_homografia.txt"
    )

    np.savetxt(
        caminho_matriz,
        H,
        fmt="%.8f"
    )

    print("\nPontos selecionados (pixels):")
    print(pontos_ordenados)

    print("\nMatriz de homografia H:")
    print(H)

    print("\nArquivos gerados:")
    print(caminho_pontos)
    print(caminho_corrigida)
    print(caminho_cinza)
    print(caminho_matriz)

    print("\nImagem processada com sucesso.")


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main():

    print("=" * 65)
    print("TA03 - TRANSFORMACAO PERSPECTIVA")
    print("=" * 65)

    extensoes = ("*.jpg", "*.jpeg", "*.png")

    imagens = []

    for extensao in extensoes:

        imagens.extend(
            glob.glob(
                os.path.join(
                    PASTA_IMAGENS,
                    extensao
                )
            )
        )

    imagens = sorted(imagens)

    print(
        f"\nImagens encontradas: {len(imagens)}"
    )

    if not imagens:

        print(
            "Nenhuma imagem encontrada na pasta imagens/."
        )
        return

    for i, caminho in enumerate(imagens):

        print(
            f"\n[{i + 1}/{len(imagens)}]"
        )

        processar_imagem(caminho)

    print("\n" + "=" * 65)
    print("PROCESSAMENTO CONCLUIDO")
    print("=" * 65)

    print(
        f"\nResultados salvos em: {PASTA_RESULTADOS}/"
    )


if __name__ == "__main__":
    main()
