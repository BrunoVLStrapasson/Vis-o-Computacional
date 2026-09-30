import cv2
import numpy as np
import glob
import os


# ============================================================
# CONFIGURAÇÕES
# ============================================================

PASTA_IMAGENS = "imagens"
PASTA_RESULTADOS = "resultados"
PASTA_CORRIGIDAS = os.path.join(
    PASTA_RESULTADOS,
    "corrigidas"
)

ARQUIVO_CALIBRACAO = os.path.join(
    PASTA_RESULTADOS,
    "calibracao.npz"
)


# ============================================================
# PREPARAÇÃO
# ============================================================

os.makedirs(
    PASTA_CORRIGIDAS,
    exist_ok=True
)


# ============================================================
# CARREGAR CALIBRAÇÃO
# ============================================================

if not os.path.exists(ARQUIVO_CALIBRACAO):
    raise RuntimeError(
        f"Arquivo de calibração não encontrado: "
        f"{ARQUIVO_CALIBRACAO}"
    )


dados = np.load(ARQUIVO_CALIBRACAO)

matriz_camera = dados["matriz_camera"]
distorcao = dados["distorcao"]


print("=" * 65)
print("CORREÇÃO DE DISTORÇÃO")
print("=" * 65)


print("\nMatriz intrínseca K:")
print(matriz_camera)


print("\nCoeficientes de distorção:")
print(distorcao)


# ============================================================
# LOCALIZAR IMAGENS
# ============================================================

imagens = sorted(
    glob.glob(
        os.path.join(
            PASTA_IMAGENS,
            "*.jpg"
        )
    )
)


if len(imagens) == 0:
    raise RuntimeError(
        f"Nenhuma imagem encontrada em "
        f"'{PASTA_IMAGENS}'."
    )


print(
    f"\nImagens encontradas: "
    f"{len(imagens)}"
)


# ============================================================
# CORRIGIR CADA IMAGEM
# ============================================================

for indice, caminho in enumerate(
    imagens,
    start=1
):

    nome = os.path.basename(caminho)

    print(
        f"\n[{indice}/{len(imagens)}] "
        f"Corrigindo: {nome}"
    )


    imagem = cv2.imread(caminho)


    if imagem is None:

        print(
            "    ERRO - não foi possível "
            "abrir a imagem."
        )

        continue


    altura, largura = imagem.shape[:2]


    # ========================================================
    # CALCULAR NOVA MATRIZ DA CÂMERA
    # ========================================================

    nova_matriz, roi = (
        cv2.getOptimalNewCameraMatrix(
            matriz_camera,
            distorcao,
            (largura, altura),
            1,
            (largura, altura)
        )
    )


    # ========================================================
    # REMOVER DISTORÇÃO
    # ========================================================

    corrigida = cv2.undistort(
        imagem,
        matriz_camera,
        distorcao,
        None,
        nova_matriz
    )


    # ========================================================
    # RECORTAR REGIÃO VÁLIDA
    # ========================================================

    x, y, w, h = roi


    if w > 0 and h > 0:

        corrigida_recortada = corrigida[
            y:y+h,
            x:x+w
        ]

    else:

        corrigida_recortada = corrigida


    # ========================================================
    # SALVAR IMAGEM CORRIGIDA COMPLETA
    # ========================================================

    nome_base = os.path.splitext(nome)[0]


    caminho_completa = os.path.join(
        PASTA_CORRIGIDAS,
        f"{nome_base}_corrigida.jpg"
    )


    cv2.imwrite(
        caminho_completa,
        corrigida
    )


    # ========================================================
    # SALVAR VERSÃO RECORTADA
    # ========================================================

    caminho_recortada = os.path.join(
        PASTA_CORRIGIDAS,
        f"{nome_base}_corrigida_recortada.jpg"
    )


    cv2.imwrite(
        caminho_recortada,
        corrigida_recortada
    )


    print(
        "    OK"
    )

    print(
        f"    ROI: "
        f"x={x}, y={y}, "
        f"w={w}, h={h}"
    )


print("\n" + "=" * 65)
print("PROCESSAMENTO CONCLUÍDO")
print("=" * 65)


print(
    f"\nImagens corrigidas salvas em:\n"
    f"{PASTA_CORRIGIDAS}/"
)