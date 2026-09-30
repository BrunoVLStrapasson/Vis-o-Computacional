import cv2
import numpy as np
import glob
import os


# ============================================================
# CONFIGURAÇÕES
# ============================================================

PASTA_IMAGENS = "imagens"
PASTA_RESULTADOS = "resultados"

# Número de CANTOS INTERNOS do tabuleiro.
# O padrão possui 10 x 7 quadrados,
# portanto possui 9 x 6 cantos internos.
COLUNAS = 9
LINHAS = 6

TAMANHO_TABULEIRO = (COLUNAS, LINHAS)

# Tamanho de cada quadrado.
#
# Nesta etapa utilizamos 1.0 unidade por quadrado.
# Caso o tamanho físico seja medido posteriormente,
# por exemplo 35 mm:
#
# TAMANHO_QUADRADO = 35.0
#
TAMANHO_QUADRADO = 1.0


# ============================================================
# PREPARAÇÃO DAS PASTAS
# ============================================================

os.makedirs(
    PASTA_RESULTADOS,
    exist_ok=True
)

PASTA_CANTOS = os.path.join(
    PASTA_RESULTADOS,
    "cantos"
)

os.makedirs(
    PASTA_CANTOS,
    exist_ok=True
)


# ============================================================
# CRITÉRIO DE REFINAMENTO SUBPIXEL
# ============================================================

criterio = (
    cv2.TERM_CRITERIA_EPS
    + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)


# ============================================================
# CRIAÇÃO DOS PONTOS 3D DO TABULEIRO
# ============================================================

# Como o tabuleiro é plano, todos os pontos possuem Z = 0.
#
# Exemplo:
#
# (0,0,0) (1,0,0) (2,0,0) ...
# (0,1,0) (1,1,0) (2,1,0) ...
# (0,2,0) (1,2,0) (2,2,0) ...
#
# Esses pontos representam as posições conhecidas
# dos cantos internos no sistema de coordenadas
# do tabuleiro.

pontos_objeto = np.zeros(
    (LINHAS * COLUNAS, 3),
    np.float32
)

pontos_objeto[:, :2] = (
    np.mgrid[
        0:COLUNAS,
        0:LINHAS
    ]
    .T
    .reshape(-1, 2)
)

pontos_objeto *= TAMANHO_QUADRADO


# ============================================================
# LISTAS PARA CALIBRAÇÃO
# ============================================================

# Pontos 3D conhecidos do tabuleiro.
objpoints = []

# Pontos 2D detectados nas imagens.
imgpoints = []

# Guarda somente as imagens onde o padrão
# foi detectado corretamente.
imagens_validas = []


# ============================================================
# LOCALIZAR AS IMAGENS
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
        f"Nenhuma imagem JPG encontrada "
        f"em '{PASTA_IMAGENS}'."
    )


print("=" * 65)
print("CALIBRAÇÃO DE CÂMERA")
print("=" * 65)

print(
    f"\nImagens encontradas: "
    f"{len(imagens)}"
)

print(
    f"Padrão procurado: "
    f"{COLUNAS} x {LINHAS} cantos internos"
)


tamanho_imagem = None


# ============================================================
# DETECÇÃO DOS CANTOS
# ============================================================

for indice, caminho in enumerate(
    imagens,
    start=1
):

    imagem = cv2.imread(caminho)

    nome = os.path.basename(caminho)


    print(
        f"\n[{indice}/{len(imagens)}] "
        f"Processando: {nome}"
    )


    if imagem is None:

        print(
            "    ERRO - não foi possível "
            "abrir a imagem."
        )

        continue


    # --------------------------------------------------------
    # Converter para tons de cinza
    # --------------------------------------------------------

    cinza = cv2.cvtColor(
        imagem,
        cv2.COLOR_BGR2GRAY
    )


    # OpenCV utiliza:
    #
    # (largura, altura)
    #
    tamanho_imagem = cinza.shape[::-1]


    # ========================================================
    # TENTATIVA 1
    #
    # Detector moderno do OpenCV.
    # ========================================================

    encontrou, cantos = (
        cv2.findChessboardCornersSB(
            cinza,
            TAMANHO_TABULEIRO,
            flags=(
                cv2.CALIB_CB_NORMALIZE_IMAGE
                | cv2.CALIB_CB_EXHAUSTIVE
                | cv2.CALIB_CB_ACCURACY
            )
        )
    )


    detector_utilizado = "SB"


    # ========================================================
    # TENTATIVA 2
    #
    # Se o detector moderno falhar,
    # utilizamos o detector clássico.
    # ========================================================

    if not encontrou:

        flags_classico = (
            cv2.CALIB_CB_ADAPTIVE_THRESH
            | cv2.CALIB_CB_NORMALIZE_IMAGE
        )


        encontrou, cantos = (
            cv2.findChessboardCorners(
                cinza,
                TAMANHO_TABULEIRO,
                flags_classico
            )
        )


        detector_utilizado = "Clássico"


        # ----------------------------------------------------
        # Refinamento subpixel
        # ----------------------------------------------------

        if encontrou:

            cantos = cv2.cornerSubPix(
                cinza,
                cantos,
                (11, 11),
                (-1, -1),
                criterio
            )


    # ========================================================
    # PADRÃO ENCONTRADO
    # ========================================================

    if encontrou:

        print(
            f"    OK - {len(cantos)} "
            f"cantos encontrados."
        )

        print(
            f"    Detector: "
            f"{detector_utilizado}"
        )


        # Pontos 3D conhecidos.
        objpoints.append(
            pontos_objeto.copy()
        )


        # Garante representação consistente:
        #
        # (N, 1, 2)
        #
        cantos = np.asarray(
            cantos,
            dtype=np.float32
        ).reshape(-1, 1, 2)


        # Pontos encontrados na fotografia.
        imgpoints.append(
            cantos
        )


        # Nome da imagem associada à pose.
        imagens_validas.append(
            nome
        )


        # ----------------------------------------------------
        # Desenhar os cantos detectados
        # ----------------------------------------------------

        visualizacao = imagem.copy()


        cv2.drawChessboardCorners(
            visualizacao,
            TAMANHO_TABULEIRO,
            cantos,
            encontrou
        )


        # ----------------------------------------------------
        # Salvar resultado
        # ----------------------------------------------------

        nome_base = os.path.splitext(
            nome
        )[0]


        caminho_saida = os.path.join(
            PASTA_CANTOS,
            f"{nome_base}_cantos.jpg"
        )


        cv2.imwrite(
            caminho_saida,
            visualizacao
        )


    # ========================================================
    # PADRÃO NÃO ENCONTRADO
    # ========================================================

    else:

        print(
            "    FALHA - padrão não encontrado."
        )


# ============================================================
# RESULTADO DA DETECÇÃO
# ============================================================

print("\n" + "=" * 65)
print("RESULTADO DA DETECÇÃO")
print("=" * 65)


print(
    f"\nImagens válidas: "
    f"{len(imagens_validas)}/"
    f"{len(imagens)}"
)


print("\nImagens utilizadas:")


for nome in imagens_validas:

    print(
        f" - {nome}"
    )


# ============================================================
# VERIFICAR QUANTIDADE DE IMAGENS
# ============================================================

if len(imagens_validas) < 5:

    raise RuntimeError(
        "\nPoucas imagens tiveram o padrão detectado.\n"
        "São necessárias pelo menos 5 para "
        "continuar neste experimento."
    )


# ============================================================
# CALIBRAÇÃO DA CÂMERA
# ============================================================

print(
    "\nExecutando "
    "cv2.calibrateCamera()..."
)


(
    rms,
    matriz_camera,
    distorcao,
    rvecs,
    tvecs

) = cv2.calibrateCamera(

    objpoints,
    imgpoints,
    tamanho_imagem,
    None,
    None
)


# ============================================================
# RESULTADOS DA CALIBRAÇÃO
# ============================================================

print("\n" + "=" * 65)
print("CALIBRAÇÃO CONCLUÍDA")
print("=" * 65)


print(
    "\nErro RMS retornado "
    "pelo OpenCV:"
)

print(rms)


print(
    "\nMatriz intrínseca K:"
)

print(matriz_camera)


print(
    "\nCoeficientes de distorção:"
)

print(distorcao)


# ============================================================
# PARÂMETROS INTRÍNSECOS
# ============================================================

fx = matriz_camera[0, 0]
fy = matriz_camera[1, 1]

cx = matriz_camera[0, 2]
cy = matriz_camera[1, 2]


print(
    "\nParâmetros principais:"
)

print(
    f"fx = {fx:.6f}"
)

print(
    f"fy = {fy:.6f}"
)

print(
    f"cx = {cx:.6f}"
)

print(
    f"cy = {cy:.6f}"
)


# ============================================================
# ERRO DE REPROJEÇÃO
# ============================================================

print("\n" + "=" * 65)
print("ERRO DE REPROJEÇÃO")
print("=" * 65)


erro_total = 0.0
rmse_total = 0.0

erros_por_imagem = []
rmse_por_imagem = []


for i in range(
    len(objpoints)
):


    # --------------------------------------------------------
    # Reprojetar os pontos 3D para a imagem
    # --------------------------------------------------------

    pontos_projetados, _ = (
        cv2.projectPoints(
            objpoints[i],
            rvecs[i],
            tvecs[i],
            matriz_camera,
            distorcao
        )
    )


    # --------------------------------------------------------
    # Padronizar para (N, 2)
    # --------------------------------------------------------

    pontos_detectados = (
        imgpoints[i]
        .reshape(-1, 2)
    )


    pontos_projetados = (
        pontos_projetados
        .reshape(-1, 2)
    )


    # ========================================================
    # MÉTRICA 1
    #
    # Norma L2 dividida pelo número de pontos.
    # ========================================================

    erro = cv2.norm(
        pontos_detectados,
        pontos_projetados,
        cv2.NORM_L2
    )

    erro /= len(
        pontos_projetados
    )


    erro_total += erro

    erros_por_imagem.append(
        erro
    )


    # ========================================================
    # MÉTRICA 2
    #
    # RMSE geométrico em pixels.
    # ========================================================

    diferenca = (
        pontos_detectados
        - pontos_projetados
    )


    distancias_quadradas = np.sum(
        diferenca ** 2,
        axis=1
    )


    rmse = np.sqrt(
        np.mean(
            distancias_quadradas
        )
    )


    rmse_total += rmse

    rmse_por_imagem.append(
        rmse
    )


    # --------------------------------------------------------
    # Mostrar resultados
    # --------------------------------------------------------

    print(
        f"\n{imagens_validas[i]}"
    )

    print(
        f"    Erro L2/N: "
        f"{erro:.6f} pixels"
    )

    print(
        f"    RMSE: "
        f"{rmse:.6f} pixels"
    )


# ============================================================
# MÉDIAS
# ============================================================

erro_medio = (
    erro_total
    / len(objpoints)
)


rmse_medio = (
    rmse_total
    / len(objpoints)
)


print("\n" + "-" * 65)


print(
    f"Erro médio L2/N: "
    f"{erro_medio:.6f} pixels"
)


print(
    f"RMSE médio de reprojeção: "
    f"{rmse_medio:.6f} pixels"
)


# ============================================================
# SALVAR PARÂMETROS
# ============================================================

caminho_parametros = os.path.join(
    PASTA_RESULTADOS,
    "calibracao.npz"
)


np.savez(
    caminho_parametros,

    matriz_camera=matriz_camera,
    distorcao=distorcao,

    rvecs=np.array(rvecs),
    tvecs=np.array(tvecs),

    imagens_validas=np.array(
        imagens_validas
    ),

    objpoints=np.array(
        objpoints
    ),

    imgpoints=np.array(
        imgpoints
    ),

    colunas=COLUNAS,
    linhas=LINHAS,

    tamanho_quadrado=TAMANHO_QUADRADO,

    tamanho_imagem=np.array(
        tamanho_imagem
    ),

    rms=rms,
    erro_medio=erro_medio,
    rmse_medio=rmse_medio,

    erros_por_imagem=np.array(
        erros_por_imagem
    ),

    rmse_por_imagem=np.array(
        rmse_por_imagem
    )
)


# ============================================================
# RESULTADO FINAL
# ============================================================

print("\n" + "=" * 65)
print("ARQUIVOS GERADOS")
print("=" * 65)


print(
    f"\nParâmetros salvos em:\n"
    f"{caminho_parametros}"
)


print(
    f"\nImagens com os cantos "
    f"detectados em:\n"
    f"{PASTA_CANTOS}/"
)


print(
    "\nO arquivo calibracao.npz "
    "contém:"
)

print(
    " - matriz_camera"
)

print(
    " - distorcao"
)

print(
    " - rvecs"
)

print(
    " - tvecs"
)

print(
    " - imagens_validas"
)

print(
    " - tamanho do tabuleiro"
)

print(
    " - tamanho da imagem"
)

print(
    " - métricas de reprojeção"
)


print(
    "\nProcessamento concluído."
)
