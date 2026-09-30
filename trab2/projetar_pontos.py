import cv2
import numpy as np
import os


# ============================================================
# CONFIGURAÇÕES
# ============================================================

PASTA_IMAGENS = "imagens"
PASTA_RESULTADOS = "resultados"

PASTA_PROJECOES = os.path.join(
    PASTA_RESULTADOS,
    "projecoes"
)

ARQUIVO_CALIBRACAO = os.path.join(
    PASTA_RESULTADOS,
    "calibracao.npz"
)


os.makedirs(
    PASTA_PROJECOES,
    exist_ok=True
)


# ============================================================
# CARREGAR CALIBRAÇÃO
# ============================================================

dados = np.load(
    ARQUIVO_CALIBRACAO
)


matriz_camera = dados[
    "matriz_camera"
]

distorcao = dados[
    "distorcao"
]

rvecs = dados[
    "rvecs"
]

tvecs = dados[
    "tvecs"
]

imagens_validas = dados[
    "imagens_validas"
]

tamanho_quadrado = float(
    dados["tamanho_quadrado"]
)


print("=" * 65)
print("PROJEÇÃO DE PONTOS 3D -> IMAGEM 2D")
print("=" * 65)


print(
    f"\nImagens disponíveis: "
    f"{len(imagens_validas)}"
)


# ============================================================
# DEFINIÇÃO DOS PONTOS 3D
# ============================================================

# O sistema de coordenadas está preso ao tabuleiro.
#
# Origem:
# primeiro canto interno do tabuleiro.
#
# X -> horizontal
# Y -> vertical
# Z -> perpendicular ao plano
#
# Como TAMANHO_QUADRADO = 1.0:
#
# cada unidade representa atualmente
# o lado de um quadrado.


s = tamanho_quadrado


pontos_3d = np.array(
    [
        [0, 0, 0],       # Origem

        [3*s, 0, 0],     # eixo X

        [0, 3*s, 0],     # eixo Y

        [0, 0, -3*s]     # eixo Z
    ],
    dtype=np.float32
)


nomes_pontos = [
    "O",
    "X",
    "Y",
    "Z"
]


# ============================================================
# CORES BGR
# ============================================================

COR_ORIGEM = (
    255,
    255,
    255
)

COR_X = (
    0,
    0,
    255
)

COR_Y = (
    0,
    255,
    0
)

COR_Z = (
    255,
    0,
    0
)


# ============================================================
# PROCESSAR CADA IMAGEM
# ============================================================

for i, nome in enumerate(
    imagens_validas
):


    nome = str(nome)


    print("\n" + "-" * 65)

    print(
        f"Imagem: {nome}"
    )


    caminho = os.path.join(
        PASTA_IMAGENS,
        nome
    )


    imagem = cv2.imread(
        caminho
    )


    if imagem is None:

        print(
            "ERRO: imagem não encontrada."
        )

        continue


    # ========================================================
    # PROJETAR OS PONTOS 3D
    # ========================================================

    pontos_2d, _ = cv2.projectPoints(
        pontos_3d,
        rvecs[i],
        tvecs[i],
        matriz_camera,
        distorcao
    )


    pontos_2d = pontos_2d.reshape(
        -1,
        2
    )


    # ========================================================
    # MOSTRAR COORDENADAS
    # ========================================================

    print(
        "\nPonto 3D -> coordenada 2D"
    )


    for j in range(
        len(pontos_3d)
    ):

        X, Y, Z = pontos_3d[j]

        u, v = pontos_2d[j]


        print(
            f"{nomes_pontos[j]} "
            f"({X:.2f}, "
            f"{Y:.2f}, "
            f"{Z:.2f})"
            f" -> "
            f"({u:.2f}, "
            f"{v:.2f})"
        )


    # ========================================================
    # CONVERTER PARA INTEIROS PARA DESENHO
    # ========================================================

    pontos_int = np.round(
        pontos_2d
    ).astype(int)


    origem = tuple(
        pontos_int[0]
    )

    ponto_x = tuple(
        pontos_int[1]
    )

    ponto_y = tuple(
        pontos_int[2]
    )

    ponto_z = tuple(
        pontos_int[3]
    )


    # ========================================================
    # DESENHAR EIXOS
    # ========================================================

    # X = vermelho

    cv2.line(
        imagem,
        origem,
        ponto_x,
        COR_X,
        8
    )


    # Y = verde

    cv2.line(
        imagem,
        origem,
        ponto_y,
        COR_Y,
        8
    )


    # Z = azul

    cv2.line(
        imagem,
        origem,
        ponto_z,
        COR_Z,
        8
    )


    # ========================================================
    # DESENHAR PONTOS
    # ========================================================

    cv2.circle(
        imagem,
        origem,
        14,
        COR_ORIGEM,
        -1
    )


    cv2.circle(
        imagem,
        ponto_x,
        14,
        COR_X,
        -1
    )


    cv2.circle(
        imagem,
        ponto_y,
        14,
        COR_Y,
        -1
    )


    cv2.circle(
        imagem,
        ponto_z,
        14,
        COR_Z,
        -1
    )


    # ========================================================
    # ESCREVER NOMES
    # ========================================================

    pontos_nomeados = [
        (
            "O",
            origem,
            COR_ORIGEM
        ),

        (
            "X",
            ponto_x,
            COR_X
        ),

        (
            "Y",
            ponto_y,
            COR_Y
        ),

        (
            "Z",
            ponto_z,
            COR_Z
        )
    ]


    for texto, ponto, cor in pontos_nomeados:

        px, py = ponto


        cv2.putText(
            imagem,
            texto,
            (
                px + 15,
                py - 15
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            cor,
            3,
            cv2.LINE_AA
        )


    # ========================================================
    # SALVAR
    # ========================================================

    nome_base = os.path.splitext(
        nome
    )[0]


    caminho_saida = os.path.join(
        PASTA_PROJECOES,
        f"{nome_base}_projecao.jpg"
    )


    cv2.imwrite(
        caminho_saida,
        imagem
    )


    print(
        f"\nResultado salvo em: "
        f"{caminho_saida}"
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 65)
print("PROCESSAMENTO CONCLUÍDO")
print("=" * 65)


print(
    "\nAs imagens contendo os "
    "eixos 3D projetados foram salvas em:"
)

print(
    PASTA_PROJECOES
)