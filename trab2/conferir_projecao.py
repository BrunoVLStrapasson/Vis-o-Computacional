import cv2
import numpy as np
import os


# ============================================================
# CONFIGURAÇÕES
# ============================================================

PASTA_IMAGENS = "imagens"
PASTA_RESULTADOS = "resultados"

PASTA_CONFERENCIA = os.path.join(
    PASTA_RESULTADOS,
    "conferencia_projecao"
)

ARQUIVO_CALIBRACAO = os.path.join(
    PASTA_RESULTADOS,
    "calibracao.npz"
)

os.makedirs(
    PASTA_CONFERENCIA,
    exist_ok=True
)


# ============================================================
# CARREGAR DADOS
# ============================================================

if not os.path.exists(ARQUIVO_CALIBRACAO):
    raise RuntimeError(
        f"Arquivo não encontrado: {ARQUIVO_CALIBRACAO}"
    )


dados = np.load(
    ARQUIVO_CALIBRACAO
)


# Verificar se os dados necessários existem
campos_necessarios = [
    "matriz_camera",
    "distorcao",
    "rvecs",
    "tvecs",
    "imagens_validas",
    "objpoints",
    "imgpoints"
]


for campo in campos_necessarios:

    if campo not in dados.files:

        raise RuntimeError(
            f"O campo '{campo}' não existe em "
            f"{ARQUIVO_CALIBRACAO}.\n"
            f"Execute calibracao.py novamente "
            f"salvando objpoints e imgpoints."
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

objpoints = dados[
    "objpoints"
]

imgpoints = dados[
    "imgpoints"
]


# ============================================================
# INÍCIO
# ============================================================

print("=" * 70)
print("CONFERÊNCIA DA PROJEÇÃO 3D -> 2D")
print("=" * 70)

print(
    f"\nImagens válidas: "
    f"{len(imagens_validas)}"
)

print(
    f"Pontos por imagem: "
    f"{objpoints.shape[1]}"
)


# ============================================================
# ERROS GLOBAIS
# ============================================================

todos_erros = []

rmse_imagens = []


# ============================================================
# PROCESSAR CADA IMAGEM
# ============================================================

for i, nome in enumerate(
    imagens_validas
):

    nome = str(nome)

    print("\n" + "-" * 70)

    print(
        f"Imagem: {nome}"
    )


    # --------------------------------------------------------
    # Abrir fotografia
    # --------------------------------------------------------

    caminho_imagem = os.path.join(
        PASTA_IMAGENS,
        nome
    )


    imagem = cv2.imread(
        caminho_imagem
    )


    if imagem is None:

        print(
            "ERRO: não foi possível "
            "abrir a imagem."
        )

        continue


    # ========================================================
    # PONTOS 3D
    # ========================================================

    pontos_3d = np.asarray(
        objpoints[i],
        dtype=np.float32
    ).reshape(-1, 3)


    # ========================================================
    # PONTOS 2D REALMENTE DETECTADOS
    # ========================================================

    detectados = np.asarray(
        imgpoints[i],
        dtype=np.float32
    ).reshape(-1, 2)


    # ========================================================
    # PROJETAR OS MESMOS PONTOS 3D
    # ========================================================

    projetados, _ = cv2.projectPoints(
        pontos_3d,
        rvecs[i],
        tvecs[i],
        matriz_camera,
        distorcao
    )


    projetados = projetados.reshape(
        -1,
        2
    )


    # ========================================================
    # CALCULAR ERRO DE CADA PONTO
    # ========================================================

    diferencas = (
        detectados
        - projetados
    )


    erros = np.linalg.norm(
        diferencas,
        axis=1
    )


    todos_erros.extend(
        erros.tolist()
    )


    # ========================================================
    # MÉTRICAS DA IMAGEM
    # ========================================================

    erro_medio = np.mean(
        erros
    )


    erro_minimo = np.min(
        erros
    )


    erro_maximo = np.max(
        erros
    )


    rmse = np.sqrt(
        np.mean(
            np.sum(
                diferencas ** 2,
                axis=1
            )
        )
    )


    rmse_imagens.append(
        rmse
    )


    print(
        f"\nErro médio: "
        f"{erro_medio:.4f} px"
    )

    print(
        f"Erro mínimo: "
        f"{erro_minimo:.4f} px"
    )

    print(
        f"Erro máximo: "
        f"{erro_maximo:.4f} px"
    )

    print(
        f"RMSE: "
        f"{rmse:.4f} px"
    )


    # ========================================================
    # MOSTRAR ALGUNS PONTOS PARA CONFERÊNCIA
    # ========================================================

    # Índices escolhidos em diferentes regiões
    # do tabuleiro.
    indices_exemplo = [
        0,
        3,
        8,
        27,
        45,
        53
    ]


    print(
        "\nExemplos:"
    )

    print(
        "3D                  "
        "Detectado 2D            "
        "Projetado 2D            "
        "Erro"
    )


    for indice in indices_exemplo:

        if indice >= len(
            pontos_3d
        ):
            continue


        X, Y, Z = pontos_3d[
            indice
        ]

        ud, vd = detectados[
            indice
        ]

        up, vp = projetados[
            indice
        ]

        erro = erros[
            indice
        ]


        print(
            f"({X:4.1f},{Y:4.1f},{Z:4.1f})   "
            f"({ud:8.2f},{vd:8.2f})   "
            f"({up:8.2f},{vp:8.2f})   "
            f"{erro:7.3f} px"
        )


    # ========================================================
    # VISUALIZAÇÃO
    # ========================================================

    visualizacao = imagem.copy()


    # --------------------------------------------------------
    # Desenhar TODOS os pontos projetados
    #
    # Vermelho = projetado
    # --------------------------------------------------------

    for ponto in projetados:

        u, v = np.round(
            ponto
        ).astype(int)


        cv2.circle(
            visualizacao,
            (u, v),
            8,
            (0, 0, 255),
            -1
        )


    # --------------------------------------------------------
    # Desenhar TODOS os pontos detectados
    #
    # Verde = detectado
    # --------------------------------------------------------

    for ponto in detectados:

        u, v = np.round(
            ponto
        ).astype(int)


        cv2.circle(
            visualizacao,
            (u, v),
            5,
            (0, 255, 0),
            -1
        )


    # --------------------------------------------------------
    # Linhas entre detectado e projetado
    # --------------------------------------------------------

    for detectado, projetado in zip(
        detectados,
        projetados
    ):

        ud, vd = np.round(
            detectado
        ).astype(int)

        up, vp = np.round(
            projetado
        ).astype(int)


        cv2.line(
            visualizacao,
            (ud, vd),
            (up, vp),
            (255, 255, 255),
            2
        )


    # ========================================================
    # LEGENDA
    # ========================================================

    cv2.putText(
        visualizacao,
        "VERDE: detectado",
        (50, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (0, 255, 0),
        3,
        cv2.LINE_AA
    )


    cv2.putText(
        visualizacao,
        "VERMELHO: projetado",
        (50, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (0, 0, 255),
        3,
        cv2.LINE_AA
    )


    cv2.putText(
        visualizacao,
        f"RMSE: {rmse:.3f} px",
        (50, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (255, 255, 255),
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
        PASTA_CONFERENCIA,
        f"{nome_base}_conferencia.jpg"
    )


    cv2.imwrite(
        caminho_saida,
        visualizacao
    )


    print(
        f"\nImagem salva em: "
        f"{caminho_saida}"
    )


# ============================================================
# RESULTADO GLOBAL
# ============================================================

todos_erros = np.asarray(
    todos_erros
)


rmse_imagens = np.asarray(
    rmse_imagens
)


print("\n" + "=" * 70)
print("RESULTADO GLOBAL")
print("=" * 70)


if len(todos_erros) > 0:

    print(
        f"\nNúmero total de pontos: "
        f"{len(todos_erros)}"
    )


    print(
        f"Erro médio entre pontos "
        f"detectados e projetados: "
        f"{np.mean(todos_erros):.4f} px"
    )


    print(
        f"Erro mínimo: "
        f"{np.min(todos_erros):.4f} px"
    )


    print(
        f"Erro máximo: "
        f"{np.max(todos_erros):.4f} px"
    )


    print(
        f"RMSE médio entre imagens: "
        f"{np.mean(rmse_imagens):.4f} px"
    )


print(
    f"\nResultados visuais em:\n"
    f"{PASTA_CONFERENCIA}/"
)


print(
    "\nProcessamento concluído."
)