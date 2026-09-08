from sklearn.cluster import KMeans
import numpy as np
import os
import cv2

# CONFIGURAÇAO
TAMANHO_IMAGEM = 512
TAMANHO_JANELA = 64

NUM_ESCALAS = 3
NUM_FILTROS = 8

PASTA_IMAGENS = "imagens"
PASTA_RESULTADOS = "resultados"

# PRÉ-PROCESSAMENTO

def carregar_imagem(caminho):
    """   Carrega uma imagem JPG, converte para escala de cinza  e verifica se possui tamanho 512x512. """

    imagem = cv2.imread(caminho)

    if imagem is None:
        raise ValueError(
            f"Não foi possível abrir: {caminho}"
        )

    cinza = cv2.cvtColor(
        imagem,
        cv2.COLOR_BGR2GRAY
    )

    altura, largura = cinza.shape

    if altura != TAMANHO_IMAGEM or largura != TAMANHO_IMAGEM:
        raise ValueError(
            f"A imagem {caminho} possui tamanho "
            f"{largura}x{altura}. "
            f"Esperado: 512x512."
        )

    return cinza


# ESCALAS

def gerar_escalas(imagem):
    """ Gera as três escalas: 512x512,256x256 E 128x128"""

    escala1 = imagem.copy()

    suavizada2 = cv2.GaussianBlur(
        escala1,
        (5, 5),
        1.0
    )

    escala2 = cv2.resize(
        suavizada2,
        (256, 256),
        interpolation=cv2.INTER_AREA
    )

    suavizada3 = cv2.GaussianBlur(
        escala2,
        (5, 5),
        1.0
    )

    escala3 = cv2.resize(
        suavizada3,
        (128, 128),
        interpolation=cv2.INTER_AREA
    )

    return [
        escala1,
        escala2,
        escala3
    ]


# FILTROS DE GABOR
# ============================================================

def criar_gabor(theta):
    """
    Cria um filtro de Gabor.

    theta:
        orientação em graus.
    """

    theta_rad = np.deg2rad(theta)

    kernel = cv2.getGaborKernel(
        (21, 21),
        sigma=4.0,
        theta=theta_rad,
        lambd=10.0,
        gamma=0.5,
        psi=0,
        ktype=cv2.CV_32F
    )

    kernel -= kernel.mean()

    norma = np.sum(np.abs(kernel))

    if norma != 0:
        kernel /= norma

    return kernel


# ============================================================
# FILTROS ISOTRÓPICOS / CIRCULARES
# ============================================================

def criar_laplaciano():
    """
    Filtro Laplaciano.
    """

    kernel = np.array([
        [0,  1, 0],
        [1, -4, 1],
        [0,  1, 0]
    ], dtype=np.float32)

    return kernel


def criar_log():
    """
    Cria um filtro Laplaciano da Gaussiana (LoG).
    """

    tamanho = 15
    sigma = 2.0

    gauss = cv2.getGaussianKernel(
        tamanho,
        sigma,
        ktype=cv2.CV_32F
    )

    gauss2d = gauss @ gauss.T

    laplaciano = cv2.Laplacian(
        gauss2d,
        cv2.CV_32F
    )

    laplaciano -= laplaciano.mean()

    return laplaciano


def criar_dog():
    """
    Diferença de Gaussianas.
    """

    tamanho = 15

    g1 = cv2.getGaussianKernel(
        tamanho,
        1.0
    )

    g2 = cv2.getGaussianKernel(
        tamanho,
        3.0
    )

    g1 = g1 @ g1.T
    g2 = g2 @ g2.T

    dog = g1 - g2

    dog -= dog.mean()

    return dog.astype(np.float32)


def criar_filtro_circular():
    """
    Cria um filtro circular simples
    """

    tamanho = 15
    centro = tamanho // 2

    kernel = np.zeros(
        (tamanho, tamanho),
        dtype=np.float32
    )

    for y in range(tamanho):

        for x in range(tamanho):

            distancia = np.sqrt(
                (x - centro) ** 2 +
                (y - centro) ** 2
            )

            if distancia <= 3:

                kernel[y, x] = 1

            elif distancia <= 6:

                kernel[y, x] = -0.25

    kernel -= kernel.mean()

    norma = np.sum(
        np.abs(kernel)
    )

    if norma != 0:
        kernel /= norma

    return kernel


# ============================================================
# BANCO DE FILTROS
# ============================================================

def criar_filtros():

    filtros = []

    filtros.append(
        ("Gabor_0", criar_gabor(0))
    )

    filtros.append(
        ("Gabor_45", criar_gabor(45))
    )

    filtros.append(
        ("Gabor_90", criar_gabor(90))
    )

    filtros.append(
        ("Gabor_135", criar_gabor(135))
    )

    filtros.append(
        ("Laplaciano", criar_laplaciano())
    )

    filtros.append(
        ("LoG", criar_log())
    )

    filtros.append(
        ("DoG", criar_dog())
    )

    filtros.append(
        ("Circular", criar_filtro_circular())
    )

    return filtros


# ============================================================
# APLICAÇÃO DOS FILTROS
# ============================================================

def aplicar_filtro(imagem, kernel):
    """
    Aplica o filtro e retorna o módulo da resposta.
    """

    resposta = cv2.filter2D(
        imagem,
        cv2.CV_32F,
        kernel
    )

    resposta = np.abs(
        resposta
    )

    return resposta


# ============================================================
# SALVAR IMAGEM NORMALIZADA
# ============================================================

def salvar_imagem_normalizada(caminho, imagem):
    """
    Normaliza uma resposta de filtro para 0-255
    apenas para visualização.
    """

    imagem_normalizada = cv2.normalize(
        imagem,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    imagem_normalizada = (
        imagem_normalizada.astype(
            np.uint8
        )
    )

    cv2.imwrite(
        caminho,
        imagem_normalizada
    )


# ============================================================
# SALVAR RESULTADOS VISUAIS DE UMA IMAGEM
# ============================================================

def salvar_resultados_imagem(
    imagem,
    escalas,
    filtros,
    nome_imagem
):
    """
    Salva os resultados de uma imagem.

    Cada imagem possui sua própria pasta.
    """

    nome_base = os.path.splitext(
        nome_imagem
    )[0]

    pasta_imagem = os.path.join(
        PASTA_RESULTADOS,
        nome_base
    )

    os.makedirs(
        pasta_imagem,
        exist_ok=True
    )

    # --------------------------------------------------------
    # ORIGINAL
    # --------------------------------------------------------

    cv2.imwrite(
        os.path.join(
            pasta_imagem,
            "01_original.png"
        ),
        imagem
    )

    # --------------------------------------------------------
    # ESCALAS
    # --------------------------------------------------------

    for i, escala in enumerate(escalas):

        numero_escala = i + 1

        altura, largura = (
            escala.shape
        )

        nome = (
            f"escala_{numero_escala}_"
            f"{largura}x{altura}.png"
        )

        caminho = os.path.join(
            pasta_imagem,
            nome
        )

        cv2.imwrite(
            caminho,
            escala
        )

    # --------------------------------------------------------
    # FILTROS
    # --------------------------------------------------------

    pasta_filtros = os.path.join(
        pasta_imagem,
        "filtros"
    )

    os.makedirs(
        pasta_filtros,
        exist_ok=True
    )

    for numero_escala, escala in enumerate(
        escalas
    ):

        indice_escala = (
            numero_escala + 1
        )

        pasta_escala = os.path.join(
            pasta_filtros,
            f"escala_{indice_escala}"
        )

        os.makedirs(
            pasta_escala,
            exist_ok=True
        )

        for nome_filtro, kernel in filtros:

            resposta = aplicar_filtro(
                escala,
                kernel
            )

            caminho = os.path.join(
                pasta_escala,
                f"{nome_filtro}.png"
            )

            salvar_imagem_normalizada(
                caminho,
                resposta
            )


# ============================================================
# DIVISÃO EM JANELAS
# ============================================================

def dividir_janelas(
    imagem,
    tamanho_janela
):
    """
    Divide a imagem em regiões quadradas.
    """

    altura, largura = imagem.shape

    janelas = []

    for y in range(
        0,
        altura,
        tamanho_janela
    ):

        for x in range(
            0,
            largura,
            tamanho_janela
        ):

            janela = imagem[
                y:y + tamanho_janela,
                x:x + tamanho_janela
            ]

            if janela.shape == (
                tamanho_janela,
                tamanho_janela
            ):

                janelas.append(
                    (x, y, janela)
                )

    return janelas


# ============================================================
# EXTRAÇÃO DO VETOR DE 24 CARACTERÍSTICAS
# ============================================================

def extrair_vetor_24(
    imagem,
    filtros
):
    """
    Cria 64 vetores de 24 características.

    8 filtros × 3 escalas = 24 características.

    Retorno:

        matriz 64 × 24
    """

    escalas = gerar_escalas(
        imagem
    )

    numero_regioes = 8 * 8

    matriz = np.zeros(
        (
            numero_regioes,
            NUM_FILTROS * NUM_ESCALAS
        ),
        dtype=np.float32
    )

    coluna = 0

    # --------------------------------------------------------
    # ESCALAS
    # --------------------------------------------------------

    for numero_escala, escala in enumerate(
        escalas
    ):

        tamanho_janela = (
            TAMANHO_JANELA //
            (2 ** numero_escala)
        )

        # ----------------------------------------------------
        # FILTROS
        # ----------------------------------------------------

        for nome_filtro, kernel in filtros:

            resposta = aplicar_filtro(
                escala,
                kernel
            )

            janelas = dividir_janelas(
                resposta,
                tamanho_janela
            )

            valores = []

            for _, _, janela in janelas:

                media = np.mean(
                    janela
                )

                valores.append(
                    media
                )

            matriz[
                :,
                coluna
            ] = valores

            coluna += 1

    return matriz


# ============================================================
# PROCESSAR TODAS AS IMAGENS
# ============================================================

def processar_todas_imagens(
    filtros
):
    """
    Processa todas as imagens da pasta imagens/.

    Retorna:

        dados = matriz 2048 × 24

        nomes = nome de cada imagem/região
    """

    extensoes = (
        ".jpg",
        ".jpeg",
        ".JPG",
        ".JPEG"
    )

    arquivos = [
        arquivo
        for arquivo in os.listdir(
            PASTA_IMAGENS
        )
        if arquivo.endswith(
            extensoes
        )
    ]

    arquivos.sort(
        key=lambda nome: int(
            os.path.splitext(nome)[0].replace("img", "")
        )
    )

    if len(arquivos) == 0:

        raise ValueError(
            f"Nenhuma imagem encontrada "
            f"na pasta '{PASTA_IMAGENS}'."
        )

    print(
        f"\nForam encontradas "
        f"{len(arquivos)} imagens."
    )

    todos_vetores = []

    identificadores = []

    # ========================================================
    # PROCESSAR CADA IMAGEM
    # ========================================================

    for numero, nome_arquivo in enumerate(
        arquivos,
        start=1
    ):

        caminho = os.path.join(
            PASTA_IMAGENS,
            nome_arquivo
        )

        print(
            f"\n[{numero}/{len(arquivos)}] "
            f"Processando: {nome_arquivo}"
        )

        # ----------------------------------------------------
        # CARREGAR
        # ----------------------------------------------------

        imagem = carregar_imagem(
            caminho
        )

        # ----------------------------------------------------
        # ESCALAS
        # ----------------------------------------------------

        escalas = gerar_escalas(
            imagem
        )

        # ----------------------------------------------------
        # RESULTADOS VISUAIS
        # ----------------------------------------------------

        salvar_resultados_imagem(
            imagem,
            escalas,
            filtros,
            nome_arquivo
        )

        # ----------------------------------------------------
        # CARACTERÍSTICAS
        # ----------------------------------------------------

        vetores = extrair_vetor_24(
            imagem,
            filtros
        )

        print(
            f"  Vetores: {vetores.shape}"
        )

        # ----------------------------------------------------
        # GUARDAR
        # ----------------------------------------------------

        todos_vetores.append(
            vetores
        )

        # Identificação de cada região
        for regiao in range(
            len(vetores)
        ):

            linha = (
                f"{nome_arquivo};"
                f"regiao_{regiao + 1}"
            )

            identificadores.append(
                linha
            )

    # ========================================================
    # JUNTAR TODAS AS IMAGENS
    # ========================================================

    dados = np.vstack(
        todos_vetores
    )

    return dados, identificadores


# ============================================================
# SALVAR CARACTERÍSTICAS
# ============================================================

def salvar_caracteristicas(
    dados,
    identificadores
):
    """
    Salva todos os vetores em CSV.

    Resultado esperado:

        2048 linhas × 24 características
    """

    caminho = os.path.join(
        PASTA_RESULTADOS,
        "caracteristicas.csv"
    )

    nomes_filtros = [
        "Gabor_0",
        "Gabor_45",
        "Gabor_90",
        "Gabor_135",
        "Laplaciano",
        "LoG",
        "DoG",
        "Circular"
    ]

    cabecalho = [
        "imagem",
        "regiao"
    ]

    for escala in range(
        1,
        NUM_ESCALAS + 1
    ):

        for filtro in nomes_filtros:

            cabecalho.append(
                f"E{escala}_{filtro}"
            )

    with open(
        caminho,
        "w",
        encoding="utf-8"
    ) as arquivo:

        arquivo.write(
            ";".join(cabecalho)
            + "\n"
        )

        for i, vetor in enumerate(
            dados
        ):

            imagem, regiao = (
                identificadores[i]
                .split(";")
            )

            valores = [
                imagem,
                regiao
            ]

            valores.extend(
                [
                    f"{valor:.6f}"
                    for valor in vetor
                ]
            )

            arquivo.write(
                ";".join(valores)
                + "\n"
            )

    print(
        f"\nCaracterísticas salvas em:"
        f"\n{caminho}"
    )

# ============================================================
# NORMALIZAÇÃO DAS CARACTERÍSTICAS
# ============================================================

def normalizar_caracteristicas(dados):
    """
    Normaliza cada uma das 24 características utilizando
    Z-score.

    Para cada característica:

        z = (x - média) / desvio_padrao

    A média e o desvio padrão são calculados considerando
    todas as 2048 regiões das 32 imagens.

    Retorna uma matriz com as mesmas dimensões da original.
    """

    media = np.mean(
        dados,
        axis=0
    )

    desvio_padrao = np.std(
        dados,
        axis=0
    )

    # Evita divisão por zero caso alguma característica
    # tenha exatamente o mesmo valor em todas as regiões.
    desvio_padrao[desvio_padrao == 0] = 1.0

    dados_normalizados = (
        dados - media
    ) / desvio_padrao

    return dados_normalizados.astype(
        np.float32
    )

# ============================================================
# SALVAR CARACTERÍSTICAS NORMALIZADAS
# ============================================================

def salvar_caracteristicas_normalizadas(
    dados,
    identificadores
):
    """
    Salva os vetores normalizados em CSV.
    """

    caminho = os.path.join(
        PASTA_RESULTADOS,
        "caracteristicas_normalizadas.csv"
    )

    nomes_filtros = [
        "Gabor_0",
        "Gabor_45",
        "Gabor_90",
        "Gabor_135",
        "Laplaciano",
        "LoG",
        "DoG",
        "Circular"
    ]

    cabecalho = [
        "imagem",
        "regiao"
    ]

    for escala in range(
        1,
        NUM_ESCALAS + 1
    ):

        for filtro in nomes_filtros:

            cabecalho.append(
                f"E{escala}_{filtro}"
            )

    with open(
        caminho,
        "w",
        encoding="utf-8"
    ) as arquivo:

        arquivo.write(
            ";".join(cabecalho)
            + "\n"
        )

        for i, vetor in enumerate(
            dados
        ):

            imagem, regiao = (
                identificadores[i]
                .split(";")
            )

            valores = [
                imagem,
                regiao
            ]

            valores.extend(
                [
                    f"{valor:.6f}"
                    for valor in vetor
                ]
            )

            arquivo.write(
                ";".join(valores)
                + "\n"
            )

    print(
        f"\nCaracterísticas normalizadas "
        f"salvas em:\n{caminho}"
    )

# ============================================================
# TESTE PRINCIPAL
# ============================================================

def main():

    print("=" * 60)
    print("PROCESSAMENTO DO CONJUNTO DE TEXTURAS")
    print("=" * 60)

    # --------------------------------------------------------
    # CRIAR PASTA DE RESULTADOS
    # --------------------------------------------------------

    os.makedirs(
        PASTA_RESULTADOS,
        exist_ok=True
    )

    # --------------------------------------------------------
    # CRIAR FILTROS
    # --------------------------------------------------------

    filtros = criar_filtros()

    print("\nBanco de filtros:")

    for nome, _ in filtros:

        print(
            " -",
            nome
        )

    print(
        f"\nTotal de filtros: "
        f"{len(filtros)}"
    )

    print(
        f"Total de escalas: "
        f"{NUM_ESCALAS}"
    )

    print(
        f"Características por região: "
        f"{NUM_FILTROS * NUM_ESCALAS}"
    )

    # --------------------------------------------------------
    # PROCESSAR IMAGENS
    # --------------------------------------------------------

    dados, identificadores = (
        processar_todas_imagens(
            filtros
        )
    )

    # --------------------------------------------------------
    # RESULTADOS
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("RESULTADO FINAL")
    print("=" * 60)

    print(
        "\nDimensões da matriz:",
        dados.shape
    )

    print(
        "\nNúmero total de regiões:",
        len(dados)
    )

    print(
        "Número de características:",
        dados.shape[1]
    )

    # --------------------------------------------------------
    # SALVAR CSV
    # --------------------------------------------------------

    salvar_caracteristicas(
        dados,
        identificadores
    )

    # --------------------------------------------------------
    # NORMALIZAÇÃO
    # --------------------------------------------------------

    dados_normalizados = normalizar_caracteristicas(
        dados
    )

    print(
        "\nDimensões após normalização:",
        dados_normalizados.shape
    )

    print(
        "\nPrimeiro vetor normalizado:"
    )

    print(
        dados_normalizados[0]
    )

    # --------------------------------------------------------
    # SALVAR DADOS NORMALIZADOS
    # --------------------------------------------------------

    salvar_caracteristicas_normalizadas(
        dados_normalizados,
        identificadores
    )

    # APLICANDO K-MEANS E VISUALIZAÇÃO
    print("\n" + "=" * 60)
    print("INICIANDO K-MEANS")
    print("=" * 60)

    # 1. Configurar e rodar o K-Means
    num_clusters = 4 # Assumindo 4 tipos principais de textura
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(dados_normalizados)

    # Definir 4 cores distintas para a visualização (formato BGR do OpenCV)
    cores = [
        (0, 0, 255),   # Vermelho
        (0, 255, 0),   # Verde
        (255, 0, 0),   # Azul
        (0, 255, 255)  # Amarelo
    ]

    janelas_por_imagem = 64
    tamanho_janela = 64

    # 2. Descobrir os nomes únicos das imagens a partir dos identificadores
    nomes_arquivos = []
    for ident in identificadores:
        nome_arquivo = ident.split(";")[0]
        if nome_arquivo not in nomes_arquivos:
            nomes_arquivos.append(nome_arquivo)

    # 3. Reconstruir a visualização carregando as imagens novamente
    for i, nome_arquivo in enumerate(nomes_arquivos):
        
        caminho_imagem = os.path.join(PASTA_IMAGENS, nome_arquivo)
        # Carregamos a imagem colorida original (BGR) para o resultado visual ficar melhor
        img = cv2.imread(caminho_imagem) 
        
        if img is None:
            continue

        overlay = np.zeros((512, 512, 3), dtype=np.uint8)
        
        # Pegar os 64 rótulos específicos desta imagem
        inicio = i * janelas_por_imagem
        fim = inicio + janelas_por_imagem
        labels_img = labels[inicio:fim]
        
        # Pintar cada região
        for j, label in enumerate(labels_img):
            linha = (j // 8) * tamanho_janela
            coluna = (j % 8) * tamanho_janela
            cor = cores[label % len(cores)]
            cv2.rectangle(overlay, (coluna, linha), (coluna + tamanho_janela, linha + tamanho_janela), cor, -1)
        
        # Sobrepor as cores com 30% de transparência na imagem original
        imagem_final = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
        
        # Salvar o resultado dentro da pasta "resultados"
        caminho_salvar = os.path.join(PASTA_RESULTADOS, f'cluster_{nome_arquivo}')
        cv2.imwrite(caminho_salvar, imagem_final)

    print("\nK-Means concluído!")
    print(f"As imagens categorizadas foram salvas na pasta '{PASTA_RESULTADOS}/'.")
    print("\nProcessamento finalizado com sucesso.")

if __name__ == "__main__":
    main()