import math
import os
import threading
import time

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.animation as anim
import matplotlib.pyplot as plot
import numpy as np
from numpy.linalg import linalg

MAX_ITER = 100000
# Minimum of 3 elements, first is inputs,
# second is Kohonen dimensionality (root squared)
# and last is outputs
NEURONS_BY_LAYER = [2, 5, 4, 1]
ETA = .3
ERROR_THRESHOLD = 0.1
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(THIS_FOLDER, 'data/xor.txt')
GAMA = .9
MOMENTUM = False

# style.use('fivethirtyeight')

np.random.seed(int(time.time()))

fig = plot.figure()
ax = fig.add_subplot(1, 1, 1)
xs = []
ys = []
lock = threading.Lock()

stop_anim = False


def animate(i):
    lock.acquire()
    if len(xs) > 0 and len(ys) > 0:
        ax.plot(xs, ys, 'b-')
        last_x = xs[len(xs) - 1]
        last_y = ys[len(ys) - 1]
        xs.clear()
        ys.clear()
        xs.append(last_x)
        ys.append(last_y)
    lock.release()


an = anim.FuncAnimation(fig, animate, interval=50, blit=False)
plot.interactive(False)
plot.show(block=False)


def mag(x):
    if x.shape[0] == 1:
        return np.asscalar(np.sqrt(np.dot(x, x.T)))
    elif x.shape[1] == 1:
        return np.asscalar(np.sqrt(np.dot(x.T, x)))
    else:
        return linalg.norm(x)


def norm(x):
    return x / mag(x)


def transfer_f(x):
    return linear_tf(x)


def derivative_tf(x):
    return linear_derivative(x)


def step_tf(x):
    return (x > 0) * 1


def step_derivative(x):
    return (np.abs(x) < 0.001) * .5


def linear_tf(x):
    return x


def linear_derivative(x):
    return np.ones(x.shape)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def rbf(x, c, s):
    # Equivalente à -(x - c)²/sigma, mas aplicado como uma operação entre matrizes
    return np.exp(np.sum(-((x[..., np.newaxis] - c[np.newaxis, ...]) ** 2) / s[np.newaxis, ...], axis=1))


def coalesce_error(x):
    # e = amax(abs(x))
    e = np.average(abs(x))
    # e = mag(x)
    return e


np.random.seed(int(time.time()))


# Carrega arquivo de treinamento passado
def load_dataset(path):
    data = np.loadtxt(path,
                      comments=';;', delimiter=',')
    data = np.array(data)
    _i = NEURONS_BY_LAYER[0]
    _o = NEURONS_BY_LAYER[-1]
    x = data[:, :_i]
    y = data[:, -_o:]
    s = data.shape[0]
    return x, y, s


# Calcula grafo que relaciona a distância entre cada elemento de x e cada centroide de grupo c
def distance_graph(x, c):
    return np.sqrt(np.sum((x[..., np.newaxis] - c[np.newaxis, ...]) ** 2, axis=1))


# Calcula o grafo de pertinência (quais elementos de x pertencem ao grupo dos centróides c)
def pertains_graph(x, c):
    # Grafo de distancia (pontos x -versus- medias m)
    d = distance_graph(x, c.T)
    m = np.reshape(np.amin(d, axis=1), (-1, 1))
    return (d == m) * 1, d


# Calcula o grafo que indica os elementos de x mais distantes de cada centros c
def farthest_graph(x, c):
    # Grafo de grupos
    u, d = pertains_graph(x, c.T)
    # Marca zeros com -1 para evitar confusão na seleção de pontos com distância 0
    u = np.where(u == 0, -1, u)
    # Números positivos indicam distância do ponto até o centro do grupo
    g = u * d
    # Indice de linha do maior valor
    # (é por causa desta linha que marcamos com números
    # negativos distância de grupos não pertencentes, pois
    # grupos de um único elemento tem distância zero, o que
    # faria pontos de outros grupo serem selecionados caso
    # tambem fossem marcados com 0)
    m = np.argmax(g, axis=0)
    # Indices de coluna dos grupos
    n = np.arange(u.shape[1])
    # Marca com 1 os pontos mais distante de cada grupo
    f = np.zeros(np.shape(u))
    f[m, n] = 1
    # Matrix de pertinencia original
    u = (u > 0) * 1
    # Um grupo sem elementos não tera nenhum ponto selecionado por causa de f * u
    return f * u, d, u


#
def k_means_clustering(x, k):
    dim = x.shape[1]
    # Maximo de cada eixo (coluna)
    axis_max = np.amax(x, axis=0)
    # Minimo de cada eixo (coluna)
    axis_min = np.amin(x, axis=0)
    # Medias inicializadas aleatoriamente (dentro do maximo e minimo em cada eixo)
    m = np.random.random((k, dim)) * (axis_max - axis_min) + axis_min
    lm = None

    # Continua execucao ate nao haver mais alteração nas medias
    while not np.array_equal(m, lm):
        lm = m
        # Matriz de pertinencia (pontos x -versus- medias m)
        u, d = pertains_graph(x, m)
        # Calculo das centroides
        c = group_centroid(x, u)
        # Atualiza novas médias com centróides dos grupos encontrados
        m = np.where(np.isnan(c), m, c)

    return m.T


# Calcula dispersão pela média das distâncias
def dispersion_mean(x, c, u):
    # Vetores de distância (padrões p x entradas i x clusters c)
    v = x[..., np.newaxis] - c[np.newaxis, ...]
    # Rotaciona vetor de pertinencia u para ficar no formato de v (padrões p x 1 x clusters c)
    r = np.rot90(u[np.newaxis, ...], -1)
    # Quantidade de elementos em cada grupo
    s = np.sum(u, axis=0)
    # Seleciona a distância em cada eixo
    d = abs(r * v)
    # Média das distâncias
    m = np.sum(d, axis=0) / s.reshape((1, -1))
    # Aplica uma Disperção minima positiva diferente de 0
    return np.where(m > 0, m, .1) * 1.2


# Calcula dispersão pelo elemento mais distante
def dispersion_max_elem(x, c):
    # existem duas formas de fazer o calculo:
    # Primeira: usando a média dos N vetores mais proximos =. sigma = 1/N somatorio ||centro mais proximo - vetores mais proximos||, N = vetores mais proximos
    # Segunda: pelo ponto mais distantes pertencentes ao cluster do centro Cj
    # A segunda e mais facil a implementacao
    #
    # 1 - Recebe sigma e os centros
    # 2 - dentro de um loop, calcula qual é o maior em ||centro - entradai || e atribui o maior para sigma
    #  repare que quem se altera é somente a entrada
    # 3 - retorna sigma

    # Matriz indica qual ponto x é o mais distante de cada grupo
    f, _, _ = farthest_graph(x, c)

    # Marcando com nan grupos vazios
    n = np.sum(f, axis=0)
    f = f / n

    # Seleção do vetor mais distante
    v = np.dot(x.T, f)
    # Distância entre vetor e centro do grupo
    d = np.abs(c - v)

    # Disperação padrão para distância zero ou grupo vazio
    s = np.where(np.logical_and(d != 0, np.logical_not(np.isnan(d))), d, 0.1)
    # Aumenta em 20% area de dispersão para melhorar englobamento
    return s * 1.2


# Calcula as centroides de um grupo
def group_centroid(x, u):
    # Quantidade de pontos em cada grupo
    n = np.sum(u, axis=0)
    # Somatório das coordenadas dos pontos
    s = np.dot(x.T, u)
    # Calculo das centroides
    c = (s / n).T
    return c


# Calcula o gráfico de pertinência de cada grupo a partir dos padrões de entradas x e os pesos da rede de kohonen w
def kohonen_clustering(x, w):
    # Saída da rede dado os padrões de entrada
    y = np.dot(x, w)
    # Indice dos neurônios com maior ativação por padrão
    m = np.argmax(y, axis=1)
    # Mapeamento d dos neurônios m para um grupo n
    n = np.unique(m)
    d = dict({k: v for v, k in enumerate(n, 0)})
    # Indice dos grupos
    j = np.array(list(map(lambda it: d[it], m)))
    i = np.arange(j.size)
    # Matriz de pertinência dos grupos,
    # ativo no indice dos grupos correspondentes dos neuronios vencedores para cada padrão
    u = np.zeros((j.size, n.size))
    u[i, j] = 1
    return u


# Faz clusterização utilizando mapas de kohonen,
# onde x são os padrões de entradas e k é a raiz quadrada da dimensão da rede
def kohonen(x, k):
    # Coeficiente de aprendizagem
    alpha = 0.08
    # Número de padrões é o numero de linhas em x
    p = x.shape[0]
    # Dimensão das entradas é o numero de colunas em x
    m = x.shape[1]
    # Dimensão do mapa de kohonen
    n = k ** 2
    # Inicialização da matriz de pesos
    w = np.random.random((m, n))
    # Normalizando para cada neurônio
    w = w / np.sqrt(np.sum(w ** 2, axis=0))

    # Treinamento da rede
    for epoch in range(1000):
        # Saída da rede em relação para cada padrão de entrada
        y = np.dot(x, w)
        # Indices dos neurônios vencedores na ativação de cada padrão
        i = np.arange(p)
        j = np.argmax(y, axis=1)
        # Matriz de correlação entre padrões e os neurônios vencedores
        s = np.zeros((p, n))
        s[i, j] = 1
        # Matriz de peso na vizinhança
        v = np.array(s)
        v += (np.roll(v, 1, axis=1) +
              np.roll(v, -1, axis=1) +
              np.roll(v, k, axis=1) +
              np.roll(v, -k, axis=1)) * .5
        v += (np.roll(v, 1, axis=1) +
              np.roll(v, -1, axis=1) +
              np.roll(v, k, axis=1) +
              np.roll(v, -k, axis=1)) * .5
        # Normalização do máximo para 1 na matriz de vizinhanças
        h = np.amax(v)
        v /= h
        # Pesos dos neurônios selecionados
        ws = np.dot(s, w.T)
        # Correção em relação à cada padrão
        u = x - ws
        # Correção para a neurônios selecionados e vizinhança
        d = np.dot(u.T, v)
        # Aplica correção do treinamento
        w += alpha * d
        # Normalização para cada neurônio
        # w = w / np.sqrt(np.sum(w ** 2, axis=0))
    return w


def training():
    time.sleep(1)
    x, y, n_patterns = load_dataset(PATH)

    k = kohonen(x, NEURONS_BY_LAYER[1])
    clusters = kohonen_clustering(x, k)

    NEURONS_BY_LAYER[1] = clusters.shape[1]

    # Parametro de centro para RBF
    c = group_centroid(x, clusters).T

    # Vetor de dispersão
    sigma = dispersion_mean(x, c, clusters)

    # Aplicando a RBF
    rbf_input = rbf(x, c, sigma)
    x_p = np.hstack((np.ones((n_patterns, 1)), rbf_input))
    y_p = y

    NEURONS_BY_LAYER.pop(0)

    w = []

    # Vetor de pesos aleatorizado para cada camada
    n_layers = len(NEURONS_BY_LAYER) - 1
    for i in range(n_layers):
        row = NEURONS_BY_LAYER[i] + 1
        col = NEURONS_BY_LAYER[i + 1] + (1 if i < n_layers - 1 else 0)
        rn = np.random.random((row, col))
        wl = 2 * rn - 1
        w.append(wl)

    stopped_by_error_threshold = True
    gi = 0

    gama = GAMA

    d2_w = []
    for w_it in w:
        d2_w.append(w_it * 0)

    eta = ETA
    for epoch in range(MAX_ITER):
        for i in range(len(w) - 1):
            w[i][:, 0] = 0
            w[i][0][0] = 1

        g = []
        u = []
        y_n = x_p
        for i, w_it in enumerate(w):
            u_n = np.dot(y_n, w_it)
            g_n = transfer_f(u_n)
            if i < len(w) - 1:
                g_n[:, 0] = 1
            u.append(u_n)
            g.append(g_n)
            y_n = g_n

        dg = list(map(lambda u_it: derivative_tf(u_it), u))

        e_n = y_p - g[-1]
        e = [e_n]
        for dg_it, w_it in zip(reversed(dg[1:]), reversed(w[1:])):
            delta_n = e_n * dg_it
            e_n = np.dot(delta_n, w_it.T)
            e.insert(0, e_n)

        if np.average(coalesce_error(e[-1])) < ERROR_THRESHOLD:
            print("Final Epoch :")
            print(epoch)
            stopped_by_error_threshold = True
            break

        if epoch % 500 == 0:
            print("Current Epoch:")
            print(epoch)
            print("Errors: ")
            for i, e_it in enumerate(e):
                print("  Layer: %d" % i)
                print("  Avg: %.6f" % np.average(abs(e_it)))
                print("  Mag: %.6f\n" % mag(e_it))
            print("\nEtha")
            print(eta)
            print("\nWeights")
            print(w)
            print("\nError")
            print(e)

        lock.acquire()
        xs.append(gi)
        ys.append(coalesce_error(e[-1]))
        lock.release()

        gi += 1

        e = list(map(lambda e_it, dg_it: e_it * dg_it, e, dg))

        y_h = g[:-1]
        y_h.insert(0, x_p)
        d_w = list(map(lambda y_it, e_it: np.dot(y_it.T, e_it) / n_patterns, y_h, e))

        for i in range(len(w)):
            w[i] += eta * d_w[i] + (gama * d2_w[i] if MOMENTUM else 0)

        d2_w = list(map(lambda d_w_it: eta * d_w_it, d_w))

    if not stopped_by_error_threshold:
        print("Iteration :")
        print(MAX_ITER)

    print("\nInput ")
    print(x)
    print("\nRBF Layer")
    print(x_p)
    print("\nCenter")
    print(c)
    print("\nDispersion")
    print(sigma)

    # testing output
    g = []
    y_n = x_p
    for i, w_it in enumerate(w):
        u_n = np.dot(y_n, w_it)
        g_n = transfer_f(u_n)
        if i < len(w) - 1:
            g_n[:, 0] = 1
        g.append(g_n)
        y_n = g_n

    print("\nOutput")
    print(y_n)
    print("\nTraining Data")
    print(y_p)

    print("\nError")
    dg = list(map(lambda g_it: derivative_tf(g_it), g))
    e_n = y_p - g[-1]
    e = [e_n]
    for dg_it, w_it in zip(reversed(dg[1:]), reversed(w[1:])):
        delta_n = e_n * dg_it
        e_n = np.dot(delta_n, w_it.T)
        e.insert(0, e_n)

    for i, e_it in enumerate(e):
        print("  Layer: %d" % i)
        print("  Avg: %.6f" % np.average(abs(e_it)))
        print("  Mag: %.6f\n" % mag(e_it))

    print("\nEtha")
    print(eta)

    print("\nSynaptic Weights")
    for i in range(len(w) - 1):
        w[i][:, 0] = 0
        w[i][0][0] = 1

    for i, w_it in enumerate(w):
        print("Layer: %d" % i)
        print(w_it)

    while True:
        time.sleep(0)


t = threading.Thread(target=training)
t.start()
