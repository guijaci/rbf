import threading
import time

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.animation as anim
import matplotlib.pyplot as plot
import numpy as np
from numpy.linalg import linalg

MAX_ITER = 100000
# Minimum of 2 elements, first is inputs, last is outputs
NEURONS_BY_LAYER = [2, 1]
ETA = .03
ERROR_THRESHOLD = 0.1
PATH = '/home/guilherme/PycharmProjects/rbf/data/xor.txt'
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
    # Equivalente à (x - c)²/sigma, mas aplicado como uma operação entre matrizes
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


def training():
    x, y, n_patterns = load_dataset(PATH)

    # Parametro de centro para RBF
    c = np.array([[1, 0],
                  [0, 1]])

    # Vetor de dispersão
    sigma = np.array([[.01, .01],
                      [.01, .01]])

    #Aplicando a RBF
    x_p = np.hstack((np.ones((n_patterns, 1)), rbf(x, c, sigma)))
    y_p = y

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
time.sleep(1)
