from cProfile import label
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy import stats
from scipy import linalg
import matplotlib.colors as colors


def paso_RKN(funcion, h, t_n, y_n, butcher_tableau):
    """
    Un integrador númerico generico que resuelve el
    sistema de Ecuaciones diferenciales :

    dy/dt = f(t, y)  con y = [y1(t), y2(t), y3(t),..., yn(t)]
    Calcula un paso iterativo con algún Método de Runge Kutta
    explicito o implicito de orden N

    Parametros
    -----------
    funcion : lado derecho de la ecuación diferencial
              type(funcion) = numpy.ndarray
              np.shape(funcion) = (n,)  n filas 1 columna

    h       : paso del método iterativo, debe ser distinto de 0
              type(h) = float

    t_n     : variable independiente del sistema
              type(t_n) = float

    y_n     : vector de funciones del sistema
              type(y) = numpy.ndarray
              np.shape(y) = (n,)  n filas 1 columna

    butcher_tableau : corresponde a una matriz que especifica
    el método numérico a utilizar, es de la forma :

    butcher_tableau = ([[c1, a11, a12, . , . , . , a_1N],
                        [c2, a21, a22, . , . , . , a_2N],
                        [c3, a31, a32, . , . , . , a_3N],
                        [ . , . , . ,              , . ],
                        [ . , . , . ,              , . ],
                        [ . , . , . ,              , . ],
                        [c_N1, c_N2 , . , . , .  , c_NN],
                        [ 0 , b1 , b2 , . , . , . , b_N]])

                  y_n+1 = y_n + h * s1

    donde  s1 = suma (b_i * k_i, i=1 hasta i=N)
    k_i = funcion( t_n + c_i*h, y_n + h * s2)
    con s2 = suma(a_ij * k_j, j=1 hasta j=N)

    Que corresponde a métodos implicitos (incluye los explicitos ya que
    su matriz es triangular inferior, --> s2 suma desde j=1 hasta j=i-1)

    ejemplo :
    Metodo_punto_medio = np.array([[0, 0, 0],
                                  [1/2, 1/2, 0],
                                  [0, 0, 1]])
    entrega la solución
    y_n+1 = y_n + h * funcion(t_n + 1/2 * h, y_n + 1/2 * h * funcion(t_n, y_n))

    salida:
    output : paso siguente y_n+1 dado t_n , y_n , la funcion, h, y el método
             type(output) = numpy.ndarray
             np.shape(output) = (n,)  n filas 1 columna
    """
    b = np.delete(butcher_tableau[-1, :], 0)
    a = butcher_tableau[0:len(b), 1:len(b)+1]
    c = np.delete(butcher_tableau[:, 0], -1)
    k = np.zeros((len(b), len(y_n)))
    s1 = np.zeros(np.shape(y_n))
    for i in range(0, len(b)):
        s2 = np.zeros(len(y_n))
        for j in range(0, len(b)):
            s2 = s2 + a[i][j] * k[j]
        A = t_n + c[i] * h
        B = y_n + h * s2
        k[i] = funcion(A, B)
        s1 = s1 + b[i] * k[i]
    output = (y_n + h * s1)
    return output


Euler = np.array([[0, 0],
                  [0, 1]])


EulerImplicit = np.array([[1, 1],
                          [0, 1]])


LobattoIIIC = np.array([[0, 1/2, -1/2],
                        [1, 1/2, 1/2],
                        [0, 1/2, 1/2]])

LobattoIIID = np.array([[0, 1/6, 0, -1/6],
                        [1/2, 1/12, 5/12, 0],
                        [1, 1/2, 1/3, 1/6],
                        [0, 1/6, 2/3, 1/6]])


RadauIIA_3  = np.array([[1/3, 5/12, -1/12],
                        [1, 3/4, 1/4],
                        [0, 3/4, 1/4]])



RadauIIA_5 = np.array([[2/5-np.sqrt(6)/10, 11/45-7*np.sqrt(6)/360, 37/225-169*np.sqrt(6)/1800, -2/225+np.sqrt(6)/75],
                       [2/5+np.sqrt(6)/10, 37/225+169*np.sqrt(6)/1800, 11/45+7*np.sqrt(6)/360, -2/225-np.sqrt(6)/75],
                       [1, 4/9-np.sqrt(6)/36, 4/9+np.sqrt(6)/36, 1/9],
                       [0, 4/9-np.sqrt(6)/36, 4/9+np.sqrt(6)/36, 1/9]])


Fehlberg2 = np.array([[0, 0, 0, 0],
                      [1/2, 1/2, 0, 0],
                      [1, 1/256, 255/256, 0],
                      [0, 1/256, 255/256, 0]])


RK2 = np.array([[0, 0, 0],
                [1/2, 1/2, 0],
                [0, 0, 1]])


RK3 = np.array([[0, 0, 0, 0],
                [1/3, 1/3, 0, 0],
                [3/4, -3/16, 15/16, 0],
                [0, 1/6, 3/10, 8/15]])


RK4 = np.array([[0, 0, 0, 0, 0],
                [1/2, 1/2, 0, 0, 0],
                [1/2, 0, 1/2, 0, 0],
                [1, 0, 0, 1, 0],
                [0, 1/6, 1/3, 1/3, 1/6]])


RK5 = np.array([[0, 0, 0, 0, 0, 0, 0],
                [1/4, 1/4, 0, 0, 0, 0, 0],
                [1/4, 1/8, 1/8, 0, 0, 0, 0],
                [1/2, 0, -1/2, 1, 0, 0, 0],
                [3/4, 3/16, 0, 0, 9/16, 0, 0],
                [1, -3/7, 2/7, 12/7, -12/7, 8/7, 0],
                [0, 7/90, 0, 32/90, 12/90, 32/90, 7/90]])

DP4 = np.array([[0, 0, 0, 0, 0, 0, 0],
                [1/5, 1/5, 0, 0, 0, 0, 0],
                [3/10, 3/40, 9/40, 0, 0, 0, 0],
                [3/5, 3/10, -9/10, 6/5, 0, 0, 0],
                [1, -11/54, 5/2, -70/27, 35/27, 0, 0],
                [7/8, 1631/55296, 175/512, 575/13824,
                 44275/110592, 253/4096, 0],
                [0, 2825/27648, 0, 18575/48384, 13525/55296, 277/14336, 1/4]])

DP5 = np.array([[0, 0, 0, 0, 0, 0, 0],
                [1/5, 1/5, 0, 0, 0, 0, 0],
                [3/10, 3/40, 9/40, 0, 0, 0, 0],
                [3/5, 3/10, -9/10, 6/5, 0, 0, 0],
                [1, -11/54, 5/2, -70/27, 35/27, 0, 0],
                [7/8, 1631/55296, 175/512, 575/13824,
                 44275/110592, 253/4096, 0],
                [0, 37/378, 0, 250/621, 125/594, 0, 512/1771]])


def RKN_adaptativo(fun, dt, t, y, Metodos, orden1, orden2, tolerancia):
    """
    Un integrador númerico generico que resuelve el
    sistema de Ecuaciones diferenciales :

    dy/dt = f(t, y)  con y = [y1(t), y2(t), y3(t),..., yn(t)]

    calcula usando paso adaptativo mediante dos metodos de orden N y
    el otro metodo de orden N+1

    Parametros
    -----------
    fun      : lado derecho de la ecuación diferencial
              type(funcion) = numpy.ndarray
              np.shape(funcion) = (n,)

    dt       : paso que va ha ser aumentado o disminuido segun corresponda
              type(h) = float

    t       : variable independiente del sistema que será cambiada usando dt
              type(t_n) = float

    y_n     : vector de funciones del sistema
              type(y) = numpy.ndarray
              np.shape(y) = (n,)  n filas 1 columna

    Metodos  : vector de metodos a utilizar, deben ser dos
               type(Metodos) = numpy.ndarray or list
               np.shape(y) = (2,)  dos filas 1 columna

    orden1   : orden del primer metodo , debe ser el menor
             : type(orden1) = int

    orden2   : orden del segundo metodo, debe ser el mayor
             : type(orden2) = int

    tolerancia : medida aproximada del error de truncacion local,
                 al acotar h por este indirectamente se acota el error global
                 asumiendo que el metodo2 es lo suficientemente bueno

    salida:

    output : Entrega el vector [y_suguiente, t_suguiente, dt] es decir
             el paso siguiente considerando la estrategia para controlar dt
             type(output) = numpy.ndarray
             np.shape(output) = (,3)  1 filas 3 columnas
    """
    y1 = paso_RKN(fun, dt, t, y, Metodos[0])   # avanza mediante un metodo O(N)
    y2 = paso_RKN(fun, dt, t, y, Metodos[1])   # y con un metodo mas eficiente
    error = max(abs(y2 - y1))   # local truncation error estimation
    if error < tolerancia or error == 0:  # acepto el paso
        y_suguiente = paso_RKN(fun, dt, t, y, Metodos[1])  # avanzo primero
        t_suguiente = t + dt
        dt = dt * (tolerancia / error) ** (1 / (orden2))  # y lo aumento
    else:
        while error > tolerancia:
            dt = dt * (tolerancia / error) ** (1 / (orden1))   # lo disminuyo
            # dt = dt / 2
            y1 = paso_RKN(fun, dt, t, y, Metodos[0])
            y2 = paso_RKN(fun, dt, t, y, Metodos[1])
            error = max(abs(y2 - y1))
        y_suguiente = paso_RKN(fun, dt, t, y, Metodos[1])
        t_suguiente = t + dt
    return [y_suguiente, t_suguiente, dt]


def solver_ode(funcion, t0, y0, h0, n_steps, Metodos=[RK4, RK5],
               ordenes=[4, 5], tolerancia=1e-4):
    t = np.zeros(n_steps)
    y = np.zeros((n_steps, len(y0)))
    h = np.zeros(n_steps)
    t[0] = t0
    y[0] = y0
    h[0] = h0
    for i in range(1, n_steps):
        if len(Metodos) == 1:
            h[i] = h0
            y[i] = paso_RKN(funcion, h0, t[i-1], y[i-1], Metodos[0])
            t[i] = h0+t[i-1]
        else:
            solver = RKN_adaptativo(funcion, h[i-1], t[i-1], y[i-1],
                                    [Metodos[0], Metodos[1]], ordenes[0],
                                    ordenes[1], tolerancia)
            h[i] = solver[2]
            y[i] = solver[0]
            t[i] = solver[1]
    return [y, t, h]


def edo_prueba(t, y):
    return np.array([-0.6 * y[0] + 10 * np.exp(-(t-2)**2/(2*(0.075)**2))])


def analitica_prueba(t):
    output = np.exp(-0.6*t)*(3.62402-3.12402*sp.special.erf(18.888-9.42809*t))
    return output


"""
Prueba = solver_ode(edo_prueba, 0, np.array([0.5]), 0.1, 21)
x = Prueba[1]
y = Prueba[0][:, 0]
t = np.linspace(0, 4, 1000)
plt.figure()
plt.title("Gráfico solución de la EDO y'=10exp(-((x-2)^2/(2*(0.075)))^2)",
          fontsize=16)
plt.plot(t, analitica_prueba(t), label='Solución analítica con y(0) = 1/2')
plt.plot(x, y, 'bo', label='RK45 paso adaptativo')
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.legend(fontsize=14)
plt.show()


sigma = 10
rho = 28
beta = 8/3


def funcion_lorentz(t, y):
    output = np.array([sigma * (y[1] - y[0]),
                       y[0] * (rho - y[2]) - y[1],
                       y[0] * y[1] - beta * y[2]])
    return output


fig4 = plt.figure(frameon=False)
fig4.clf()
ax = fig4.add_subplot(111, projection='3d')

LORENTZ = solver_ode(funcion_lorentz, 0, np.array([1, -2, 1]), 0.01, 2000)
solucion_lorentz = LORENTZ[0]
tiempo_lorentz = LORENTZ[1]
u = solucion_lorentz[:, 0]
v = solucion_lorentz[:, 1]
w = solucion_lorentz[:, 2]

cs = CubicSpline(tiempo_lorentz, solucion_lorentz)
xs = np.linspace(0, tiempo_lorentz[-1], 10**4)
ax.set_title("Atractor de Lorentz  ", fontsize=16)
ax.plot(u, v, w, 'o', markersize=5, label='Puntos obtenidos mediante RK45')
ax.plot(cs(xs)[:, 0], cs(xs)[:, 1], cs(xs)[:, 2],
        'k-', label='Ajuste Spline cúbica', linewidth=1)
ax.set_xlabel('x', fontsize=16)
ax.set_ylabel('y', fontsize=16)
ax.set_zlabel('z', fontsize=16)
plt.legend(fontsize=14)
plt.show()
"""


def Stability_Function(z, butcher_tableau):
    b = np.delete(butcher_tableau[-1, :], 0)
    a = butcher_tableau[0:len(b), 1:len(b)+1]
    identidad = np.identity(len(a))
    ones = np.ones((len(b), 1))
    r = 1+z*b@linalg.inv(identidad-z*a)@ones
    return r[0]


def Evalue_Stability_Function(Method, xlim, ylim, resolution):
    x = np.linspace(xlim[0], xlim[1], resolution).reshape((resolution, 1))
    y = np.linspace(ylim[0], ylim[1], resolution).reshape((1, resolution))[0]
    z = np.zeros((len(x), len(y)), dtype=complex)

    for i in range(len(x)):
        for j in range(len(y)):
            A = np.abs((Stability_Function(complex(x[i], y[j]), Method)))
            z[j][i] = (A < 1).astype(int)*(1-A)
    output = z
    return output

N = 300
xlim = [-4, 2]
ylim = [-4, 4]

AS_Euler = Evalue_Stability_Function(Euler, xlim, ylim, N)
AS_RK2 = Evalue_Stability_Function(RK2, xlim, ylim, N)
AS_RK3 = Evalue_Stability_Function(RK3, xlim, ylim, N)
AS_RK4 = Evalue_Stability_Function(RK4, xlim, ylim, N)
AS_RK5 = Evalue_Stability_Function(RK5, xlim, ylim, N)
xlim_2 = [-15,15]
ylim_2 = [-15,15]
AS_EulerImplicit = Evalue_Stability_Function(EulerImplicit, xlim_2, ylim_2, N)
AS_LobattoIIIC = Evalue_Stability_Function(LobattoIIIC, xlim_2, ylim_2, N)
AS_LobattoIIID = Evalue_Stability_Function(LobattoIIID, xlim_2, ylim_2, N)
AS_RadauIIA_3 = Evalue_Stability_Function(RadauIIA_3, xlim_2, ylim_2, N)
AS_RadauIIA_5 = Evalue_Stability_Function(RadauIIA_5, xlim_2, ylim_2, N)


fig, axes = plt.subplots(2, 5)
fig.suptitle("A-Stability of Explicit and Implicit Runge Kutta schemes")  

axes[0, 0].set_title('Euler ')
axes[0, 1].set_title('Runge Kutta 2')
axes[0, 2].set_title('Runge Kutta 3')
axes[0, 3].set_title('Runge Kutta 4')
axes[0, 4].set_title('Runge Kutta 5')
axes[1, 0].set_title('Implicit Euler ')
axes[1, 1].set_title('LobattoIIIC ')
axes[1, 2].set_title('LobattoIIID ')
axes[1, 3].set_title('RadauIIA_3')
axes[1, 4].set_title('RadauIIA_5')

axes[0, 0].imshow(AS_Euler.astype(float), extent=(xlim[0], xlim[1],
           ylim[0], ylim[1]), origin="upper", cmap="inferno")

axes[0, 1].imshow(AS_RK2.astype(float), extent=(xlim[0], xlim[1],
           ylim[0], ylim[1]), origin="upper", cmap="inferno")

axes[0, 2].imshow(AS_RK3.astype(float), extent=(xlim[0], xlim[1],
           ylim[0], ylim[1]), origin="upper", cmap="inferno")

axes[0, 3].imshow(AS_RK4.astype(float), extent=(xlim[0], xlim[1],
           ylim[0], ylim[1]), origin="upper", cmap="inferno")

axes[0, 4].imshow(AS_RK5.astype(float), extent=(xlim[0], xlim[1],
           ylim[0], ylim[1]), origin="upper", cmap="inferno")

axes[1, 0].imshow(AS_EulerImplicit.astype(float), extent=(xlim_2[0], xlim_2[1],
           ylim_2[0], ylim_2[1]), origin="upper", cmap="inferno")

axes[1, 1].imshow(AS_LobattoIIIC.astype(float), extent=(xlim_2[0], xlim_2[1],
           ylim_2[0], ylim_2[1]), origin="upper", cmap="inferno")

axes[1, 2].imshow(AS_LobattoIIID.astype(float), extent=(xlim_2[0], xlim_2[1],
           ylim_2[0], ylim_2[1]), origin="upper", cmap="inferno")

axes[1, 3].imshow(AS_RadauIIA_3.astype(float), extent=(xlim_2[0], xlim_2[1],
           ylim_2[0], ylim_2[1]), origin="upper", cmap="inferno")

axes[1, 4].imshow(AS_RadauIIA_5.astype(float), extent=(xlim_2[0], xlim_2[1],
           ylim_2[0], ylim_2[1]), origin="upper", cmap="inferno")
plt.show()
