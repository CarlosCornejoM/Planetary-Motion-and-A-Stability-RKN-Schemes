import numpy as np
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import rkn


class NCuerpos(object):

    def __init__(self, posiciones_iniciales, velocidades_iniciales, mass_dist, G=1):
        self.G = G
        self.N = int(len(posiciones_iniciales)/2)
        self.m = mass_dist
        self.y = np.concatenate((posiciones_iniciales, velocidades_iniciales),dtype=float)
        self.orbit_list = [self.y]

    def _f(self, p1, p2, m1, m2):
        '''Calcula el vector fuerza gravitacional entre 2 partículas

        p1 y p2 son las posiciones de las partículas y m1 y m2 sus masas.
        La fuerza calculada es la fueza que ejerce la partícula 2 sobre la
        partícula 2.
        '''
        distancia = np.linalg.norm(np.array(p2) - np.array(p1))
        n = (np.array(p2) - np.array(p1)) / distancia
        fuerza = (self.G * m1 * m2/(distancia**2)) * n
        
        return fuerza

    def gravity(self):
        '''Calcula la aceleración total del sistema.

        La ecuación a resolver es d2r/dt2 = a = gravity (este metodo).

        Este metodo es util tal y como esta para Verlet o Beeman pero
        probablemente sea necesario otro metodo para escribir el lado derecho
        correcto para el metodo de RK una vez que se transforme la ecuacion de
        segundo orden en un sistema de ecuaciones.
        '''
        p = np.zeros((2*self.N,2))
        gravity = np.zeros((self.N,2))

        for i in range((2*self.N)) :
            a=int(2*i)
            b=int(2*(i+1))
            p[i] = self.y[a:b]
        for i in range((self.N)):
            s=0
            for j in range((self.N)):
                if i==j:
                    continue
                s+= self._f(p[i], p[j], self.m[i], self.m[j])
            gravity[i]=s/self.m[i]

        a = np.concatenate(gravity)       
        return a

    def funcion(self,t, y):
        p = np.zeros((2*self.N,2))
        gravity = np.zeros((self.N,2))

        for i in range((2*self.N)) :
            a=int(2*i)
            b=int(2*(i+1))
            p[i] = y[a:b]
        for i in range((self.N)):
            s=0
            for j in range((self.N)):
                if i==j:
                    continue
                s+= self._f(p[i], p[j], self.m[i], self.m[j])
            gravity[i]=s/self.m[i]

        F = np.concatenate(gravity)
        vel =y[2*self.N:4*self.N]
        output = np.concatenate([vel, F])
        return output

    def rk4_step(self, h):
        '''Avanza el tiempo integrando la ecuacion de movimiento.

        Utiliza el algoritmo de Runge-Kutta de orden ? para integrar las
        ecuaciones de movimiento con un paso de tamano h.
        '''
        nuevo_y = rkn.paso_RKN(self.funcion, h, 0, self.y, rkn.DP5) 
        self.y = nuevo_y.copy()
        self.orbit_list.append(self.y)

    def verlet_step(self, h):
        '''Avanza el tiempo integrando la ecuacion de movimiento.

        Utiliza el algoritmo de Verlet o Beeman para integrar las ecuaciones de
        movimiento con un paso de tamano h.
        '''
        
        pos_actual = self.y[0:2*self.N]
        vel_actual = self.y[2*self.N:4*self.N]
        acel_actual = self.gravity()
        pos_nueva = pos_actual + vel_actual * h + 1/2 * acel_actual * h**2
        self.y[0:2*self.N] = pos_nueva
        acel_nueva = self.gravity()
        vel_nueva = vel_actual + 1/2 * (acel_actual + acel_nueva) * h
        self.y[2*self.N:4*self.N] = vel_nueva
        self.y = self.y.copy()
        self.orbit_list.append(self.y)

    def energia_del_sistema(self):
        '''Calcula la energía del sistema'''
        p = np.zeros((2*self.N,2))
        for i in range((2*self.N)) :
            a=int(2*i)
            b=int(2*(i+1))
            p[i] = self.y[a:b]
        masa = np.repeat(self.m, 2)
        #r12 = np.linalg.norm(r2-r1)
        #r13 = np.linalg.norm(r3-r1)
        #r23 = np.linalg.norm(r3-r2)
        velociad = self.y[2*self.N:4*self.N]
        energia_cinetica = 1/2 * np.sum((masa*velociad**2))
        potencial=np.zeros((self.N))
        for i in range((self.N)):
            s=0
            for j in range((self.N)):
                if i==j:
                    continue
                s+= self.m[i]*self.m[j]/np.linalg.norm(p[i]-p[j])
            potencial[i]=s

        energia_potencial = -self.G * np.sum(potencial)/2  #estoy contando doble ij y ji
        return energia_cinetica + energia_potencial

    def anim_update(self, frame):
        '''Método necesario para animar el grafico de orbitas'''
        orbit_array = np.array(self.orbit_list)
        for i in range(self.N):

            self.particles[i][0].set_data(orbit_array[frame, i*2], orbit_array[frame, i*2+1])
            self.particles[i][1].set_data(orbit_array[frame-10:frame, i*2],
                                   orbit_array[frame-10:frame, i*2+1])


        return np.concatenate(self.particles)

    def show_animated_orbit(self, fig_number, step, xlim, ylim):
        plt.style.use('dark_background')
        orbit_array = np.array(self.orbit_list)
        fig = plt.figure(fig_number, figsize=(10,10))
        plt.close(fig_number)
        fig = plt.figure(fig_number)
        fig.clf()
        ax = plt.gca()
        ax.set_aspect(1)
        self.particles=[]
        color = iter(cm.hsv(np.linspace(0, 1, self.N)))

     
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
        self.animation = FuncAnimation(fig, self.anim_update,
                                       frames=range(step, len(orbit_array)),
                                       interval=1,
                                       blit=True)
        for i in range(self.N):    
            c = next(color)
            self.particles.append([plt.plot([], [], 'o',color=c, markersize=10)[0]    , plt.plot([], [], 'r--' )[0]   ])
            plt.plot(orbit_array[:, 2*i], orbit_array[:, 2*i+1], '-',color=c ,alpha=0.75)
            
        #self.animation.save('C:/Users/carlo/Documents/GitHub/Tarea 5/gifs/'+str(fig_number)+'.gif',fps=50, writer='ffmpeg', dpi=200)
        plt.grid(alpha=0.25)
        plt.show()


Broucke = np.array([[-0.9892620043, 0, 2.2096177241, 0, -1.2203557197, 0],
                    [0, 1.9169244185, 0, 0.1910268738, 0, -2.1079512924],
                    [1,1,1]])


BrouckeA2 = np.array([[0.336130095,0, 0.7699893804,0, -1.1061194753,0],
                    [0,1.532431537, 0, -0.6287350978, 0, -0.9036964391],
                    [1,1,1]])


BrouckeA10 = np.array([[-0.5426216182,0, 2.5274928067,0, -1.9848711885,0],
                    [0,0.8750200467, 0, -0.0526955841, 0, -0.8223244626],
                    [1,1,1]])


LiFreefall = np.array([[-0.5,0, 0.5,0, 0.083924021,0.3307729197],
                    [0, 0, 0, 0, 0, 0],
                    [1,0.8,0.4]])


posiciones_iniciales = np.array([-0.97000436, 0.24308753,
                                 0.97000436, -0.24308753,
                                 0, 0])


velocidades_iniciales = np.array([0.466203685, 0.43236573,
                                  0.466203685, 0.43236573,
                                  -0.93240737,  -0.86473146])

Au =149_597_870_700
dist = np.array([0, 0, -6.9817079e10, 0, 1.0894185e11, 0, 1.4709511e11,0, 2.4922873e11, 0, 8.1608146e11, 0, 1.503983e12, 0, 3.0063894e12, 0])  # aphelion distance (mercury at perihelia)
vel = np.array([0, 0, 0, -1.2962e-4, 0 , 1.167e-4, 0, 2127/21413747, 0, 8.031e-5, 0, 4.3597e-5, 0, 3.232e-5, 0, 2.272e-5])    # tangential velocity at aphelion 
mass = np.array([1477, 2.451e-4, 0.003615, 0.00444, 0.004764, 1.409, 0.4219, 0.06445])
sistema_solar = np.array([dist,
                          vel,
                          mass])

sistema_solar_verlet = NCuerpos(sistema_solar[0], sistema_solar[1], sistema_solar[2])
verlet_step_size = 9.460895e12
N_steps = 1000    # un año es 9.460895e15 en sistema natural
EMT_sistema_solar = np.zeros(N_steps)
for i in range(N_steps):
    EMT_sistema_solar[i] = sistema_solar_verlet.energia_del_sistema()
    sistema_solar_verlet.verlet_step(verlet_step_size)

sistema_solar_verlet.show_animated_orbit(2, 0, [-2*Au,2*Au], [-2*Au,2*Au])   #2au


# Geometrized Units
# sun mass in GU = Sun mass kg /((1.3466 e27 kg)=c/G)
# time in GU = s/c
# distance in GU = m
#