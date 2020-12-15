import numpy as np
from sko.SA import SA
import matplotlib.pyplot as plt
import pandas as pd

obj_func = lambda x: 0.2 + x[0] ** 2 + x[1] ** 2 - 0.1 * np.cos(6 * np.pi * x[0]) - 0.1 * np.cos(6 * np.pi * x[1])


def simulated_annealing(T_max, T_min, L, max_stay_counter):
    sa = SA(func=obj_func, x0=[0.8, -0.5], T_max=T_max, T_min=T_min, L=L, max_stay_counter=max_stay_counter)
    best_x, best_y = sa.run()
    print("best_x: " + str(best_x) + " , best_y: " + str(best_y))

    y_plot, = plt.plot(pd.DataFrame(sa.best_y_history).cummin(axis=0), color='red')
    plt.legend(handles=[y_plot], labels=['y'], loc='best')
    plt.show()

    x1_plot, x2_plot = plt.plot(pd.DataFrame(sa.best_x_history).cummin())
    plt.legend(handles=[x1_plot, x2_plot], labels=['x1', 'x2'], loc='best')
    plt.show()


if __name__ == '__main__':
    simulated_annealing(1, 1e-9, 300, 150)
