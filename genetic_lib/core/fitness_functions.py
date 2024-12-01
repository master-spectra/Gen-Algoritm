"""
Модуль с реализациями различных функций приспособленности
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
from genetic_lib.core.fitness import FitnessFunction
from genetic_lib.core.chromosome import Chromosome

# Пробуем импортировать seaborn для улучшенной визуализации
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def rastrigin_function(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Вычисление значения функции Растригина

    f(x,y) = 20 + (x² - 10cos(2πx)) + (y² - 10cos(2πy))
    Глобальный минимум: f(0,0) = 0

    Args:
        X: Массив значений x
        Y: Массив значений y

    Returns:
        Массив значений функции Растригина
    """
    return 20 + (X**2 - 10*np.cos(2*np.pi*X)) + (Y**2 - 10*np.cos(2*np.pi*Y))


class RastriginFunction(FitnessFunction):
    """
    Класс для оптимизации функции Растригина
    Используется как тестовая функция для оптимизации
    """

    def __init__(self):
        super().__init__(objectives=['minimize'])

    async def evaluate(self, chromosome: Chromosome) -> float:
        """
        Оценка хромосомы с помощью функции Растригина

        Args:
            chromosome: Хромосома для оценки

        Returns:
            Значение функции приспособленности (отрицательное, т.к. ищем минимум)
        """
        x = chromosome.genes['x'].value
        y = chromosome.genes['y'].value
        return -rastrigin_function(np.array([x]), np.array([y]))[0]


def plot_rastrigin_landscape(x_opt: float = None,
                           y_opt: float = None,
                           title: str = 'Rastrigin Function Landscape',
                           save_path: str = None) -> None:
    """
    Визуализация ландшафта функции Растригина

    Args:
        x_opt: Оптимальное значение x (если есть)
        y_opt: Оптимальное значение y (если есть)
        title: Заголовок графика
        save_path: Путь для сохранения графика
    """
    # Настройка стиля
    if HAS_SEABORN:
        sns.set_style("whitegrid")
    else:
        plt.style.use('default')
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3

    # Создание данных
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = rastrigin_function(X, Y)

    # Создание графика
    fig, ax = plt.subplots(figsize=(10, 8))

    # Построение контурного графика
    contour = ax.contour(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(contour, label='f(x,y)')

    # Добавление точек оптимума, если они есть
    if x_opt is not None and y_opt is not None:
        ax.plot(x_opt, y_opt, 'r*', markersize=15, label='Found solution')
        ax.plot(0, 0, 'g*', markersize=15, label='Global minimum')
        ax.legend(loc='upper right')

    # Настройка осей и заголовка
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(title, fontsize=14, pad=20)

    # Добавление сетки
    ax.grid(True, alpha=0.3, linestyle='--')

    # Настройка внешнего вида
    fig.tight_layout()

    # Сохранение или отображение
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
