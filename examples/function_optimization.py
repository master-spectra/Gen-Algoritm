"""
Пример оптимизации функции Растригина с помощью генетического алгоритма
"""
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import asyncio
import os
import sys
import random

# Добавляем путь к корневой директории проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genetic_lib.core.chromosome import FloatGene, Chromosome
from genetic_lib.core.evolution import Evolution, EvolutionConfig
from genetic_lib.core.fitness_functions import RastriginFunction, plot_rastrigin_landscape
from genetic_lib.operators.selection import TournamentSelection
from genetic_lib.operators.crossover import UniformCrossover
from genetic_lib.operators.mutation import GaussianMutation
from genetic_lib.utils.visualization import EvolutionVisualizer
import matplotlib.pyplot as plt

class OptimizationChromosome(Chromosome):
    """Хромосома для оптимизации функции"""

    def __init__(self):
        super().__init__()
        # Определяем гены для x и y в диапазоне [-5, 5]
        self.genes = {
            'x': FloatGene(-5.0, 5.0),
            'y': FloatGene(-5.0, 5.0)
        }

    def init_random(self) -> None:
        """Инициализация случайными значениями"""
        for gene_name, gene in self.genes.items():
            min_val = gene.min_value
            max_val = gene.max_value
            gene.value = random.uniform(min_val, max_val)

    def randomize(self) -> None:
        """Случайное изменение генов"""
        self.init_random()

    def crossover(self, other: 'OptimizationChromosome') -> tuple['OptimizationChromosome', 'OptimizationChromosome']:
        """
        Скрещивание двух хромосом с помощью арифметического кроссовера

        Args:
            other: Вторая хромосома для скрещивания

        Returns:
            Два потомка
        """
        child1, child2 = OptimizationChromosome(), OptimizationChromosome()
        alpha = random.random()

        for gene_name in ['x', 'y']:
            v1 = self.genes[gene_name].value
            v2 = other.genes[gene_name].value
            child1.genes[gene_name].value = alpha * v1 + (1 - alpha) * v2
            child2.genes[gene_name].value = (1 - alpha) * v1 + alpha * v2

        return child1, child2


async def main():
    """Основная функция оптимизации"""

    # Создаем визуализатор
    visualizer = EvolutionVisualizer(output_dir='optimization_results')

    # Создаем конфигурацию
    config = EvolutionConfig(
        initial_population_size=100,
        elite_size=5,
        max_generations=50,
        output_dir='optimization_results'
    )

    # Настраиваем эволюцию
    evolution = Evolution(
        chromosome_class=OptimizationChromosome,
        fitness_function=RastriginFunction(),
        selection_operator=TournamentSelection(tournament_size=3),
        crossover_operator=UniformCrossover(swap_probability=0.7),
        mutation_operator=GaussianMutation(scale=0.2),
        config=config
    )

    print("Starting optimization...")
    best_solution = await evolution.evolve()

    # Получаем оптимальные значения
    x_opt = best_solution.genes['x'].value
    y_opt = best_solution.genes['y'].value
    fitness = -best_solution.fitness  # Конвертируем обратно в исходное значение

    print("\nOptimization Results:")
    print(f"Found minimum: f({x_opt:.4f}, {y_opt:.4f}) = {fitness:.4f}")
    print(f"Known global minimum: f(0, 0) = 0")

    # Визуализируем результат
    plot_rastrigin_landscape(
        x_opt=x_opt,
        y_opt=y_opt,
        title='Rastrigin Function Optimization Result',
        save_path='optimization_result.png'
    )


if __name__ == "__main__":
    # Визуализируем начальный ландшафт
    plot_rastrigin_landscape(save_path='rastrigin_landscape.png')

    # Запускаем оптимизацию
    asyncio.run(main())
