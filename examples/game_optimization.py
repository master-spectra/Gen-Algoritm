"""
Пример оптимизации стратегии для игры Охотник-Добыча с помощью генетического алгоритма
"""
import matplotlib
matplotlib.use('Agg')

import asyncio
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

# Добавляем путь к корневой директории проекта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genetic_lib.core.base_types import GameChromosome, FloatGene
from genetic_lib.core.evolution import Evolution, EvolutionConfig
from genetic_lib.core.fitness import FitnessFunction
from genetic_lib.core.population import Population
from genetic_lib.operators.selection import TournamentSelection
from genetic_lib.operators.crossover import UniformCrossover
from genetic_lib.operators.mutation import GaussianMutation
from genetic_lib.utils.visualization import EvolutionVisualizer

class GameEnvironment:
    """Игровое окружение Охотник-Добыча"""

    def __init__(self, size: int = 10):
        self.size = size
        self.reset()

    def reset(self) -> None:
        """Сброс игрового поля"""
        # Случайные начальные позиции
        self.hunter_pos = np.array([
            random.randint(0, self.size-1),
            random.randint(0, self.size-1)
        ])
        self.prey_pos = np.array([
            random.randint(0, self.size-1),
            random.randint(0, self.size-1)
        ])
        # Убеждаемся, что начальные позиции различны
        while np.array_equal(self.hunter_pos, self.prey_pos):
            self.prey_pos = np.array([
                random.randint(0, self.size-1),
                random.randint(0, self.size-1)
            ])

        self.steps = 0
        self.max_steps = self.size * 2

    def step(self, hunter_action: np.ndarray, prey_action: np.ndarray) -> Tuple[bool, int]:
        """
        Выполнить один шаг игры

        Args:
            hunter_action: Вектор движения охотника
            prey_action: Вектор движения добычи

        Returns:
            Tuple[bool, int]: (игра закончена, награда)
        """
        # Нормализуем векторы движения
        hunter_action = np.clip(hunter_action, -1, 1)
        prey_action = np.clip(prey_action, -1, 1)

        # Обновляем позиции
        self.hunter_pos += np.round(hunter_action).astype(int)
        self.prey_pos += np.round(prey_action).astype(int)

        # Ограничиваем позиции полем
        self.hunter_pos = np.clip(self.hunter_pos, 0, self.size-1)
        self.prey_pos = np.clip(self.prey_pos, 0, self.size-1)

        self.steps += 1

        # Проверяем условия окончания
        if np.array_equal(self.hunter_pos, self.prey_pos):
            return True, 100 - self.steps  # Награда за поимку

        if self.steps >= self.max_steps:
            return True, -50  # Штраф за превышение лимита ходов

        return False, -1  # Штраф за каждый ход

class HunterChromosome(GameChromosome):
    """Хромосома для стратегии охотника"""

    def __init__(self):
        super().__init__()
        # Гены для весов нейронной сети
        self.genes = {
            'w1': FloatGene(-1.0, 1.0),  # Вес для расстояния по X
            'w2': FloatGene(-1.0, 1.0),  # Вес для расстояния по Y
            'b1': FloatGene(-1.0, 1.0),  # Смещение для X
            'b2': FloatGene(-1.0, 1.0)   # Смещение для Y
        }

    def init_random(self) -> None:
        """Инициализация случайными значениями"""
        for gene in self.genes.values():
            gene.value = random.uniform(gene.min_value, gene.max_value)

    def randomize(self, strategy: str = 'uniform') -> None:
        """
        Случайное изменение генов с разными стратегиями

        Args:
            strategy: Стратегия рандомизации ('uniform', 'gaussian', 'cauchy')
        """
        if strategy == 'uniform':
            self.init_random()
        elif strategy == 'gaussian':
            for gene in self.genes.values():
                current = gene.value
                std = (gene.max_value - gene.min_value) * 0.1
                gene.value = np.clip(
                    current + random.gauss(0, std),
                    gene.min_value,
                    gene.max_value
                )
        elif strategy == 'cauchy':
            for gene in self.genes.values():
                current = gene.value
                scale = (gene.max_value - gene.min_value) * 0.1
                gene.value = np.clip(
                    current + random.uniform(-1, 1) * scale / (random.random() + 0.5),
                    gene.min_value,
                    gene.max_value
                )
        else:
            raise ValueError(f"Unknown randomization strategy: {strategy}")

    def crossover(self, other: 'HunterChromosome') -> tuple['HunterChromosome', 'HunterChromosome']:
        """
        Скрещивание двух хромосом

        Args:
            other: Вторая хромосома для скрещивания

        Returns:
            Два потомка
        """
        child1, child2 = HunterChromosome(), HunterChromosome()

        # Арифметический кроссовер
        alpha = random.random()
        for gene_name in self.genes:
            v1 = self.genes[gene_name].value
            v2 = other.genes[gene_name].value

            child1.genes[gene_name].value = alpha * v1 + (1 - alpha) * v2
            child2.genes[gene_name].value = (1 - alpha) * v1 + alpha * v2

        return child1, child2

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Получить действие на основе состояния

        Args:
            state: Вектор состояния [dx, dy] - разница в координатах

        Returns:
            np.ndarray: Вектор действия [ax, ay]
        """
        # Простая нейронная сеть для принятия решений
        action_x = np.tanh(state[0] * self.genes['w1'].value + self.genes['b1'].value)
        action_y = np.tanh(state[1] * self.genes['w2'].value + self.genes['b2'].value)
        return np.array([action_x, action_y])

class GameFitness(FitnessFunction):
    """Функция приспособленности для игры"""

    def __init__(self, num_episodes: int = 5):
        super().__init__(objectives=['maximize'])
        self.env = GameEnvironment()
        self.num_episodes = num_episodes

    async def evaluate(self, chromosome: HunterChromosome) -> float:
        """Оценка хромосомы"""
        total_reward = 0

        for _ in range(self.num_episodes):
            self.env.reset()
            done = False

            while not done:
                # Вычисляем разницу в позициях
                state = self.env.prey_pos - self.env.hunter_pos

                # Получаем действие охотника
                hunter_action = chromosome.get_action(state)

                # Простая стратегия для добычи - случайное движение
                prey_action = np.random.uniform(-1, 1, 2)

                # Выполняем шаг
                done, reward = self.env.step(hunter_action, prey_action)
                total_reward += reward

        return total_reward / self.num_episodes

async def main():
    """Основная функция оптимизации"""

    # Создаем визуализатор
    visualizer = EvolutionVisualizer(output_dir='game_optimization_results')

    # Создаем конфигурацию
    config = EvolutionConfig(
        initial_population_size=50,
        elite_size=3,
        max_generations=30,
        output_dir='game_optimization_results'
    )

    # Настраиваем эволюцию
    evolution = Evolution(
        chromosome_class=HunterChromosome,
        fitness_function=GameFitness(num_episodes=5),
        selection_operator=TournamentSelection(tournament_size=3),
        crossover_operator=UniformCrossover(swap_probability=0.7),
        mutation_operator=GaussianMutation(scale=0.2),
        config=config
    )

    print("Starting optimization...")
    best_solution = await evolution.evolve()

    # Визуализируем результат
    await visualize_game(best_solution)

async def visualize_game(hunter: HunterChromosome, num_steps: int = 20) -> None:
    """Визуализация игры с лучшей стратегией"""
    env = GameEnvironment()
    env.reset()

    # Создаем анимацию
    fig, ax = plt.subplots(figsize=(8, 8))
    frames = []

    for step in range(num_steps):
        # Очищаем текущий кадр
        ax.clear()

        # Отрисовываем поле
        ax.set_xlim(-1, env.size)
        ax.set_ylim(-1, env.size)
        ax.grid(True)

        # Отрисовываем охотника и добычу
        ax.plot(env.hunter_pos[0], env.hunter_pos[1], 'ro', label='Hunter', markersize=15)
        ax.plot(env.prey_pos[0], env.prey_pos[1], 'bo', label='Prey', markersize=10)

        ax.set_title(f'Step {step}')
        ax.legend()

        # Сохраняем кадр
        plt.savefig(f'game_optimization_results/step_{step:03d}.png')

        # Выполняем шаг
        state = env.prey_pos - env.hunter_pos
        hunter_action = hunter.get_action(state)
        prey_action = np.random.uniform(-1, 1, 2)

        done, _ = env.step(hunter_action, prey_action)
        if done:
            break

    plt.close()

if __name__ == "__main__":
    asyncio.run(main())
