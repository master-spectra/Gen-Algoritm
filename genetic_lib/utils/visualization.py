import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import List, Dict, Any, Optional, Union
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Import Population type from core module
from ..core.population import Population

class EvolutionVisualizer:
    """Класс для визуализации процесса эволюции"""

    def __init__(self, output_dir: str = 'evolution_results'):
        """
        Инициализация визуализатора

        Args:
            output_dir: Директория для сохранения результатов
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Пробуем установить стиль seaborn, если он доступен
        try:
            import seaborn as sns
            plt.style.use('seaborn')
        except (ImportError, OSError):
            # Если seaborn недоступен, используем стандартный стиль
            plt.style.use('default')

        # Настройка стиля графиков
        plt.rcParams['figure.figsize'] = [10, 6]
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3

    def plot_fitness_trends(self, metadata: Dict[str, Any]) -> None:
        """Plot fitness trends over generations"""
        generations = range(len(metadata['fitness']['best']))

        plt.figure(figsize=(12, 6))
        plt.plot(generations, metadata['fitness']['best'], label='Best Fitness', marker='o')
        plt.plot(generations, metadata['fitness']['average'], label='Average Fitness', marker='s')
        plt.plot(generations, metadata['fitness']['worst'], label='Worst Fitness', marker='^')

        plt.xlabel('Generation')
        plt.ylabel('Fitness Value')
        plt.title('Fitness Trends Over Generations')
        plt.legend()
        plt.grid(True)

        plt.savefig(os.path.join(self.output_dir, 'fitness_trends.png'))
        plt.close()

    def plot_parameter_evolution(self, metadata: Dict[str, Any]) -> None:
        """Plot evolution of parameters over generations"""
        generations = range(len(metadata['parameters']['mutation_rates']))

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Parameter Evolution Over Generations')

        # Plot mutation rates
        axes[0, 0].plot(generations, metadata['parameters']['mutation_rates'])
        axes[0, 0].set_title('Mutation Rates')
        axes[0, 0].set_xlabel('Generation')
        axes[0, 0].set_ylabel('Rate')

        # Plot crossover rates
        axes[0, 1].plot(generations, metadata['parameters']['crossover_rates'])
        axes[0, 1].set_title('Crossover Rates')
        axes[0, 1].set_xlabel('Generation')
        axes[0, 1].set_ylabel('Rate')

        # Plot population sizes
        axes[1, 0].plot(generations, metadata['parameters']['population_sizes'])
        axes[1, 0].set_title('Population Sizes')
        axes[1, 0].set_xlabel('Generation')
        axes[1, 0].set_ylabel('Size')

        # Plot selection pressures
        axes[1, 1].plot(generations, metadata['parameters']['selection_pressures'])
        axes[1, 1].set_title('Selection Pressures')
        axes[1, 1].set_xlabel('Generation')
        axes[1, 1].set_ylabel('Pressure')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'parameter_evolution.png'))
        plt.close()

    def create_interactive_dashboard(self, metadata: Dict[str, Any]) -> None:
        """Create an interactive dashboard using plotly"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Fitness Trends',
                'Population Diversity',
                'Parameter Evolution',
                'Convergence Speed',
                'Performance Metrics',
                'Stagnation Counter'
            )
        )

        generations = list(range(len(metadata['fitness']['best'])))

        # Fitness trends
        fig.add_trace(
            go.Scatter(x=generations, y=metadata['fitness']['best'], name='Best Fitness'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=generations, y=metadata['fitness']['average'], name='Average Fitness'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=generations, y=metadata['fitness']['worst'], name='Worst Fitness'),
            row=1, col=1
        )

        # Diversity
        fig.add_trace(
            go.Scatter(x=generations, y=metadata['diversity'], name='Population Diversity'),
            row=1, col=2
        )

        # Parameters
        fig.add_trace(
            go.Scatter(x=generations, y=metadata['parameters']['mutation_rates'], name='Mutation Rate'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=generations, y=metadata['parameters']['crossover_rates'], name='Crossover Rate'),
            row=2, col=1
        )

        # Convergence speed
        fig.add_trace(
            go.Scatter(x=generations[1:], y=metadata['convergence']['convergence_speed'], name='Convergence Speed'),
            row=2, col=2
        )

        # Performance metrics
        fig.add_trace(
            go.Scatter(x=generations, y=metadata['performance']['generation_times'], name='Generation Time'),
            row=3, col=1
        )

        # Stagnation counter
        fig.add_trace(
            go.Scatter(x=generations, y=[metadata['convergence']['stagnation_counter']] * len(generations),
                      name='Stagnation Counter'),
            row=3, col=2
        )

        fig.update_layout(height=1000, width=1200, title_text="Evolution Dashboard")
        fig.write_html(os.path.join(self.output_dir, 'evolution_dashboard.html'))

    def create_statistical_report(self, metadata: Dict[str, Any]) -> pd.DataFrame:
        """Create a statistical report of the evolution process"""
        stats = {
            'Generation': list(range(len(metadata['fitness']['best']))),
            'Best_Fitness': metadata['fitness']['best'],
            'Average_Fitness': metadata['fitness']['average'],
            'Worst_Fitness': metadata['fitness']['worst'],
            'Population_Diversity': metadata['diversity'],
            'Mutation_Rate': metadata['parameters']['mutation_rates'],
            'Crossover_Rate': metadata['parameters']['crossover_rates'],
            'Population_Size': metadata['parameters']['population_sizes'],
            'Selection_Pressure': metadata['parameters']['selection_pressures'],
            'Generation_Time': metadata['performance']['generation_times'],
            'Stagnation_Counter': [metadata['convergence']['stagnation_counter']] * len(metadata['fitness']['best'])
        }

        return pd.DataFrame(stats)

    def plot_population_distribution(self, populations: List[Population]) -> None:
        """Plot the distribution of individuals in the populations"""
        all_fitness = []
        pop_labels = []

        for i, pop in enumerate(populations):
            fitness_values = [ind.fitness for ind in pop.chromosomes]
            all_fitness.extend(fitness_values)
            pop_labels.extend([f'Population {i+1}'] * len(fitness_values))

        plt.figure(figsize=(10, 6))
        plt.violinplot([all_fitness[i:i+len(populations[0].chromosomes)]
                       for i in range(0, len(all_fitness), len(populations[0].chromosomes))])

        plt.xlabel('Population')
        plt.ylabel('Fitness Value')
        plt.title('Distribution of Fitness Values Across Populations')
        plt.xticks(range(1, len(populations) + 1), [f'Pop {i+1}' for i in range(len(populations))])

        plt.savefig(os.path.join(self.output_dir, 'population_distribution.png'))
        plt.close()

    # ... rest of the class implementation ...
