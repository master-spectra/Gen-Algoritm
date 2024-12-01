import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import List, Dict, Any, Optional, Union, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Import Population type from core module
from ..core.population import Population
from ..core.fitness_functions import rastrigin_function

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

    def plot_3d_fitness_landscape(self, fitness_function, bounds, resolution=50):
        """
        Создает интерактивную 3D визуализацию ландшафта фитнес-функции

        Args:
            fitness_function: Функция приспособленн��сти
            bounds: Границы пространства поиска [(x_min, x_max), (y_min, y_max)]
            resolution: Разрешение сетки
        """
        x = np.linspace(bounds[0][0], bounds[0][1], resolution)
        y = np.linspace(bounds[1][0], bounds[1][1], resolution)
        X, Y = np.meshgrid(x, y)

        # Вычисляем значения фитнес-функции
        Z = np.zeros((resolution, resolution))
        for i in range(resolution):
            for j in range(resolution):
                Z[i,j] = -fitness_function(np.array([X[i,j]]), np.array([Y[i,j]]))

        # Создаем 3D поверхность
        fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])

        # Настраиваем внешний вид
        fig.update_layout(
            title='Fitness Landscape',
            scene = dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Fitness',
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=800,
            height=800
        )

        # Сохраняем визуализацию
        fig.write_html(os.path.join(self.output_dir, 'fitness_landscape_3d.html'))

    def plot_3d_population_evolution(self, populations: List[Population],
                                   fitness_function,
                                   bounds: List[Tuple[float, float]],
                                   generation: int) -> None:
        """
        Создает интерактивную 3D визуализацию популяций на ландшафте фитнес-функции

        Args:
            populations: Список популяций
            fitness_function: Функция приспособленности
            bounds: Границы пространства поиска [(x_min, x_max), (y_min, y_max)]
            generation: Номер текущего поколения
        """
        # Создаем базовый ландшафт
        x = np.linspace(bounds[0][0], bounds[0][1], 50)
        y = np.linspace(bounds[1][0], bounds[1][1], 50)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros((50, 50))

        # Используем rastrigin_function вместо fitness_function.evaluate
        for i in range(50):
            for j in range(50):
                Z[i,j] = rastrigin_function(X[i,j], Y[i,j])

        # Создаем фигуру
        fig = go.Figure()

        # Добавляем поверхность ландшафта
        fig.add_trace(go.Surface(x=X, y=Y, z=Z, opacity=0.8,
                               colorscale='Viridis',
                               name='Fitness Landscape'))

        # Добавляем точки популяций
        for pop_idx, population in enumerate(populations):
            x_vals = [ind.genes['x'].value for ind in population.chromosomes]
            y_vals = [ind.genes['y'].value for ind in population.chromosomes]
            z_vals = [-ind.fitness for ind in population.chromosomes]

            fig.add_trace(go.Scatter3d(
                x=x_vals,
                y=y_vals,
                z=z_vals,
                mode='markers',
                marker=dict(
                    size=5,
                    color=z_vals,
                    colorscale='Plasma',
                ),
                name=f'Population {pop_idx + 1}'
            ))

        # Настройка внешнего вида
        fig.update_layout(
            title=f'Population Distribution (Generation {generation})',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Fitness',
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=1000,
            height=800
        )

        # Сохраняем визуализацию
        fig.write_html(os.path.join(self.output_dir,
                                   f'population_evolution_3d_gen_{generation}.html'))

    def create_evolution_animation(self, evolution_history: List[Dict],
                                   fitness_function,
                                   bounds: List[Tuple[float, float]]) -> None:
        """
        Создает анимацию эволюционного процесса

        Args:
            evolution_history: История эволюции по поколениям
            fitness_function: Функция приспособленности
            bounds: Границы пространства поиска
        """
        frames = []

        # Создаем базовый ландшафт
        x = np.linspace(bounds[0][0], bounds[0][1], 50)
        y = np.linspace(bounds[1][0], bounds[1][1], 50)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros((50, 50))

        for i in range(50):
            for j in range(50):
                Z[i,j] = rastrigin_function(X[i,j], Y[i,j])

        # Создаем кадры анимации
        for gen_data in evolution_history:
            frame_data = []

            # Добавляем поверхность ландшафта
            frame_data.append(
                go.Surface(x=X, y=Y, z=Z, opacity=0.8,
                          colorscale='Viridis',
                          name='Fitness Landscape')
            )

            # Добавляем точки популяций
            for pop_idx, population in enumerate(gen_data['populations']):
                x_vals = [ind.genes['x'].value for ind in population.chromosomes]
                y_vals = [ind.genes['y'].value for ind in population.chromosomes]
                z_vals = [-ind.fitness for ind in population.chromosomes]

                frame_data.append(
                    go.Scatter3d(
                        x=x_vals,
                        y=y_vals,
                        z=z_vals,
                        mode='markers',
                        marker=dict(
                            size=5,
                            color=z_vals,
                            colorscale='Plasma',
                        ),
                        name=f'Population {pop_idx + 1}'
                    )
                )

            frames.append(go.Frame(data=frame_data,
                                 name=f'Generation {gen_data["generation"]}'))

        # Создаем финальную анимацию
        fig = go.Figure(
            data=frames[0].data,
            layout=go.Layout(
                title="Evolution Process Animation",
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'buttons': [{
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 500, 'redraw': True},
                            'fromcurrent': True,
                        }]
                    }]
                }],
                sliders=[{
                    'currentvalue': {'prefix': 'Generation: '},
                    'steps': [
                        {
                            'method': 'animate',
                            'label': f'{i}',
                            'args': [[f'Generation {i}']]
                        }
                        for i in range(len(frames))
                    ]
                }]
            ),
            frames=frames
        )

        # Сохраняем анимацию
        fig.write_html(os.path.join(self.output_dir, 'evolution_animation.html'))

    def create_advanced_dashboard(self, metadata: Dict[str, Any]) -> None:
        """Создает расширенный интерактивный дашборд"""

        # Создаем подграфики
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Fitness Trends',
                'Population Diversity',
                'Parameter Evolution',
                'Performance Metrics',
                'Population Distribution',
                'Convergence Analysis'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'violin'}, {'type': 'scatter3d'}]
            ]
        )

        generations = list(range(len(metadata['fitness']['best'])))

        # 1. Fitness Trends
        fig.add_trace(
            go.Scatter(x=generations,
                      y=metadata['fitness']['best'],
                      name='Best Fitness',
                      mode='lines+markers'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=generations,
                      y=metadata['fitness']['average'],
                      name='Average Fitness',
                      mode='lines+markers'),
            row=1, col=1
        )

        # 2. Population Diversity
        fig.add_trace(
            go.Scatter(x=generations,
                      y=metadata['diversity'],
                      name='Diversity',
                      mode='lines+markers'),
            row=1, col=2
        )

        # 3. Parameter Evolution
        fig.add_trace(
            go.Scatter(x=generations,
                      y=metadata['parameters']['mutation_rates'],
                      name='Mutation Rate',
                      mode='lines'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=generations,
                      y=metadata['parameters']['crossover_rates'],
                      name='Crossover Rate',
                      mode='lines'),
            row=2, col=1
        )

        # 4. Performance Metrics
        fig.add_trace(
            go.Scatter(x=generations,
                      y=metadata['performance']['generation_times'],
                      name='Generation Time',
                      mode='lines+markers'),
            row=2, col=2
        )

        # 5. Population Distribution (Violin Plot)
        if 'population_stats' in metadata:
            fig.add_trace(
                go.Violin(y=metadata['population_stats']['fitness_distribution'],
                         name='Fitness Distribution'),
                row=3, col=1
            )

        # Настройка макета
        fig.update_layout(
            height=1200,
            width=1600,
            title_text="Advanced Evolution Dashboard",
            showlegend=True,
            template='plotly_dark'
        )

        # Добавляем интерактивные элементы
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(
                            args=[{"visible": [True] * len(fig.data)}],
                            label="Show All",
                            method="update"
                        ),
                        dict(
                            args=[{"visible": [i < 3 for i in range(len(fig.data))]}],
                            label="Fitness Only",
                            method="update"
                        )
                    ]),
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.11,
                    xanchor="left",
                    y=1.1,
                    yanchor="top"
                ),
            ]
        )

        # Сохраняем дашборд
        fig.write_html(os.path.join(self.output_dir, 'advanced_dashboard.html'))

    # ... rest of the class implementation ...
