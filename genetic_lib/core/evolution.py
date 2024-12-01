import asyncio
from typing import List, Type, Optional, Dict, Any, Tuple, Callable, TypeVar, Generic
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from dataclasses import dataclass, field
import json
import time
import os
import random
from functools import partial

from .chromosome import Chromosome
from .population import Population
from .fitness import FitnessFunction, MultiFitness
from ..operators.selection import SelectionOperator, AdaptiveSelection
from ..operators.crossover import CrossoverOperator, AdaptiveCrossover
from ..operators.mutation import MutationOperator, AdaptiveMutation
from ..utils.visualization import EvolutionVisualizer
from ..utils.persistence import StatePersistence, AutoCheckpoint
from .pareto_front import ParetoFront

C = TypeVar('C', bound=Chromosome)

@dataclass
class EvolutionConfig:
    """Расширенная конфигурация эволюционного процесса"""
    # Базовые параметры
    initial_population_size: int = 100
    min_population_size: int = 50
    max_population_size: int = 200
    population_size_adaptation_rate: float = 0.1
    num_populations: int = 4
    elite_size: int = 2

    # Параметры давления отбора
    initial_selection_pressure: float = 2.0
    min_selection_pressure: float = 1.2
    max_selection_pressure: float = 4.0
    selection_pressure_adaptation_rate: float = 0.1
    tournament_size: int = 3

    # Адаптивные параметры
    initial_mutation_rate: float = 0.1
    min_mutation_rate: float = 0.01
    max_mutation_rate: float = 0.3
    mutation_adaptation_rate: float = 0.1

    initial_crossover_rate: float = 0.8
    min_crossover_rate: float = 0.5
    max_crossover_rate: float = 0.95
    crossover_adaptation_rate: float = 0.1

    # Параметры миграции
    migration_interval: int = 10
    migration_size: int = 2

    # Параметры выполнения
    max_generations: int = 100
    max_workers: int = 4
    convergence_threshold: float = 1e-6
    stagnation_limit: int = 20

    # Метаданные
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Параметры для вывода результатов
    output_dir: str = "evolution_results"
    save_checkpoints: bool = True
    checkpoint_frequency: int = 10
    max_checkpoints: int = 5

class AdaptiveParameters:
    """Класс для управления адаптивными параметрами"""

    def __init__(self, config: EvolutionConfig):
        self.config = config
        # Параметры мутации и скрещивания
        self.mutation_rates = [config.initial_mutation_rate] * config.num_populations
        self.crossover_rates = [config.initial_crossover_rate] * config.num_populations

        # Размеры популяций и давление отбора
        self.population_sizes = [config.initial_population_size] * config.num_populations
        self.selection_pressures = [config.initial_selection_pressure] * config.num_populations

        # История успешности
        self.success_history = []
        self.stagnation_counters = [0] * config.num_populations

    def update(self, population_index: int,
              fitness_improvement: float,
              diversity: float,
              convergence_speed: float) -> None:
        """
        Обновление параметров на основе результатов

        Args:
            population_index: Индекс популяции
            fitness_improvement: Улучшение фитнеса
            diversity: Разнообразие популяции
            convergence_speed: Скорость сходимости (изменение среднего фитнеса)
        """
        # Обновляем счетчик стагнации
        if fitness_improvement <= 0:
            self.stagnation_counters[population_index] += 1
        else:
            self.stagnation_counters[population_index] = 0

        # Адаптация размера популяции
        self._adapt_population_size(population_index, convergence_speed, diversity)

        # Адаптация давления отбора
        self._adapt_selection_pressure(population_index, fitness_improvement, diversity)

        # Адаптация скорости мутации
        if diversity < 0.1:  # Низкое разнообразие
            self.mutation_rates[population_index] *= (1 + self.config.mutation_adaptation_rate)
        elif diversity > 0.5:  # Высокое разнообразие
            self.mutation_rates[population_index] *= (1 - self.config.mutation_adaptation_rate)

        # Адаптация скорости скрещивания
        if fitness_improvement > 0:
            self.crossover_rates[population_index] *= (1 + self.config.crossover_adaptation_rate)
        else:
            self.crossover_rates[population_index] *= (1 - self.config.crossover_adaptation_rate)

        # Ограничение значений
        self._clip_parameters(population_index)

    def _adapt_population_size(self,
                             population_index: int,
                             convergence_speed: float,
                             diversity: float) -> None:
        """Адаптация размера популяции"""
        current_size = self.population_sizes[population_index]

        # Увеличиваем размер при медленной сходимости или низком разнообразии
        if convergence_speed < 0.01 or diversity < 0.1:
            adjustment = current_size * self.config.population_size_adaptation_rate
            self.population_sizes[population_index] = int(current_size + adjustment)

        # Уменьшаем размер при быстрой сходимости и высоком разнообразии
        elif convergence_speed > 0.05 and diversity > 0.3:
            adjustment = current_size * self.config.population_size_adaptation_rate
            self.population_sizes[population_index] = int(current_size - adjustment)

    def _adapt_selection_pressure(self,
                                population_index: int,
                                fitness_improvement: float,
                                diversity: float) -> None:
        """Адаптация давления отбора"""
        current_pressure = self.selection_pressures[population_index]

        # Увеличиваем давление при застое и высоком разнообразии
        if self.stagnation_counters[population_index] > 5 and diversity > 0.3:
            self.selection_pressures[population_index] = current_pressure * (
                1 + self.config.selection_pressure_adaptation_rate
            )

        # Уменьшаем давление при хорошем улучшении или низком разнообразии
        elif fitness_improvement > 0.1 or diversity < 0.1:
            self.selection_pressures[population_index] = current_pressure * (
                1 - self.config.selection_pressure_adaptation_rate
            )

    def _clip_parameters(self, population_index: int) -> None:
        """Ограничение параметров в допустимых пределах"""
        # Ограничение размера популяции
        self.population_sizes[population_index] = np.clip(
            self.population_sizes[population_index],
            self.config.min_population_size,
            self.config.max_population_size
        )

        # Ограничение давления отбора
        self.selection_pressures[population_index] = np.clip(
            self.selection_pressures[population_index],
            self.config.min_selection_pressure,
            self.config.max_selection_pressure
        )

        # Ограничение скорости мутации
        self.mutation_rates[population_index] = np.clip(
            self.mutation_rates[population_index],
            self.config.min_mutation_rate,
            self.config.max_mutation_rate
        )

        # Ограничение скорости скрещивания
        self.crossover_rates[population_index] = np.clip(
            self.crossover_rates[population_index],
            self.config.min_crossover_rate,
            self.config.max_crossover_rate
        )

class ExtendedMetadata:
    """Расширенный класс для работы с метаданными"""

    def __init__(self):
        self.data = {
            'start_time': time.time(),
            'generation': 0,
            'populations': [],
            'fitness': {
                'best': [],
                'average': [],
                'worst': []
            },
            'diversity': [],
            'parameters': {
                'mutation_rates': [],
                'crossover_rates': [],
                'population_sizes': [],
                'selection_pressures': []
            },
            'performance': {
                'generation_times': [],
                'evaluation_times': [],
                'selection_times': [],
                'crossover_times': []
            },
            'convergence': {
                'stagnation_counter': 0,
                'best_fitness_history': [],
                'convergence_speed': []
            },
            'operators': {
                'selection_success': {},
                'crossover_success': {}
            }
        }

    def _prepare_for_json(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        return obj

    def update_generation(self,
                         populations: List[Population],
                         adaptive_params: AdaptiveParameters,
                         generation_time: float) -> None:
        """Обновление метаданных поколения"""
        self.data['generation'] += 1

        # Сбор статистики по популяциям
        all_individuals = [ind for pop in populations for ind in pop.chromosomes]
        fitnesses = [ind.fitness for ind in all_individuals]

        # Обновляем фитнес
        current_best = max(fitnesses)
        current_avg = np.mean(fitnesses)
        current_worst = min(fitnesses)

        self.data['fitness']['best'].append(current_best)
        self.data['fitness']['average'].append(current_avg)
        self.data['fitness']['worst'].append(current_worst)

        # Вычисляем скорость сходимости
        if len(self.data['fitness']['average']) > 1:
            prev_avg = self.data['fitness']['average'][-2]
            convergence_speed = abs(current_avg - prev_avg) / abs(prev_avg) if prev_avg != 0 else 0
            self.data['convergence']['convergence_speed'].append(convergence_speed)
        else:
            self.data['convergence']['convergence_speed'].append(0)

        # Разнообразие
        diversity = np.mean([pop.calculate_diversity() for pop in populations])
        self.data['diversity'].append(diversity)

        # Параметры
        self.data['parameters']['mutation_rates'].append(
            adaptive_params.mutation_rates.copy()
        )
        self.data['parameters']['crossover_rates'].append(
            adaptive_params.crossover_rates.copy()
        )
        self.data['parameters']['population_sizes'].append(
            adaptive_params.population_sizes.copy()
        )
        self.data['parameters']['selection_pressures'].append(
            adaptive_params.selection_pressures.copy()
        )

        # Время выполнения
        self.data['performance']['generation_times'].append(generation_time)

        # Проверка стагнации
        if len(self.data['fitness']['best']) > 1:
            if self.data['fitness']['best'][-1] <= self.data['fitness']['best'][-2]:
                self.data['convergence']['stagnation_counter'] += 1
            else:
                self.data['convergence']['stagnation_counter'] = 0

    def save(self, filename: str) -> None:
        """Сохранение метаданных в файл"""
        with open(filename, 'w') as f:
            json_data = self._prepare_for_json(self.data)
            json.dump(json_data, f, indent=4)

class Evolution(Generic[C]):
    """Расширенный класс эволюционного процесса"""

    def __init__(self,
                 chromosome_class: Type[C],
                 fitness_function: FitnessFunction,
                 selection_operator: SelectionOperator,
                 crossover_operator: CrossoverOperator,
                 mutation_operator: MutationOperator,
                 config: EvolutionConfig = None):
        """
        Args:
            chromosome_class: Класс хромосомы
            fitness_function: Функция приспособленности
            selection_operator: Оператор селекции
            crossover_operator: Оператор скрещивания
            mutation_operator: Оператор мутации
            config: Конфигурация эволюции
        """
        self.chromosome_class = chromosome_class
        self.fitness_function = fitness_function
        self.selection_operator = selection_operator
        self.crossover_operator = crossover_operator
        self.mutation_operator = mutation_operator
        self.config = config or EvolutionConfig()

        # Создаем директории для результатов
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Инициализируем популяции (пустые)
        self.populations = []

        # Инициализируем адаптивные параметры
        self.adaptive_params = AdaptiveParameters(self.config)

        # Инициализируем метаданные
        self.metadata = ExtendedMetadata()

        # Инициализируем визуализатор
        self.visualizer = EvolutionVisualizer(
            os.path.join(self.config.output_dir, "plots")
        )

        # Инициализируем систему сохранения
        self.persistence = StatePersistence(
            os.path.join(self.config.output_dir, "states")
        )

        # Настраиваем автосохранение
        if self.config.save_checkpoints:
            self.checkpoint_manager = AutoCheckpoint(
                self.persistence,
                self.config.checkpoint_frequency,
                self.config.max_checkpoints
            )
        else:
            self.checkpoint_manager = None

        # Пул процессов
        self.executor = ProcessPoolExecutor(max_workers=self.config.max_workers)

        # Семафор для ограничения параллельных вычислений
        self._sem = asyncio.Semaphore(self.config.max_workers)

    async def initialize(self) -> None:
        """Асинхронная инициализация популяций"""
        print("Initializing populations...")

        # Создаем популяции асинхронно
        init_tasks = []
        for _ in range(self.config.num_populations):
            init_tasks.append(self._initialize_population(self.config.initial_population_size))

        # Ждем создания всех популяций
        self.populations = await asyncio.gather(*init_tasks)

        print(f"Created {len(self.populations)} populations")

        # Оцениваем начальные популяции
        await self._evaluate_all_populations()

        print("Initialization complete")

    async def _initialize_population(self, size: int) -> Population:
        """Асинхронная инициализация одной популяции"""
        return Population.create_random(
            size=size,
            chromosome_factory=self.chromosome_class,
            metadata={'generation': 0}
        )

    async def _create_individual(self) -> Chromosome:
        """Асинхронное создание одной особи"""
        # Используем семафор для ограничения параллельных вычислений
        async with self._sem:
            individual = self.chromosome_class()
            individual.randomize()  # Предполагаем, что этот метод быстрый и не требует асинхронности
            return individual

    async def _evaluate_individual(self, individual: C) -> None:
        """Асинхронная оценка одной особи"""
        individual.fitness = await self.fitness_function.evaluate(individual)

    async def _evaluate_individuals(self, individuals: List[C]) -> None:
        """Асинхронная оценка группы особей"""
        tasks = [self._evaluate_individual(ind) for ind in individuals]
        await asyncio.gather(*tasks)

    async def _evaluate_all_populations(self) -> None:
        """Асинхронная оценка всех популяций"""
        evaluation_start = time.time()

        # Собираем всех индивидов для оценки
        all_individuals = [
            ind for pop in self.populations
            for ind in pop.chromosomes
        ]

        # Оцениваем всех индивидов
        await self._evaluate_individuals(all_individuals)

        # Обновляем метрики времени
        evaluation_time = time.time() - evaluation_start
        self.metadata.data['performance']['evaluation_times'].append(evaluation_time)

    async def evolve(self) -> Optional[Chromosome]:
        """Основной метод эволюции с расширенной функциональностью"""
        try:
            await self.initialize()

            while (self.metadata.data['generation'] < self.config.max_generations and
                   self.metadata.data['convergence']['stagnation_counter'] <
                   self.config.stagnation_limit):

                generation_start = time.time()

                # Параллельная эволюция популяций
                evolution_tasks = []
                for i, population in enumerate(self.populations):
                    evolution_tasks.append(
                        self._evolve_population(
                            population,
                            i,
                            self.adaptive_params.mutation_rates[i],
                            self.adaptive_params.crossover_rates[i]
                        )
                    )

                # Ждем завершения эволюции всех популяций
                await asyncio.gather(*evolution_tasks)

                # Миграция между популяциями
                await self._migrate()

                # Обновление метаданных и сохранение состояния
                await self._update_and_save_state(generation_start)

                # Проверка сходимости
                if await self._check_convergence():
                    break

                self.metadata.data['generation'] += 1

            # Получаем лучшую особь и создаем отчеты
            best_individual = await self._finalize_evolution()
            return best_individual

        except Exception as e:
            print(f"Error during evolution: {e}")
            await self._handle_evolution_error()
            raise
        finally:
            self.executor.shutdown()

    async def _migrate(self) -> None:
        """Асинхронная миграция между популяциями"""
        if (self.metadata.data['generation'] + 1) % self.config.migration_interval != 0:
            return

        migration_tasks = []
        for i, source_pop in enumerate(self.populations):
            # Определяем целевую популяцию (следующая по кругу)
            target_idx = (i + 1) % len(self.populations)
            migration_tasks.append(
                self._perform_migration(source_pop, self.populations[target_idx])
            )

        await asyncio.gather(*migration_tasks)

    async def _perform_migration(self, source_pop: Population, target_pop: Population) -> None:
        """Асинхронное выполнение миграции между двумя популяциями"""
        async with self._sem:
            # Выбираем лучших индивидов для миграции
            migrants = sorted(
                source_pop.chromosomes,
                key=lambda x: x.fitness,
                reverse=True
            )[:self.config.migration_size]

            # Заменяем худших индивидов в целевой популяции
            target_pop.chromosomes.sort(key=lambda x: x.fitness)
            target_pop.chromosomes[:self.config.migration_size] = [
                migrant.copy() for migrant in migrants
            ]

    async def _update_and_save_state(self, generation_start: float) -> None:
        """Асинхронное обновление и сохранение состояния"""
        # Обновление метаданных
        generation_time = time.time() - generation_start
        self.metadata.update_generation(
            self.populations,
            self.adaptive_params,
            generation_time
        )

        # Автосохранение
        if self.checkpoint_manager:
            await self._async_save_checkpoint()

    async def _async_save_checkpoint(self) -> None:
        """Асинхронное сохранение контрольной точки"""
        async with self._sem:
            self.checkpoint_manager.update(self)

    async def _check_convergence(self) -> bool:
        """Асинхронная проверка сходимости"""
        if len(self.metadata.data['fitness']['best']) < 2:
            return False

        current_best = self.metadata.data['fitness']['best'][-1]
        prev_best = self.metadata.data['fitness']['best'][-2]

        # Проверяем улучшение
        improvement = abs(current_best - prev_best)
        return improvement < self.config.convergence_threshold

    async def _finalize_evolution(self) -> Optional[Chromosome]:
        """Асинхронное завершение эволюции"""
        # Получаем лучшую особь
        best_individual = self._get_best_individual()

        # Создаем итоговые визуализации и отчеты
        await self._create_final_reports()

        return best_individual

    async def _create_final_reports(self) -> None:
        """Create final reports and visualizations"""
        # Save final metadata
        self.metadata.save(os.path.join(self.config.output_dir, 'final_metadata.json'))

        # Create visualizations if visualizer is available
        if hasattr(self, 'visualizer'):
            # Plot fitness trends
            self.visualizer.plot_fitness_trends(self.metadata.data)

            # Plot parameter evolution
            self.visualizer.plot_parameter_evolution(self.metadata.data)

            # Create interactive dashboard
            self.visualizer.create_interactive_dashboard(self.metadata.data)

            # Create statistical report
            stats_df = self.visualizer.create_statistical_report(self.metadata.data)
            stats_df.to_csv(os.path.join(self.config.output_dir, 'final_statistics.csv'))

            # Plot population distribution
            self.visualizer.plot_population_distribution(self.populations)

            # If we have multiple objectives, create Pareto front visualization
            if isinstance(self.fitness_function, MultiFitness):
                objective_names = [obj.name for obj in self.fitness_function.objectives]
                self.visualizer.plot_pareto_front(
                    self.pareto_front.solutions if hasattr(self, 'pareto_front') else [],
                    objective_names
                )

        # Save final state
        if self.persistence:
            self.persistence.create_checkpoint(self)

    async def _handle_evolution_error(self) -> None:
        """Асинхронная обработка ошибок эволюции"""
        if self.checkpoint_manager:
            async with self._sem:
                self.persistence.create_checkpoint(self)

    async def _evolve_population(self,
                               population: Population,
                               pop_index: int,
                               mutation_rate: float,
                               crossover_rate: float) -> None:
        """Асинхронная эволюция одной популяции"""
        # Получаем текущие адаптивные параметры
        current_size = self.adaptive_params.population_sizes[pop_index]
        selection_pressure = self.adaptive_params.selection_pressures[pop_index]

        # Обновляем размер популяции если нужно
        if current_size != len(population.chromosomes):
            await self._resize_population(population, current_size)

        # Сохраняем элиту
        elite_size = max(1, int(len(population.chromosomes) * 0.05))
        elite = population.get_elite(elite_size)

        # Создаем новое поколение
        new_individuals = []
        num_offspring = current_size - len(elite)
        batch_size = max(2, num_offspring // self.config.max_workers)

        # Создаем потомков пакетами
        offspring_tasks = []
        for i in range(0, num_offspring, batch_size):
            batch_size_adjusted = min(batch_size, num_offspring - i)
            offspring_tasks.append(
                self._create_offspring_batch(
                    population.chromosomes,
                    batch_size_adjusted,
                    mutation_rate,
                    crossover_rate,
                    selection_pressure
                )
            )

        # Ждем создания всех потомков
        offspring_batches = await asyncio.gather(*offspring_tasks)
        for batch in offspring_batches:
            new_individuals.extend(batch)

        # Оценка новых особей
        await self._evaluate_individuals(new_individuals)

        # Обновление популяции
        population.chromosomes = elite + new_individuals[:current_size - len(elite)]

        # Обновляем адаптивные параметры
        await self._update_adaptive_parameters(population, pop_index)

    async def _resize_population(self,
                               population: Population,
                               new_size: int) -> None:
        """Асинхронное изменение размера популяции"""
        current_size = len(population.chromosomes)

        if new_size > current_size:
            # Создаем новых особей
            additional = []
            for _ in range(new_size - current_size):
                individual = await self._create_individual()
                additional.append(individual)

            # Оцениваем новых особей
            await self._evaluate_individuals(additional)

            # Добавляем в популяцию
            population.chromosomes.extend(additional)
        else:
            # Оставляем лучших особей
            population.sort_by_fitness(reverse=True)
            population.chromosomes = population.chromosomes[:new_size]

    async def _create_offspring_batch(self,
                                    parents_pool: List[Chromosome],
                                    batch_size: int,
                                    mutation_rate: float,
                                    crossover_rate: float,
                                    selection_pressure: float) -> List[Chromosome]:
        """Асинхронное создание пакета потомков"""
        async with self._sem:
            offspring = []
            for _ in range(0, batch_size, 2):
                # Селекция родителей с учетом давления отбора
                if isinstance(self.selection_operator, AdaptiveSelection):
                    self.selection_operator.set_selection_pressure(selection_pressure)
                parents = self.selection_operator.select(parents_pool, num_parents=2)

                # Скрещивание
                if random.random() < crossover_rate:
                    children = self.crossover_operator.crossover(*parents)
                else:
                    children = parents[0].copy(), parents[1].copy()

                # Мутация
                for child in children:
                    mutated = self.mutation_operator.mutate(child, mutation_rate)
                    offspring.append(mutated)

            return offspring[:batch_size]

    async def _update_adaptive_parameters(self,
                                        population: Population,
                                        pop_index: int) -> None:
        """Асинхронное обновление адаптивных параметров"""
        async with self._sem:
            if len(self.metadata.data['fitness']['best']) > 0:
                # Вычисляем улучшение фитнеса
                prev_best = self.metadata.data['fitness']['best'][-1]
                current_best = max(ind.fitness for ind in population.chromosomes)
                improvement = current_best - prev_best

                # Вычисляем разнообразие
                diversity = population.calculate_diversity()

                # Получаем скорость сходимости
                convergence_speed = self.metadata.data['convergence']['convergence_speed'][-1]

                # Обновляем параметры
                self.adaptive_params.update(
                    pop_index,
                    improvement,
                    diversity,
                    convergence_speed
                )

                # Обновление успешности операторов
                if isinstance(self.selection_operator, AdaptiveSelection):
                    self.selection_operator.update_success_rates(improvement)
                if isinstance(self.crossover_operator, AdaptiveCrossover):
                    self.crossover_operator.update_success_rates((improvement, improvement))
                if isinstance(self.mutation_operator, AdaptiveMutation):
                    self.mutation_operator.update_success_rates(improvement)

    def _get_best_individual(self) -> Optional[Chromosome]:
        """Получение лучшей особи из всех популяций"""
        all_individuals = [
            ind for pop in self.populations
            for ind in pop.chromosomes
        ]
        return max(all_individuals, key=lambda x: x.fitness) if all_individuals else None

    def save_metadata(self, filename: str) -> None:
        """Сохранение метаданных"""
        self.metadata.save(filename)

class MultiObjectiveEvolution(Evolution[C]):
    """Класс для многокритериальной оптимизации"""

    def __init__(self,
                 fitness_function: FitnessFunction,
                 selection_operator: SelectionOperator[C],
                 crossover_operator: CrossoverOperator[C],
                 mutation_operator: MutationOperator[C],
                 chromosome_factory: Callable[[], C],
                 population_size: int = 100,
                 elite_size: int = 10,
                 max_generations: int = 100,
                 target_fitness: Optional[float] = None,
                 max_stagnation: int = 20,
                 visualization: Optional[EvolutionVisualizer] = None):
        super().__init__(fitness_function, selection_operator, crossover_operator,
                        mutation_operator, chromosome_factory, population_size,
                        elite_size, max_generations, target_fitness,
                        max_stagnation, visualization)
        self.pareto_front = ParetoFront[C]()

    async def _evaluate_population(self, population: List[C]) -> None:
        """Оценка популяции"""
        for individual in population:
            individual.fitness = await self.fitness_function.evaluate(individual)

        # Обновляем фронт Парето
        self.pareto_front.update(population)

    def _update_metadata(self, generation: int, population: List[C]) -> None:
        """Обновление метаданных эволюции"""
        super()._update_metadata(generation, population)

        # Добавляем метаданные для многокритериальной оптимизации
        if 'multi_objective' not in self.metadata:
            self.metadata['multi_objective'] = {
                'pareto_front_size': [],
                'objective_ranges': [],
                'hypervolume': []
            }

        # Размер фронта Парето
        self.metadata['multi_objective']['pareto_front_size'].append(
            len(self.pareto_front.solutions)
        )

        # Диапазоны целевых функций
        objectives = self.pareto_front.get_normalized_objectives()
        if len(objectives) > 0:
            ranges = {
                'min': objectives.min(axis=0).tolist(),
                'max': objectives.max(axis=0).tolist()
            }
            self.metadata['multi_objective']['objective_ranges'].append(ranges)

            # Вычисляем гиперобъем (если возможно)
            try:
                hypervolume = self.pareto_front.calculate_hypervolume()
                self.metadata['multi_objective']['hypervolume'].append(hypervolume)
            except Exception:
                self.metadata['multi_objective']['hypervolume'].append(None)

    def _visualize_generation(self, generation: int) -> None:
        """Визуализация текущего состояния эволюции"""
        super()._visualize_generation(generation)

        if self.visualization and generation % self.visualization_frequency == 0:
            # Визуализируем пространство целевых функций
            objective_names = self.fitness_function.objectives
            if len(objective_names) == 2:
                self.visualization.plot_objective_space(
                    self.population,
                    self.pareto_front.solutions,
                    objective_names,
                    generation
                )

            # Визуализируем фронт Парето
            self.visualization.plot_pareto_front(
                self.pareto_front.solutions,
                objective_names,
                plot_type='2d' if len(objective_names) == 2 else '3d'
            )
