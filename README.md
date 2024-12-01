# Библиотека Генетических Алгоритмов

Мощная и гибкая библиотека на Python для генетической оптимизации с поддержкой многокритериальной оптимизации и оптимизации игровых стратегий. Библиотека предоставляет полный набор инструментов для реализации генетических алгоритмов с акцентом на простоту использования и расширяемость.

## Возможности

- Однокритериальная и многокритериальная оптимизация
- Оптимизация игровых стратегий
- Гибкое представление хромосом
- Различные операторы селекции (Турнирный, Рулеточный, Ранговый и др.)
- Множество методов скрещивания (Одноточечный, Двухточечный, Равномерный)
- Адаптивные операторы мутации
- Встроенные инструменты визуализации
- Асинхронный процесс эволюции
- Сохранение контрольных точек и состояния
- Реализация алгоритма NSGA-II для многокритериальной оптимизации
- Интерактивные 3D визуализации
- Анимация процесса эволюции
- Расширенные дашборды для анализа

## Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/master-spectra/Gen-Algoritm.git
cd genetic-lib
```

2. Установите необходимые зависимости:
```bash
pip install -r requirements.txt
```

## Примеры использования

### Оптимизация функции Растригина

```python
from genetic_lib.core.chromosome import FloatGene, Chromosome
from genetic_lib.core.evolution import Evolution, EvolutionConfig
from genetic_lib.core.fitness_functions import RastriginFunction
from genetic_lib.operators.selection import TournamentSelection
from genetic_lib.operators.crossover import UniformCrossover
from genetic_lib.operators.mutation import GaussianMutation

# Определяем хромосому
class OptimizationChromosome(Chromosome):
    def __init__(self):
        super().__init__()
        self.genes = {
            'x': FloatGene(-5.0, 5.0),
            'y': FloatGene(-5.0, 5.0)
        }

# Настраиваем и запускаем оптимизацию
config = EvolutionConfig(
    initial_population_size=100,
    elite_size=5,
    max_generations=50
)

evolution = Evolution(
    chromosome_class=OptimizationChromosome,
    fitness_function=RastriginFunction(),
    selection_operator=TournamentSelection(tournament_size=3),
    crossover_operator=UniformCrossover(swap_probability=0.7),
    mutation_operator=GaussianMutation(scale=0.2),
    config=config
)

best_solution = await evolution.evolve()
```

### Оптимизация игровой стратегии

```python
from genetic_lib.core.base_types import GameChromosome
from genetic_lib.core.evolution import Evolution, EvolutionConfig
from genetic_lib.core.fitness import FitnessFunction

class HunterChromosome(GameChromosome):
    def __init__(self):
        super().__init__()
        self.genes = {
            'w1': FloatGene(-1.0, 1.0),  # Веса нейронной сети
            'w2': FloatGene(-1.0, 1.0),
            'b1': FloatGene(-1.0, 1.0),
            'b2': FloatGene(-1.0, 1.0)
        }

# Настраиваем эволюцию для игры
config = EvolutionConfig(
    initial_population_size=50,
    elite_size=3,
    max_generations=30
)

evolution = Evolution(
    chromosome_class=HunterChromosome,
    fitness_function=GameFitness(),
    selection_operator=TournamentSelection(tournament_size=3),
    crossover_operator=UniformCrossover(swap_probability=0.7),
    mutation_operator=GaussianMutation(scale=0.2),
    config=config
)

best_strategy = await evolution.evolve()
```

## Структура проекта

```
genetic_lib/
├── core/                    # Основные компоненты
│   ├── base_types.py       # Базовые типы хромосом
│   ├── chromosome.py       # Базовые классы хромосом
│   ├── evolution.py        # Движок эволюции
│   ├── fitness.py          # Интерфейсы функций приспособленности
│   ├── population.py       # Управление популяцией
│   └── multi_objective.py  # Многокритериальная оптимизация
├── operators/              # Генетические операторы
│   ├── selection.py        # Операторы селекции
│   ├── crossover.py        # Операторы скрещивания
│   └── mutation.py         # Операторы мутации
└── utils/                  # Вспомогательные функции
    ├── visualization.py    # Построение графиков и визуализация
    └── persistence.py      # Сохранение состояния
```

## Визуализация

Библиотека предоставляет богатые возможности визуализации:
- Интерактивные 3D графики ландшафта фитнес-функции
- Анимация процесса эволюции
- Визуализация игрового процесса
- Расширенные дашборды с метриками
- Графики распределения популяций
- Визуализация фронта Парето для многокритериальной оптимизации

## Примеры

В директории `examples/` можно найти примеры использования:
- Оптимизация функций (`function_optimization.py`)
- Оптимизация игровых стратегий (`game_optimization.py`)
- Многокритериальная оптимизация
- Примеры визуализации

## Лицензия

MIT License

## Контрибьюция

Приветствуются pull request'ы! Для крупных изменений, пожалуйста, сначала создайте issue для обсуждения предлагаемых изменений.
