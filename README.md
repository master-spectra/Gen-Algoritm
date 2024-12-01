# Библиотека Генетических Алгоритмов

Мощная и гибкая библиотека на Python для генетической оптимизации с поддержкой многокритериальнй оптимизации. Библиотека предоставляет полный набор инструментов для реализации генетических алгоритмов с акцентом на простоту использования и расширяемость.

## Возможности

- Однокритериальная и многокритериальная оптимизация
- Гибкое представление хромосом
- Различные операторы селекции (Турнирный, Рулеточный, Ранговый и др.)
- Множество методов скрещивания (Одноточечный, Двухточечный, Равномерный)
- Адаптивные операторы мутации
- Встроенные инструменты визуализации
- Асинхронный процесс эволюции
- Сохранение контрольных точек и состояния
- Реализация алгоритма NSGA-II для многокритериальной оптимизации

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

## Быстрый старт

Вот простой пример оптимизации функции Растигина:

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
    initial_population_size=100,  # Начальный размер популяции
    elite_size=5,                # Количество элитных особей
    max_generations=50           # Максимальное число поколений
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

## Структура проекта

```
genetic_lib/
├── core/                    # Основные компоненты
│   ├── chromosome.py        # Базовые классы хромосом
│   ├── evolution.py         # Движок эволюции
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

## Основные компоненты

### Хромосома
Базовый класс для определения специфичных для задачи хромосом. Поддерживает различные типы генов:
- FloatGene: Для непрерывной оптимизации
- IntegerGene: Для дискретной оптимизации
- BinaryGene: Для бинарной оптимизации
- DiscreteGene: Для категориальной оптимизации

### Эволюция
Основной движок, управляющий генетическим алгоритмом. Особенности:
- Асинхронный процесс эволюции
- Отслеживание прогресса
- Поддержка контрольных точек
- Настраиваемые критерии остановки

### Функции приспособленности
Абстрактный базовый класс для определения функций приспособленности:
- Однокритериальная оптимизация
- Многокритериальная оптимизация
- Встроенные тестовые функции (Растригина и др.)

## Примеры

В директории `examples/` можно найти  пример использования:
- Оптимизация функций
- Многокритериальная оптимизация
- Реализация пользовательских хромосом
- Примеры визуализации
