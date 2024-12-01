import pickle
import json
import os
from typing import Dict, Any, Optional, Type
from datetime import datetime
import numpy as np
from ..core.chromosome import Chromosome
from ..core.population import Population

class EvolutionState:
    """Класс для хранения состояния эволюции"""

    def __init__(self):
        self.populations = []
        self.metadata = {}
        self.config = None
        self.generation = 0
        self.timestamp = datetime.now()
        self.additional_data = {}

    def add_population(self, population: Population) -> None:
        """Добавление популяции"""
        self.populations.append(population)

    def set_metadata(self, metadata: Dict[str, Any]) -> None:
        """Установка метаданных"""
        self.metadata = metadata

    def set_config(self, config: Any) -> None:
        """Установка конфигурации"""
        self.config = config

    def add_data(self, key: str, value: Any) -> None:
        """Добавление дополнительных данных"""
        self.additional_data[key] = value

class StatePersistence:
    """Класс для сохранения и загрузки состояния эволюции"""

    def __init__(self, base_dir: str = "evolution_states"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def save_state(self, state: EvolutionState,
                  filename: Optional[str] = None) -> str:
        """Сохранение состояния"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evolution_state_{timestamp}.pkl"

        filepath = os.path.join(self.base_dir, filename)

        # Сохраняем состояние
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

        # Сохраняем метаданные отдельно в JSON для удобства просмотра
        metadata_file = os.path.join(
            self.base_dir,
            f"{os.path.splitext(filename)[0]}_metadata.json"
        )

        with open(metadata_file, 'w') as f:
            # Преобразуем numpy массивы в списки для JSON
            metadata = self._prepare_for_json(state.metadata)
            json.dump(metadata, f, indent=4)

        return filepath

    def load_state(self, filepath: str,
                  chromosome_class: Type[Chromosome]) -> EvolutionState:
        """Загрузка состояния"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        # Восстанавливаем связи с классом хромосомы
        for population in state.populations:
            for individual in population.individuals:
                individual.__class__ = chromosome_class

        return state

    def _prepare_for_json(self, obj: Any) -> Any:
        """Подготовка объекта для сериализации в JSON"""
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

    def list_saved_states(self) -> Dict[str, Dict[str, Any]]:
        """Получение списка сохраненных состояний"""
        states = {}
        for filename in os.listdir(self.base_dir):
            if filename.endswith('.pkl'):
                metadata_file = os.path.join(
                    self.base_dir,
                    f"{os.path.splitext(filename)[0]}_metadata.json"
                )

                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                else:
                    metadata = {}

                states[filename] = {
                    'path': os.path.join(self.base_dir, filename),
                    'timestamp': os.path.getmtime(
                        os.path.join(self.base_dir, filename)
                    ),
                    'metadata': metadata
                }

        return states

    def create_checkpoint(self, evolution: 'Evolution') -> str:
        """Создание контрольной точки текущего состояния эволюции"""
        state = EvolutionState()

        # Сохраняем популяции
        for population in evolution.populations:
            state.add_population(population)

        # Сохраняем метаданные
        state.set_metadata(evolution.metadata.data)

        # Сохраняем конфигурацию
        state.set_config(evolution.config)

        # Сохраняем дополнительные данные
        state.add_data('adaptive_params', evolution.adaptive_params)
        state.generation = evolution.metadata.data['generation']

        # Создаем имя файла с номером поколения
        filename = f"checkpoint_gen_{state.generation}.pkl"

        return self.save_state(state, filename)

    def resume_from_checkpoint(self, filepath: str,
                             evolution: 'Evolution') -> None:
        """Восстановление эволюции из контрольной точки"""
        state = self.load_state(filepath, evolution.chromosome_class)

        # Восстанавливаем состояние
        evolution.populations = state.populations
        evolution.metadata = state.metadata
        evolution.generation = state.generation
        print(f"Evolution restored from generation {state.generation}")

class AutoCheckpoint:
    """Класс для автоматического создания контрольных точек"""

    def __init__(self,
                 persistence: StatePersistence,
                 checkpoint_frequency: int = 10,
                 max_checkpoints: int = 5):
        self.persistence = persistence
        self.frequency = checkpoint_frequency
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []

    def update(self, evolution: 'Evolution') -> Optional[str]:
        """Обновление контрольных точек"""
        generation = evolution.metadata.data['generation']

        if generation % self.frequency == 0:
            # Создаем новую контрольную точку
            checkpoint_path = self.persistence.create_checkpoint(evolution)
            self.checkpoints.append(checkpoint_path)

            # Удаляем старые контрольные точки, если их слишком много
            while len(self.checkpoints) > self.max_checkpoints:
                old_checkpoint = self.checkpoints.pop(0)
                if os.path.exists(old_checkpoint):
                    os.remove(old_checkpoint)

                # Удаляем также файл метаданных
                metadata_file = f"{os.path.splitext(old_checkpoint)[0]}_metadata.json"
                if os.path.exists(metadata_file):
                    os.remove(metadata_file)

            return checkpoint_path

        return None
