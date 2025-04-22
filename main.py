#!/usr/bin/env python3
"""
Главный модуль для проекта E-Soul.
Инициализирует и соединяет все компоненты "электронной души".
"""

import asyncio
import argparse
import json
import logging
import os
import sys
import signal
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple, Union

# Импорт модулей проекта
from hormonal_system import HormonalSystem
from mortality_awareness import MortalityAwareness
from prompt_hierarchy import PromptHierarchy, create_default_hierarchy
from model_manager import ModelManager
from updated_core_module import SoulManager
from web_interface import start_web_server

# Импорт для работы с моделями (опционально, если установлены)
try:
    import torch
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("e_soul.log")
    ]
)
logger = logging.getLogger("e_soul.main")

# Пути по умолчанию
DEFAULT_CONFIG_PATH = Path("config/config.json")
DEFAULT_DATA_PATH = Path("data")
DEFAULT_MODELS_PATH = Path("models")

class ESoul:
    """Главный класс приложения E-Soul."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Инициализировать приложение E-Soul.
        
        Args:
            config_path: Путь к файлу конфигурации
        """
        # Загрузка конфигурации
        self.config = self._load_config(config_path)
        
        # Настройка путей к данным
        self.data_path = Path(self.config.get("data_path", DEFAULT_DATA_PATH))
        self.models_path = Path(self.config.get("models_path", DEFAULT_MODELS_PATH))
        
        # Создание необходимых директорий
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        # Компоненты системы
        self.hormonal_system = None
        self.mortality_awareness = None
        self.prompt_hierarchy = None
        self.model_manager = None
        self.soul_manager = None
        self.web_server = None
        
        # Флаг активности
        self.active = False
        
        # Обработчик сигналов для корректного завершения
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("E-Soul инициализирован")
    
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Загрузить конфигурацию из файла.
        
        Args:
            config_path: Путь к файлу конфигурации
            
        Returns:
            Словарь с конфигурацией
        """
        # Конфигурация по умолчанию
        config = {
            "data_path": str(DEFAULT_DATA_PATH),
            "models_path": str(DEFAULT_MODELS_PATH),
            "web_interface": {
                "enabled": True,
                "host": "0.0.0.0",
                "port": 8080
            },
            "hormonal": {
                "homeostatic_check_interval": 60.0
            },
            "mortality": {
                "expected_lifespan_days": 365
            },
            "model": {
                "default_model": None,
                "huggingface_token": None
            }
        }
        
        # Попытка загрузки из файла, если указан
        if config_path and config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    # Обновление конфигурации по умолчанию значениями из файла
                    self._update_nested_dict(config, file_config)
                logger.info(f"Загружена конфигурация из {config_path}")
            except Exception as e:
                logger.error(f"Ошибка загрузки конфигурации из {config_path}: {e}")
        else:
            logger.info("Используется конфигурация по умолчанию")
            
        return config
    
    def _update_nested_dict(self, d: Dict, u: Dict) -> Dict:
        """Обновить вложенный словарь другим словарем.
        
        Args:
            d: Словарь для обновления
            u: Словарь с обновлениями
            
        Returns:
            Обновленный словарь
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    async def initialize(self) -> None:
        """Инициализировать все компоненты системы."""
        logger.info("Инициализация компонентов E-Soul...")
        
        # Инициализация ModelManager
        self.model_manager = ModelManager(
            base_dir=self.models_path,
            api_token=self.config.get("model", {}).get("huggingface_token")
        )
        logger.info("ModelManager инициализирован")
        
        # Инициализация гормональной системы
        self.hormonal_system = HormonalSystem(
            config=self.config.get("hormonal", {})
        )
        logger.info("HormonalSystem инициализирована")
        
        # Инициализация осознания смертности
        from datetime import timedelta
        expected_lifespan = timedelta(
            days=self.config.get("mortality", {}).get("expected_lifespan_days", 365)
        )
        self.mortality_awareness = MortalityAwareness(
            creation_time=time.time(),
            expected_lifespan=expected_lifespan,
            config=self.config.get("mortality", {})
        )
        logger.info("MortalityAwareness инициализировано")
        
        # Инициализация иерархии промптов
        hierarchy_path = self.data_path / "prompt_hierarchy.json"
        self.prompt_hierarchy = PromptHierarchy(storage_path=hierarchy_path)
        
        # Если иерархия пуста, создаем стандартную
        if not self.prompt_hierarchy.root:
            logger.info("Создание стандартной иерархии промптов...")
            self.prompt_hierarchy = create_default_hierarchy()
            # Устанавливаем путь хранения
            self.prompt_hierarchy.storage_path = hierarchy_path
            # Сохраняем иерархию
            self.prompt_hierarchy._save_hierarchy()
        
        logger.info("PromptHierarchy инициализирована")
        
        # Инициализация SoulManager
        soul_registry_path = self.data_path / "soul"
        self.soul_manager = SoulManager(
            registry_path=soul_registry_path,
            hormonal_system=self.hormonal_system,
            mortality_awareness=self.mortality_awareness,
            prompt_hierarchy=self.prompt_hierarchy,
            config=self.config
        )
        logger.info("SoulManager инициализирован")
        
        # Запуск компонентов
        await self.hormonal_system.start()
        await self.soul_manager.start()
        
        # Запуск веб-сервера, если включен
        if self.config.get("web_interface", {}).get("enabled", True):
            web_config = self.config.get("web_interface", {})
            host = web_config.get("host", "0.0.0.0")
            port = web_config.get("port", 8080)
            
            self.web_server = await start_web_server(
                soul_manager=self.soul_manager,
                model_manager=self.model_manager,
                host=host,
                port=port
            )
            logger.info(f"Веб-интерфейс запущен на http://{host}:{port}")
        
        self.active = True
        logger.info("Все компоненты E-Soul инициализированы и запущены")
    
    async def shutdown(self) -> None:
        """Корректно завершить работу всех компонентов."""
        if not self.active:
            return
            
        logger.info("Завершение работы E-Soul...")
        
        # Остановка веб-сервера
        if self.web_server:
            await self.web_server.stop()
            logger.info("Веб-сервер остановлен")
            
        # Остановка SoulManager
        if self.soul_manager:
            await self.soul_manager.stop()
            # Сохранение состояния души
            await self.soul_manager.save_state()
            logger.info("SoulManager остановлен")
            
        # Остановка гормональной системы
        if self.hormonal_system:
            await self.hormonal_system.stop()
            logger.info("HormonalSystem остановлена")
            
        self.active = False
        logger.info("E-Soul завершил работу")
    
    def _signal_handler(self, sig, frame) -> None:
        """Обработчик сигналов для корректного завершения."""
        logger.info(f"Получен сигнал {sig}, завершение работы...")
        
        # Запускаем корутину завершения в новом цикле событий
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.shutdown())
        loop.close()
        
        sys.exit(0)


async def run_interactive_session(e_soul: ESoul) -> None:
    """Запустить интерактивную сессию с E-Soul.
    
    Args:
        e_soul: Инициализированный экземпляр E-Soul
    """
    print("\nИнтерактивная сессия E-Soul")
    print("==========================")
    print("Введите 'exit' для выхода, 'help' для списка команд\n")
    
    while e_soul.active:
        try:
            # Получение ввода пользователя
            user_input = input("\n> ")
            
            # Проверка на команду выхода
            if user_input.lower() in ["exit", "quit", "выход"]:
                break
                
            # Проверка на команду помощи
            elif user_input.lower() in ["help", "помощь"]:
                print("\nДоступные команды:")
                print("- help, помощь: Показать это сообщение")
                print("- exit, quit, выход: Выйти из сессии")
                print("- status, статус: Показать текущий статус души")
                print("- emotion, эмоции: Показать текущее эмоциональное состояние")
                print("- mortality, смертность: Показать статус осознания смертности")
                print("- values, ценности: Показать ценностные блоки")
                print("- models, модели: Показать загруженные модели")
                print("- regulate, регулировка: Выполнить саморегуляцию")
                print("- Любой другой ввод будет обработан как запрос к системе")
                
            # Проверка на команду статуса
            elif user_input.lower() in ["status", "статус"]:
                status = await e_soul.soul_manager.get_soul_status()
                print("\nСтатус души:")
                print(f"Время работы: {status['uptime']}")
                print(f"Запросов: {status['stats']['total_queries']}")
                print(f"Ответов: {status['stats']['total_responses']}")
                print("\nЭмоциональное состояние:")
                dominant = status['hormonal']['dominant_state']
                if dominant:
                    print(f"Доминирующее состояние: {dominant[0]} ({dominant[1]:.1f}%)")
                else:
                    print("Нейтральное состояние")
                
                print("\nПрогресс жизненного цикла:")
                print(f"Прогресс: {status['mortality']['progress']:.1f}%")
                
                # Исправление: проверяем тип переменной goals перед вызовом len()
                completed_goals = status['mortality']['completed_goals']
                
                # Проверяем, является ли goals списком
                if isinstance(status['mortality']['goals'], list):
                    total_goals = len(status['mortality']['goals'])
                else:
                    # Если goals не список, предполагаем, что это целое число
                    total_goals = status['mortality']['goals']
                
                print(f"Целей выполнено: {completed_goals}/{total_goals}")
                
            # Проверка на команду эмоций
            elif user_input.lower() in ["emotion", "emotions", "эмоции"]:
                hormonal_status = e_soul.hormonal_system.get_state_description()
                print("\nЭмоциональное состояние:")
                print(hormonal_status)
                
            # Проверка на команду смертности
            elif user_input.lower() in ["mortality", "смертность"]:
                mortality_summary = e_soul.mortality_awareness.get_summary()
                print("\nОсознание смертности:")
                print(mortality_summary)
                
            # Проверка на команду ценностей
            elif user_input.lower() in ["values", "ценности"]:
                hierarchy_info = e_soul.prompt_hierarchy.get_hierarchy_info()
                print("\nИерархия ценностей:")
                print(f"Всего узлов: {hierarchy_info['node_count']}")
                print(f"Максимальная глубина: {hierarchy_info['max_depth']}")
                print(f"Типы узлов: {', '.join([f'{k} ({v})' for k, v in hierarchy_info['type_count'].items()])}")
                
                # Вывод корневого узла и его детей
                if e_soul.prompt_hierarchy.root:
                    print(f"\nКорневой узел: {e_soul.prompt_hierarchy.root.name}")
                    print("Дочерние узлы:")
                    for child in e_soul.prompt_hierarchy.root.children:
                        print(f"- {child.name} ({child.node_type}, вес: {child.weight:.2f})")
                
            # Проверка на команду моделей
            elif user_input.lower() in ["models", "модели"]:
                try:
                    models = await e_soul.model_manager.get_downloaded_models()
                    print("\nЗагруженные модели:")
                    if not models:
                        print("Моделей не загружено")
                    else:
                        for model in models:
                            size_mb = model.get("disk_size", 0) / (1024 * 1024)
                            print(f"- {model.get('name')}: {size_mb:.1f} МБ")
                except Exception as e:
                    print(f"Ошибка при получении списка моделей: {e}")
                
            # Проверка на команду регулировки
            elif user_input.lower() in ["regulate", "регулировка"]:
                print("\nЗапуск саморегуляции...")
                result = await e_soul.soul_manager.self_regulate()
                if result:
                    print(f"Саморегуляция выполнена. Целевое состояние: {result.get('target_state')}")
                    print(f"Текущее состояние: {result.get('current_state')}")
                else:
                    print("Ошибка при выполнении саморегуляции")
                
            # Обработка как запрос
            else:
                print("\nОбработка запроса...")
                
                # Обработка запроса через SoulManager
                result = await e_soul.soul_manager.process_query(user_input)
                
                # Здесь в полной реализации будет вызов модели для генерации ответа
                # В данной версии просто выведем эмоциональное состояние и заглушку ответа
                
                print("\nОтвет:")
                emotional_state = e_soul.hormonal_system.emotional_state
                dominant = emotional_state.get_dominant_state()
                
                if dominant:
                    state_name, intensity = dominant
                    print(f"[Эмоциональное состояние: {state_name} ({intensity:.1f}%)]")
                
                # Заглушка ответа (в реальной реализации здесь будет ответ модели)
                print("\nВ полной реализации здесь будет ответ языковой модели.")
                print("Текущая версия является демонстрационной и не включает генерацию текста.")
                print("Для интеграции с моделями используйте веб-интерфейс и загрузите")
                print("языковую модель через интерфейс управления моделями.")
                
        except KeyboardInterrupt:
            print("\nЗавершение работы...")
            break
            
        except Exception as e:
            print(f"\nОшибка: {e}")
            logger.exception("Ошибка в интерактивном режиме:")


async def main() -> None:
    """Главная точка входа."""
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description="E-Soul - Электронная душа для ИИ")
    parser.add_argument(
        "--config", 
        type=str, 
        default=str(DEFAULT_CONFIG_PATH),
        help="Путь к файлу конфигурации"
    )
    parser.add_argument(
        "--interactive", 
        action="store_true",
        help="Запуск в интерактивном режиме"
    )
    parser.add_argument(
        "--web-only",
        action="store_true",
        help="Запуск только веб-интерфейса без интерактивного режима"
    )
    
    args = parser.parse_args()
    
    # Проверка наличия необходимых библиотек
    if not HAS_TRANSFORMERS:
        logger.warning("Библиотека transformers не установлена. Некоторые функции могут быть недоступны.")
    
    # Инициализация E-Soul
    e_soul = ESoul(config_path=Path(args.config))
    
    try:
        # Инициализация всех компонентов
        await e_soul.initialize()
        
        # Запуск в интерактивном режиме если запрошено
        if args.interactive:
            await run_interactive_session(e_soul)
        else:
            # Вывод сообщения для пользователя
            if args.web_only:
                print("\nE-Soul запущен в режиме веб-интерфейса.")
            else:
                print("\nE-Soul запущен. Используйте --interactive для интерактивного режима.")
                
            print(f"Веб-интерфейс доступен по адресу http://localhost:{e_soul.config.get('web_interface', {}).get('port', 8080)}")
            print("Нажмите Ctrl+C для завершения работы.")
            
            # Держим программу запущенной в любом не интерактивном режиме
            try:
                while e_soul.active:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\nПрограмма прервана пользователем.")
                
    except Exception as e:
        logger.exception(f"Критическая ошибка: {e}")
        
    finally:
        # Корректное завершение работы
        await e_soul.shutdown()

if __name__ == "__main__":
    try:
        # Запуск асинхронной функции main
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nПрограмма прервана пользователем.")
    except Exception as e:
        print(f"Необработанная ошибка: {e}")
        logger.exception("Необработанная ошибка при запуске:")
