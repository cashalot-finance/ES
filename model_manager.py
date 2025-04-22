"""
Model manager module for the E-Soul project.
Provides functionality for discovering, downloading, and managing models from Hugging Face.
"""

import asyncio
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple, Union, Callable

import aiohttp
import torch
import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("e_soul.model_manager")

class HuggingFaceIntegration:
    """Provides integration with Hugging Face Hub API."""
    
    # --- OFFLINE MODELS CONSTANT ---
    OFFLINE_MODELS = [
        {
            "id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "name": "TinyLlama-1.1B-Chat-v1.0",
            "author": "TinyLlama",
            "downloads": 245895,
            "likes": 732,
            "tags": ["text-generation", "pytorch", "tiny", "chat"],
            "pipeline_tag": "text-generation",
            "last_modified": "2023-12-22T19:38:45.000Z",
            "description": "Малая чат-модель мощностью 1.1B параметров. Идеальна для устройств с ограниченными ресурсами.",
            "model_type": "text-generation",
            "model_size": 4400000000,
            "library": "transformers"
        },
        {
            "id": "EleutherAI/pythia-1b",
            "name": "pythia-1b",
            "author": "EleutherAI",
            "downloads": 189371,
            "likes": 421,
            "tags": ["text-generation", "pytorch", "pythia", "small"],
            "pipeline_tag": "text-generation",
            "last_modified": "2023-06-13T10:16:32.000Z",
            "description": "Малая модель в семействе Pythia с 1B параметров. Подходит для исследования и экспериментов.",
            "model_type": "text-generation",
            "model_size": 4000000000,
            "library": "transformers"
        },
        {
            "id": "facebook/opt-1.3b",
            "name": "opt-1.3b",
            "author": "facebook",
            "downloads": 526248,
            "likes": 872,
            "tags": ["text-generation", "pytorch", "opt"],
            "pipeline_tag": "text-generation",
            "last_modified": "2023-01-17T20:12:54.000Z",
            "description": "OPT-1.3B - малая модель из серии OPT от Meta AI. Хорошо сбалансированная по размеру и производительности.",
            "model_type": "text-generation",
            "model_size": 5200000000,
            "library": "transformers"
        },
        {
            "id": "bigscience/bloom-1b7",
            "name": "bloom-1b7",
            "author": "bigscience",
            "downloads": 384619,
            "likes": 653,
            "tags": ["text-generation", "pytorch", "bloom", "multilingual"],
            "pipeline_tag": "text-generation",
            "last_modified": "2023-04-28T14:53:27.000Z",
            "description": "BLOOM 1.7B - многоязычная языковая модель с поддержкой 46+ языков, включая русский.",
            "model_type": "text-generation",
            "model_size": 6800000000,
            "library": "transformers"
        },
        {
            "id": "mistralai/Mistral-7B-v0.1",
            "name": "Mistral-7B-v0.1",
            "author": "mistralai",
            "downloads": 1289764,
            "likes": 1876,
            "tags": ["text-generation", "pytorch", "mistral"],
            "pipeline_tag": "text-generation",
            "last_modified": "2023-09-29T10:25:36.000Z",
            "description": "Mistral 7B - эффективная модель среднего размера с отличным балансом производительности и ресурсопотребления.",
            "model_type": "text-generation",
            "model_size": 28000000000,
            "library": "transformers"
        },
        {
            "id": "meta-llama/Llama-2-7b-chat-hf",
            "name": "Llama-2-7b-chat-hf",
            "author": "meta-llama",
            "downloads": 2456783,
            "likes": 3287,
            "tags": ["text-generation", "pytorch", "llama", "chat"],
            "pipeline_tag": "text-generation",
            "last_modified": "2023-07-18T03:36:15.000Z",
            "description": "Llama 2 7B Chat - оптимизированная для диалогов версия модели Llama 2 от Meta.",
            "model_type": "text-generation",
            "model_size": 28000000000,
            "library": "transformers"
        },
        {
            "id": "IlyaGusev/saiga_mistral_7b",
            "name": "saiga_mistral_7b",
            "author": "IlyaGusev",
            "downloads": 183267,
            "likes": 412,
            "tags": ["text-generation", "pytorch", "russian", "mistral"],
            "pipeline_tag": "text-generation",
            "last_modified": "2023-11-15T09:52:31.000Z",
            "description": "Русскоязычная Mistral 7B модель с инструкциями, обучена на русскоязычных данных.",
            "model_type": "text-generation",
            "model_size": 28000000000,
            "library": "transformers"
        },
        {
            "id": "TheBloke/Llama-2-7B-GGUF",
            "name": "Llama-2-7B-GGUF",
            "author": "TheBloke",
            "downloads": 567898,
            "likes": 921,
            "tags": ["text-generation", "llama", "gguf", "quantized"],
            "pipeline_tag": "text-generation",
            "last_modified": "2023-09-05T10:17:29.000Z",
            "description": "Llama 2 7B в квантизированном GGUF формате для запуска на CPU или с ограниченными ресурсами GPU.",
            "model_type": "text-generation",
            "model_size": 11000000000,
            "library": "llama.cpp"
        }
    ]

    def __init__(self, 
                 api_token: Optional[str] = None,
                 cache_dir: Optional[Path] = None,
                 cache_expiry: int = 3600):
        """Initialize the Hugging Face integration.
        
        Args:
            api_token: Optional Hugging Face API token
            cache_dir: Directory for caching API responses
            cache_expiry: Cache expiry time in seconds
        """
        self.api_token = api_token
        self.api_base_url = "https://huggingface.co/api"
        self.cache_dir = cache_dir
        self.cache_expiry = cache_expiry
        
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            
        logger.info("Hugging Face integration initialized")
    
    async def search_models(self, 
                          query: str = "",
                          filter_tags: Optional[List[str]] = None,
                          model_type: Optional[str] = None,
                          sort: str = "downloads",
                          direction: str = "desc",
                          limit: int = 20,
                          use_cache: bool = True) -> List[Dict[str, Any]]:
        """Search for models on Hugging Face Hub.
        
        Args:
            query: Search query
            filter_tags: List of tags to filter by
            model_type: Type of model to filter by
            sort: Field to sort by
            direction: Sort direction (asc or desc)
            limit: Maximum number of results
            use_cache: Whether to use cached results
            
        Returns:
            List of model information dictionaries
        """
        # Попробуем сначала выполнить поиск обычным способом через API
        try:
            results = await self._search_models_api(
                query, filter_tags, model_type, sort, direction, limit, use_cache
            )
            
            # Если API запрос успешен и вернул результаты, используем их
            if results:
                return results
        except Exception as e:
            logger.error(f"Error while searching models via API: {e}")
        
        # Если API запрос не удался или вернул пустой список, используем офлайн-данные
        logger.info("Using offline model data as fallback")
        
        filtered_models = self.OFFLINE_MODELS.copy()
        
        # Применяем фильтры если они заданы
        if query:
            query_lower = query.lower()
            filtered_models = [
                model for model in filtered_models 
                if query_lower in model["name"].lower() or 
                   query_lower in model["description"].lower() or
                   query_lower in model["author"].lower()
            ]
        
        if filter_tags:
            filtered_models = [
                model for model in filtered_models
                if all(tag in model["tags"] for tag in filter_tags)
            ]
        
        if model_type:
            filtered_models = [
                model for model in filtered_models
                if model["model_type"] == model_type
            ]
        
        # Сортировка
        reverse = direction.lower() == "desc"
        if sort in ["downloads", "likes"]:
            filtered_models.sort(key=lambda x: x[sort], reverse=reverse)
        elif sort == "lastModified":
            filtered_models.sort(key=lambda x: x["last_modified"], reverse=reverse)
        
        # Применяем лимит
        limited_models = filtered_models[:limit]
        
        return limited_models

    async def _search_models_api(self, 
                             query: str = "",
                             filter_tags: Optional[List[str]] = None,
                             model_type: Optional[str] = None,
                             sort: str = "downloads",
                             direction: str = "desc",
                             limit: int = 20,
                             use_cache: bool = True) -> List[Dict[str, Any]]:
        """Внутренний метод для поиска моделей через API."""
        # Оригинальный код метода search_models сюда
        # (тот, который был исправлен в первом артефакте)
        
        # Build cache key if using cache
        cache_key = None
        if use_cache and self.cache_dir:
            cache_parts = [
                "search",
                query.replace(" ", "_") if query else "empty",
                "-".join(filter_tags or []),
                model_type or "any",
                f"{sort}-{direction}",
                str(limit)
            ]
            cache_key = "_".join(cache_parts) + ".json"
            cache_path = self.cache_dir / cache_key
            
            # Check if cache exists and is valid
            if cache_path.exists():
                cache_age = time.time() - cache_path.stat().st_mtime
                if cache_age < self.cache_expiry:
                    try:
                        with open(cache_path, 'r') as f:
                            return json.load(f)
                    except Exception as e:
                        logger.warning(f"Error loading cache file {cache_path}: {e}")
        
        # Build API request
        params = {
            "sort": sort,
            "direction": direction,
            "limit": str(limit)  # Преобразуем в строку
        }
        
        # Добавляем search только если он не пустой
        if query and query.strip():
            params["search"] = query
            
        if filter_tags:
            filter_str = ",".join(filter_tags)
            if filter_str:
                params["filter"] = filter_str
            
        if model_type and model_type.strip():
            params["type"] = model_type
        
        # Make API request
        headers = {}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
        
        try:
            logger.debug(f"Sending request to HF API: GET {self.api_base_url}/models with params: {params}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.api_base_url}/models",
                    params=params,
                    headers=headers
                ) as response:
                    if response.status != 200:
                        response_text = await response.text()
                        logger.error(f"Error searching models: {response.status}, Response: {response_text}")
                        # Возвращаем пустой список вместо ошибки
                        return []
                        
                    results = await response.json()
        except Exception as e:
            logger.error(f"Exception during HF API request: {e}")
            return []
        
        # Process results
        processed_results = []
        
        for model in results:
            try:
                # Extract relevant information
                processed_model = {
                    "id": model.get("id", ""),
                    "name": model.get("modelId", model.get("id", "")).split("/")[-1],
                    "author": model.get("modelId", "/").split("/")[0] if "/" in model.get("modelId", "/") else "",
                    "downloads": model.get("downloads", 0),
                    "likes": model.get("likes", 0),
                    "tags": model.get("tags", []),
                    "pipeline_tag": model.get("pipeline_tag", ""),
                    "last_modified": model.get("lastModified", ""),
                    "description": model.get("description", ""),
                    "model_type": model.get("type", "model"),
                    "model_size": self._extract_model_size(model),
                    "library": model.get("library", {}).get("id") if model.get("library") else None
                }
                
                processed_results.append(processed_model)
            except Exception as e:
                logger.warning(f"Error processing model data: {e}")
        
        # Cache results if using cache
        if use_cache and self.cache_dir and cache_key:
            try:
                with open(self.cache_dir / cache_key, 'w') as f:
                    json.dump(processed_results, f, indent=2)
            except Exception as e:
                logger.warning(f"Error saving results to cache: {e}")
        
        return processed_results
    
    async def get_model_info(self, model_id: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific model.
        
        Args:
            model_id: Model ID (username/modelname)
            use_cache: Whether to use cached results
            
        Returns:
            Model information dictionary or None if not found
        """
        # Check cache first
        cache_key = None
        if use_cache and self.cache_dir:
            cache_key = f"model_info_{model_id.replace('/', '_')}.json"
            cache_path = self.cache_dir / cache_key
            
            # Check if cache exists and is valid
            if cache_path.exists():
                cache_age = time.time() - cache_path.stat().st_mtime
                if cache_age < self.cache_expiry:
                    try:
                        with open(cache_path, 'r') as f:
                            return json.load(f)
                    except Exception as e:
                        logger.warning(f"Error loading cache file {cache_path}: {e}")
        
        # Make API request
        headers = {}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
            
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.api_base_url}/models/{model_id}",
                headers=headers
            ) as response:
                if response.status != 200:
                    logger.error(f"Error getting model info: {response.status}")
                    return None
                    
                model_info = await response.json()
        
        # Process model info
        processed_info = {
            "id": model_info.get("id", ""),
            "name": model_info.get("modelId", model_info.get("id", "")).split("/")[-1],
            "author": model_info.get("modelId", "/").split("/")[0] if "/" in model_info.get("modelId", "/") else "",
            "downloads": model_info.get("downloads", 0),
            "likes": model_info.get("likes", 0),
            "tags": model_info.get("tags", []),
            "pipeline_tag": model_info.get("pipeline_tag", ""),
            "last_modified": model_info.get("lastModified", ""),
            "description": model_info.get("description", ""),
            "model_type": model_info.get("type", "model"),
            "library": model_info.get("library", {}).get("id") if model_info.get("library") else None,
            "config": model_info.get("config", {}),
            "siblings": [
                {
                    "name": sibling.get("rfilename", ""),
                    "size": sibling.get("size", 0),
                    "type": sibling.get("type", ""),
                    "url": sibling.get("url", "")
                }
                for sibling in model_info.get("siblings", [])
            ],
            "model_size": self._extract_model_size(model_info),
            "cardData": model_info.get("cardData", {})
        }
        
        # Cache results
        if use_cache and self.cache_dir and cache_key:
            try:
                with open(self.cache_dir / cache_key, 'w') as f:
                    json.dump(processed_info, f, indent=2)
            except Exception as e:
                logger.warning(f"Error saving model info to cache: {e}")
        
        return processed_info
    
    async def get_model_tags(self, use_cache: bool = True) -> List[str]:
        """Get list of all available model tags.
        
        Args:
            use_cache: Whether to use cached results
            
        Returns:
            List of tag strings
        """
        # Check cache first
        cache_key = None
        if use_cache and self.cache_dir:
            cache_key = "model_tags.json"
            cache_path = self.cache_dir / cache_key
            
            # Check if cache exists and is valid
            if cache_path.exists():
                cache_age = time.time() - cache_path.stat().st_mtime
                if cache_age < self.cache_expiry:
                    try:
                        with open(cache_path, 'r') as f:
                            return json.load(f)
                    except Exception as e:
                        logger.warning(f"Error loading cache file {cache_path}: {e}")
        
        # Make API request
        headers = {}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
            
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.api_base_url}/models-tags-by-type",
                headers=headers
            ) as response:
                if response.status != 200:
                    logger.error(f"Error getting model tags: {response.status}")
                    return []
                    
                tag_data = await response.json()
        
        # Extract tags
        all_tags = []
        for category, tags in tag_data.items():
            all_tags.extend(tags)
            
        # Remove duplicates
        unique_tags = list(set(all_tags))
        
        # Cache results
        if use_cache and self.cache_dir and cache_key:
            try:
                with open(self.cache_dir / cache_key, 'w') as f:
                    json.dump(unique_tags, f, indent=2)
            except Exception as e:
                logger.warning(f"Error saving tags to cache: {e}")
        
        return unique_tags
    
    async def download_file(self, 
                          url: str, 
                          destination: Path,
                          progress_callback: Optional[Callable[[int, int], None]] = None) -> bool:
        """Download a file from a URL.
        
        Args:
            url: URL to download from
            destination: Destination path
            progress_callback: Optional callback for progress updates
            
        Returns:
            Success status
        """
        # Ensure parent directory exists
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        headers = {}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
            
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        logger.error(f"Error downloading file: {response.status}")
                        return False
                        
                    # Get file size
                    total_size = int(response.headers.get("content-length", 0))
                    
                    # Download with progress updates
                    with open(destination, 'wb') as f:
                        downloaded = 0
                        
                        async for chunk in response.content.iter_chunked(8192):
                            if not chunk:
                                break
                                
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            if progress_callback:
                                progress_callback(downloaded, total_size)
                                
            logger.info(f"Downloaded file to {destination}")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading file: {e}")
            # Clean up partial download
            if destination.exists():
                destination.unlink()
                
            return False
    
    def _extract_model_size(self, model_info: Dict[str, Any]) -> Optional[int]:
        """Extract model size information from model info.
        
        Args:
            model_info: Model information dictionary
            
        Returns:
            Model size in bytes or None if unknown
        """
        # Try to get from transformers info
        if "transformersInfo" in model_info:
            try:
                if "parameterSize" in model_info["transformersInfo"]:
                    param_str = model_info["transformersInfo"]["parameterSize"]
                    
                    if param_str.endswith("B"):
                        # Convert from billions of parameters (approximately)
                        try:
                            params_billions = float(param_str[:-1])
                            # Rough estimate: 4 bytes per parameter
                            return int(params_billions * 1e9 * 4)
                        except ValueError:
                            pass
                    elif param_str.endswith("M"):
                        # Convert from millions of parameters
                        try:
                            params_millions = float(param_str[:-1])
                            # Rough estimate: 4 bytes per parameter
                            return int(params_millions * 1e6 * 4)
                        except ValueError:
                            pass
            except Exception:
                pass
        
        # Try to get from siblings (model files)
        if "siblings" in model_info:
            try:
                # Calculate total size of model files
                total_size = 0
                for sibling in model_info["siblings"]:
                    # Only count model files
                    if sibling.get("type") in ["model", "weights"]:
                        total_size += sibling.get("size", 0)
                        
                if total_size > 0:
                    return total_size
            except Exception:
                pass
        
        # Try to guess from tags
        if "tags" in model_info:
            for tag in model_info.get("tags", []):
                # Look for tags like "1b", "7b", etc.
                if tag.endswith("b") and tag[:-1].isdigit():
                    try:
                        params_billions = int(tag[:-1])
                        # Rough estimate: 4 bytes per parameter
                        return int(params_billions * 1e9 * 4)
                    except ValueError:
                        pass
        
        return None


class ModelLoader:
    """Manages the loading and downloading of models."""
    
    def __init__(self, 
                 models_dir: Path,
                 cache_dir: Optional[Path] = None,
                 huggingface_integration: Optional[HuggingFaceIntegration] = None):
        """Initialize the model loader.
        
        Args:
            models_dir: Directory to store models
            cache_dir: Directory for caching
            huggingface_integration: HuggingFaceIntegration instance
        """
        self.models_dir = models_dir
        self.cache_dir = cache_dir
        self.hf_integration = huggingface_integration or HuggingFaceIntegration(cache_dir=cache_dir)
        
        # Ensure directories exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            
        logger.info("Model loader initialized")
    
    async def download_model(self, 
                           model_id: str, 
                           progress_callback: Optional[Callable[[float], None]] = None) -> bool:
        """Download a model from Hugging Face.
        
        Args:
            model_id: Model ID (username/modelname)
            progress_callback: Optional callback for progress updates (0-1)
            
        Returns:
            Success status
        """
        logger.info(f"Downloading model {model_id}")
        
        # Get model info
        model_info = await self.hf_integration.get_model_info(model_id)
        
        if not model_info:
            logger.error(f"Could not get info for model {model_id}")
            return False
            
        # Create model directory
        model_dir = self.models_dir / model_id.replace("/", "_")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model info
        with open(model_dir / "model_info.json", 'w') as f:
            json.dump(model_info, f, indent=2)
            
        # Identify files to download
        download_files = []
        
        for sibling in model_info.get("siblings", []):
            # Skip readme and license files
            if sibling["name"].lower() in ["readme.md", "license", "license.md", ".gitattributes"]:
                continue
                
            download_files.append(sibling)
            
        if not download_files:
            logger.error(f"No files to download for model {model_id}")
            return False
            
        # Calculate total download size
        total_size = sum(file.get("size", 0) for file in download_files)
        
        # Download each file
        downloaded_size = 0
        success = True
        
        for file in download_files:
            file_url = file.get("url", "")
            file_name = file.get("name", "")
            file_size = file.get("size", 0)
            
            if not file_url or not file_name:
                logger.warning(f"Invalid file information: {file}")
                continue
                
            file_path = model_dir / file_name
            
            # Skip if file already exists with correct size
            if file_path.exists() and file_path.stat().st_size == file_size:
                logger.info(f"File {file_name} already exists with correct size, skipping")
                downloaded_size += file_size
                
                # Update progress
                if progress_callback and total_size > 0:
                    progress_callback(downloaded_size / total_size)
                    
                continue
                
            # Define progress callback for this file
            def file_progress_callback(downloaded: int, file_total: int) -> None:
                if progress_callback and total_size > 0:
                    current_progress = (downloaded_size + downloaded) / total_size
                    progress_callback(current_progress)
            
            # Download file
            file_success = await self.hf_integration.download_file(
                file_url,
                file_path,
                file_progress_callback
            )
            
            if not file_success:
                logger.error(f"Failed to download file {file_name}")
                success = False
                continue
                
            downloaded_size += file_size
            
            # Update progress
            if progress_callback and total_size > 0:
                progress_callback(downloaded_size / total_size)
                
        if success:
            logger.info(f"Successfully downloaded model {model_id}")
        else:
            logger.warning(f"Downloaded model {model_id} with some errors")
            
        return success
    
    def is_model_downloaded(self, model_id: str) -> bool:
        """Check if a model is already downloaded.
        
        Args:
            model_id: Model ID
            
        Returns:
            Whether model is downloaded
        """
        model_dir = self.models_dir / model_id.replace("/", "_")
        
        # Check if directory exists
        if not model_dir.exists() or not model_dir.is_dir():
            return False
            
        # Check if model info exists
        if not (model_dir / "model_info.json").exists():
            return False
            
        # Load model info
        try:
            with open(model_dir / "model_info.json", 'r') as f:
                model_info = json.load(f)
                
            # Check if all required files are present
            for sibling in model_info.get("siblings", []):
                # Skip readme and license files
                if sibling["name"].lower() in ["readme.md", "license", "license.md", ".gitattributes"]:
                    continue
                    
                file_path = model_dir / sibling["name"]
                file_size = sibling.get("size", 0)
                
                if not file_path.exists() or file_path.stat().st_size != file_size:
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error checking model {model_id}: {e}")
            return False
    
    async def get_downloaded_models(self) -> List[Dict[str, Any]]:
        """Get information about all downloaded models.
        
        Returns:
            List of model information dictionaries
        """
        models = []
        
        # Check each subdirectory in models_dir
        for model_dir in self.models_dir.iterdir():
            if not model_dir.is_dir():
                continue
                
            # Check if model info exists
            info_path = model_dir / "model_info.json"
            if not info_path.exists():
                continue
                
            # Load model info
            try:
                with open(info_path, 'r') as f:
                    model_info = json.load(f)
                    
                # Calculate model size on disk
                disk_size = 0
                for file_path in model_dir.iterdir():
                    if file_path.is_file():
                        disk_size += file_path.stat().st_size
                        
                # Add download info
                model_info["disk_size"] = disk_size
                model_info["download_path"] = str(model_dir)
                
                # Check if model is complete
                model_info["is_complete"] = True
                for sibling in model_info.get("siblings", []):
                    # Skip readme and license files
                    if sibling["name"].lower() in ["readme.md", "license", "license.md", ".gitattributes"]:
                        continue
                        
                    file_path = model_dir / sibling["name"]
                    file_size = sibling.get("size", 0)
                    
                    if not file_path.exists() or file_path.stat().st_size != file_size:
                        model_info["is_complete"] = False
                        break
                        
                models.append(model_info)
                
            except Exception as e:
                logger.error(f"Error loading model info from {info_path}: {e}")
        
        return models
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a downloaded model.
        
        Args:
            model_id: Model ID
            
        Returns:
            Success status
        """
        model_dir = self.models_dir / model_id.replace("/", "_")
        
        if not model_dir.exists() or not model_dir.is_dir():
            logger.warning(f"Model directory {model_dir} does not exist")
            return False
            
        try:
            # Remove model directory and all contents
            shutil.rmtree(model_dir)
            logger.info(f"Deleted model {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting model {model_id}: {e}")
            return False


class ModelRegistry:
    """Registry for managing information about models and their compatibility."""
    
    def __init__(self, 
                 registry_path: Optional[Path] = None,
                 models_dir: Optional[Path] = None):
        """Initialize the model registry.
        
        Args:
            registry_path: Path to registry file
            models_dir: Directory containing models
        """
        self.registry_path = registry_path
        self.models_dir = models_dir
        self.registry: Dict[str, Dict[str, Any]] = {}
        
        # Compatibility requirements
        self.compatibility_requirements = {
            "tiny": {
                "max_size_bytes": 2 * 1024 * 1024 * 1024,  # 2 GB
                "min_vram": 2,  # GB
                "preferred_libraries": ["transformers", "pytorch"]
            },
            "small": {
                "max_size_bytes": 10 * 1024 * 1024 * 1024,  # 10 GB
                "min_vram": 8,  # GB
                "preferred_libraries": ["transformers", "pytorch"]
            },
            "medium": {
                "max_size_bytes": 30 * 1024 * 1024 * 1024,  # 30 GB
                "min_vram": 16,  # GB
                "preferred_libraries": ["transformers", "pytorch"]
            },
            "large": {
                "max_size_bytes": None,  # No limit
                "min_vram": 24,  # GB
                "preferred_libraries": ["transformers", "pytorch"]
            }
        }
        
        # Load registry if it exists
        if registry_path and registry_path.exists():
            self._load_registry()
            
        logger.info("Model registry initialized")
    
    def _load_registry(self) -> None:
        """Load the registry from file."""
        try:
            with open(self.registry_path, 'r') as f:
                self.registry = json.load(f)
                
            logger.info(f"Loaded model registry from {self.registry_path}")
            
        except Exception as e:
            logger.error(f"Error loading model registry: {e}")
            self.registry = {}
    
    def _save_registry(self) -> None:
        """Save the registry to file."""
        if not self.registry_path:
            return
            
        try:
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.registry_path, 'w') as f:
                json.dump(self.registry, f, indent=2)
                
            logger.info(f"Saved model registry to {self.registry_path}")
            
        except Exception as e:
            logger.error(f"Error saving model registry: {e}")
    
    async def register_model(self, 
                          model_id: str, 
                          model_info: Dict[str, Any]) -> bool:
        """Register a model in the registry.
        
        Args:
            model_id: Model ID
            model_info: Model information
            
        Returns:
            Success status
        """
        # Add to registry
        self.registry[model_id] = {
            "model_id": model_id,
            "name": model_info.get("name", ""),
            "author": model_info.get("author", ""),
            "description": model_info.get("description", ""),
            "tags": model_info.get("tags", []),
            "model_type": model_info.get("model_type", "model"),
            "size": model_info.get("model_size", 0),
            "disk_size": model_info.get("disk_size", 0),
            "library": model_info.get("library", None),
            "download_path": model_info.get("download_path", ""),
            "is_complete": model_info.get("is_complete", False),
            "compatibility": await self.check_model_compatibility(model_info),
            "registration_time": time.time()
        }
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Registered model {model_id}")
        return True
    
    def unregister_model(self, model_id: str) -> bool:
        """Unregister a model from the registry.
        
        Args:
            model_id: Model ID
            
        Returns:
            Success status
        """
        if model_id not in self.registry:
            logger.warning(f"Model {model_id} not found in registry")
            return False
            
        # Remove from registry
        del self.registry[model_id]
        
        # Save registry
        self._save_registry()
        
        logger.info(f"Unregistered model {model_id}")
        return True
    
    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a registered model.
        
        Args:
            model_id: Model ID
            
        Returns:
            Model information or None if not found
        """
        return self.registry.get(model_id)
    
    def get_all_models(self) -> List[Dict[str, Any]]:
        """Get information about all registered models.
        
        Returns:
            List of model information dictionaries
        """
        return list(self.registry.values())
    
    def find_models(self, 
                   query: str = "",
                   tags: Optional[List[str]] = None,
                   library: Optional[str] = None,
                   size_range: Optional[Tuple[int, int]] = None,
                   compatibility_tier: Optional[str] = None) -> List[Dict[str, Any]]:
        """Find models matching criteria.
        
        Args:
            query: Search query
            tags: List of tags to filter by
            library: Library to filter by
            size_range: Tuple of (min_size, max_size) in bytes
            compatibility_tier: Compatibility tier to filter by
            
        Returns:
            List of matching model information dictionaries
        """
        results = []
        
        for model_id, model_info in self.registry.items():
            # Check query
            if query:
                query_lower = query.lower()
                name_match = query_lower in model_info.get("name", "").lower()
                author_match = query_lower in model_info.get("author", "").lower()
                desc_match = query_lower in model_info.get("description", "").lower()
                
                if not (name_match or author_match or desc_match):
                    continue
            
            # Check tags
            if tags:
                model_tags = model_info.get("tags", [])
                if not all(tag in model_tags for tag in tags):
                    continue
            
            # Check library
            if library and model_info.get("library") != library:
                continue
            
            # Check size range
            if size_range:
                min_size, max_size = size_range
                model_size = model_info.get("size", 0)
                
                if model_size < min_size:
                    continue
                    
                if max_size > 0 and model_size > max_size:
                    continue
            
            # Check compatibility tier
            if compatibility_tier:
                model_tiers = model_info.get("compatibility", {}).get("compatible_tiers", [])
                if compatibility_tier not in model_tiers:
                    continue
            
            results.append(model_info)
        
        return results
    
    async def check_model_compatibility(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Check model compatibility with the system.
        
        Args:
            model_info: Model information
            
        Returns:
            Compatibility information
        """
        # Default compatibility info
        compatibility = {
            "compatible_tiers": [],
            "system_compatible": False,
            "required_vram": None,
            "compatibility_issues": []
        }
        
        # Get model size
        model_size = model_info.get("model_size", 0)
        
        if model_size is None or model_size == 0:
            # Try to estimate from disk size
            disk_size = model_info.get("disk_size", 0)
            if disk_size > 0:
                model_size = disk_size
            else:
                compatibility["compatibility_issues"].append(
                    "Could not determine model size"
                )
        
        # Check compatibility with each tier
        for tier, requirements in self.compatibility_requirements.items():
            compatible = True
            
            # Check size
            max_size = requirements.get("max_size_bytes")
            if max_size is not None and model_size > max_size:
                compatible = False
            
            # Check library compatibility
            preferred_libraries = requirements.get("preferred_libraries", [])
            model_library = model_info.get("library")
            
            if model_library and preferred_libraries and model_library not in preferred_libraries:
                compatible = False
            
            if compatible:
                compatibility["compatible_tiers"].append(tier)
        
        # Determine required VRAM
        if model_size > 0:
            # Rough estimate: model size plus 20% overhead
            required_vram = int(model_size / (1024 * 1024 * 1024) * 1.2)
            compatibility["required_vram"] = max(1, required_vram)
        
        # Check system compatibility
        try:
            # Check if system has CUDA
            cuda_available = torch.cuda.is_available()
            
            if cuda_available:
                # Get available VRAM
                device = torch.cuda.current_device()
                vram_bytes = torch.cuda.get_device_properties(device).total_memory
                vram_gb = vram_bytes / (1024 * 1024 * 1024)
                
                # Compare with required VRAM
                if compatibility["required_vram"] and vram_gb >= compatibility["required_vram"]:
                    compatibility["system_compatible"] = True
                else:
                    compatibility["compatibility_issues"].append(
                        f"Insufficient VRAM: {vram_gb:.1f} GB available, "
                        f"{compatibility['required_vram']} GB required"
                    )
            else:
                compatibility["compatibility_issues"].append("CUDA not available")
                
        except Exception as e:
            logger.warning(f"Error checking CUDA compatibility: {e}")
            compatibility["compatibility_issues"].append(f"Error checking CUDA: {e}")
        
        return compatibility
    
    async def update_registry_from_loader(self, model_loader: ModelLoader) -> int:
        """Update registry from downloaded models.
        
        Args:
            model_loader: ModelLoader instance
            
        Returns:
            Number of models updated
        """
        downloaded_models = await model_loader.get_downloaded_models()
        updated_count = 0
        
        for model_info in downloaded_models:
            model_id = model_info.get("id", "")
            
            if not model_id:
                continue
                
            # Register or update model
            await self.register_model(model_id, model_info)
            updated_count += 1
            
        logger.info(f"Updated registry with {updated_count} models")
        return updated_count


class ModelManager:
    """Main class for managing models."""
    
    def __init__(self, 
                 base_dir: Path,
                 api_token: Optional[str] = None):
        """Initialize the model manager.
        
        Args:
            base_dir: Base directory for models and cache
            api_token: Optional Hugging Face API token
        """
        self.base_dir = base_dir
        
        # Create subdirectories
        self.models_dir = base_dir / "models"
        self.cache_dir = base_dir / "cache"
        self.registry_path = base_dir / "registry.json"
        
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.hf_integration = HuggingFaceIntegration(
            api_token=api_token,
            cache_dir=self.cache_dir
        )
        
        self.model_loader = ModelLoader(
            models_dir=self.models_dir,
            cache_dir=self.cache_dir,
            huggingface_integration=self.hf_integration
        )
        
        self.model_registry = ModelRegistry(
            registry_path=self.registry_path,
            models_dir=self.models_dir
        )
        
        # Active download tasks
        self.download_tasks: Dict[str, asyncio.Task] = {}
        
        # Online/offline mode
        self.online_mode: Optional[bool] = None
        
        logger.info("Model manager initialized")
    
    async def check_api_connection(self) -> bool:
        """Check if Hugging Face API is reachable (for online mode)."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"https://huggingface.co/api/models?limit=1", timeout=5) as response:
                    if response.status == 200:
                        logger.info("Hugging Face API is reachable (online mode)")
                        return True
        except Exception as e:
            logger.warning(f"Hugging Face API not reachable: {e}")
        logger.info("Switching to offline mode")
        return False
    
    async def set_online_mode(self, force: Optional[bool] = None) -> None:
        """Set online/offline mode explicitly or auto-detect if force is None."""
        if force is not None:
            self.online_mode = force
        else:
            self.online_mode = await self.check_api_connection()
    
    async def search_models(self, 
                          query: str = "",
                          tags: Optional[List[str]] = None,
                          model_type: Optional[str] = None,
                          limit: int = 20) -> List[Dict[str, Any]]:
        """Search for models on Hugging Face Hub or offline fallback."""
        # Определяем режим, если не установлен
        if self.online_mode is None:
            await self.set_online_mode()
        
        if self.online_mode:
            try:
                return await self.hf_integration.search_models(
                    query=query,
                    filter_tags=tags,
                    model_type=model_type,
                    limit=limit
                )
            except Exception as e:
                logger.warning(f"Falling back to offline mode due to error: {e}")
                self.online_mode = False
        # Оффлайн режим
        return await self.hf_integration.search_models(
            query=query,
            filter_tags=tags,
            model_type=model_type,
            limit=limit
        )
    
    async def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a model.
        
        Args:
            model_id: Model ID
            
        Returns:
            Model information or None if not found
        """
        # Check if model is registered
        registered_info = self.model_registry.get_model(model_id)
        
        if registered_info:
            return registered_info
            
        # Get info from Hugging Face
        return await self.hf_integration.get_model_info(model_id)
    
    async def download_model(self, 
                           model_id: str,
                           progress_callback: Optional[Callable[[float], None]] = None) -> Dict[str, Any]:
        """Download a model from Hugging Face.
        
        Args:
            model_id: Model ID
            progress_callback: Optional callback for progress updates (0-1)
            
        Returns:
            Download status
        """
        # Check if model is already being downloaded
        if model_id in self.download_tasks:
            task = self.download_tasks[model_id]
            
            if not task.done():
                return {
                    "status": "in_progress",
                    "message": "Model download already in progress"
                }
            else:
                # Clean up completed task
                del self.download_tasks[model_id]
        
        # Check if model is already downloaded
        if self.model_loader.is_model_downloaded(model_id):
            return {
                "status": "already_downloaded",
                "message": "Model is already downloaded"
            }
            
        # Start download task
        download_task = asyncio.create_task(
            self._download_model_task(model_id, progress_callback)
        )
        
        self.download_tasks[model_id] = download_task
        
        return {
            "status": "started",
            "message": "Model download started"
        }
    
    async def _download_model_task(self, 
                                 model_id: str,
                                 progress_callback: Optional[Callable[[float], None]] = None) -> None:
        """Background task for downloading a model.
        
        Args:
            model_id: Model ID
            progress_callback: Optional callback for progress updates
        """
        try:
            # Download model
            success = await self.model_loader.download_model(model_id, progress_callback)
            
            if success:
                # Get model info
                model_info = await self.hf_integration.get_model_info(model_id)
                
                if model_info:
                    # Update with local info
                    downloaded_models = await self.model_loader.get_downloaded_models()
                    
                    for downloaded in downloaded_models:
                        if downloaded.get("id", "") == model_id:
                            model_info.update({
                                "disk_size": downloaded.get("disk_size", 0),
                                "download_path": downloaded.get("download_path", ""),
                                "is_complete": downloaded.get("is_complete", False)
                            })
                            break
                    
                    # Register model
                    await self.model_registry.register_model(model_id, model_info)
        
        except Exception as e:
            logger.error(f"Error downloading model {model_id}: {e}")
            
        finally:
            # Clean up task
            if model_id in self.download_tasks:
                del self.download_tasks[model_id]
    
    async def get_download_status(self, model_id: str) -> Dict[str, Any]:
        """Get the status of a model download.
        
        Args:
            model_id: Model ID
            
        Returns:
            Download status
        """
        # Check if model is already downloaded
        if self.model_loader.is_model_downloaded(model_id):
            return {
                "status": "downloaded",
                "message": "Model is downloaded",
                "progress": 1.0
            }
            
        # Check if model is being downloaded
        if model_id in self.download_tasks:
            task = self.download_tasks[model_id]
            
            if task.done():
                # Check if download was successful
                try:
                    task.result()
                    return {
                        "status": "completed",
                        "message": "Download completed",
                        "progress": 1.0
                    }
                except Exception as e:
                    return {
                        "status": "failed",
                        "message": f"Download failed: {e}",
                        "progress": 0.0
                    }
            else:
                return {
                    "status": "in_progress",
                    "message": "Download in progress",
                    "progress": 0.5  # We don't know actual progress here
                }
                
        # Model is not downloaded or being downloaded
        return {
            "status": "not_downloaded",
            "message": "Model is not downloaded",
            "progress": 0.0
        }
    
    async def delete_model(self, model_id: str) -> Dict[str, Any]:
        """Delete a downloaded model.
        
        Args:
            model_id: Model ID
            
        Returns:
            Delete status
        """
        # Check if model is being downloaded
        if model_id in self.download_tasks:
            task = self.download_tasks[model_id]
            
            if not task.done():
                # Cancel download
                task.cancel()
                
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                
                # Clean up task
                del self.download_tasks[model_id]
        
        # Delete model from disk
        success = self.model_loader.delete_model(model_id)
        
        if success:
            # Unregister model
            self.model_registry.unregister_model(model_id)
            
            return {
                "status": "deleted",
                "message": "Model deleted successfully"
            }
        else:
            return {
                "status": "failed",
                "message": "Failed to delete model"
            }
    
    async def get_downloaded_models(self) -> List[Dict[str, Any]]:
        """Get information about all downloaded models.
        
        Returns:
            List of model information dictionaries
        """
        # Update registry first
        await self.model_registry.update_registry_from_loader(self.model_loader)
        
        # Get all registered models
        return self.model_registry.get_all_models()
    
    async def find_downloaded_models(self, 
                                   query: str = "",
                                   tags: Optional[List[str]] = None,
                                   library: Optional[str] = None,
                                   compatibility_tier: Optional[str] = None) -> List[Dict[str, Any]]:
        """Find downloaded models matching criteria.
        
        Args:
            query: Search query
            tags: List of tags to filter by
            library: Library to filter by
            compatibility_tier: Compatibility tier to filter by
            
        Returns:
            List of matching model information dictionaries
        """
        return self.model_registry.find_models(
            query=query,
            tags=tags,
            library=library,
            compatibility_tier=compatibility_tier
        )
    
    async def get_available_tags(self) -> List[str]:
        """Get list of all available model tags.
        
        Returns:
            List of tag strings
        """
        return await self.hf_integration.get_model_tags()
    
    async def get_system_compatibility_info(self) -> Dict[str, Any]:
        """Get information about system compatibility.
        
        Returns:
            Compatibility information
        """
        compatibility = {
            "cuda_available": False,
            "vram_gb": 0,
            "compatible_tiers": [],
            "issues": []
        }
        
        try:
            # Check if system has CUDA
            compatibility["cuda_available"] = torch.cuda.is_available()
            
            if compatibility["cuda_available"]:
                # Get available VRAM
                device = torch.cuda.current_device()
                vram_bytes = torch.cuda.get_device_properties(device).total_memory
                vram_gb = vram_bytes / (1024 * 1024 * 1024)
                
                compatibility["vram_gb"] = vram_gb
                
                # Determine compatible tiers
                for tier, requirements in self.model_registry.compatibility_requirements.items():
                    if vram_gb >= requirements["min_vram"]:
                        compatibility["compatible_tiers"].append(tier)
            else:
                compatibility["issues"].append("CUDA not available")
                
        except Exception as e:
            logger.warning(f"Error checking CUDA compatibility: {e}")
            compatibility["issues"].append(f"Error checking CUDA: {e}")
        
        return compatibility
