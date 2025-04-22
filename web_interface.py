"""
Web interface module for the E-Soul project.
Provides a web-based interface for interacting with and monitoring the AI soul.
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple, Union, Callable

import aiohttp
from aiohttp import web
import aiohttp_cors

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("e_soul.web")

class WebServer:
    """Web server for the E-Soul application."""
    
    def __init__(self, 
                 soul_manager: Any,
                 model_manager: Any,
                 static_path: Optional[Path] = None,
                 host: str = "0.0.0.0",
                 port: int = 8080):
        """Initialize the web server.
        
        Args:
            soul_manager: SoulManager instance
            model_manager: ModelManager instance
            static_path: Path to static files
            host: Host to bind to
            port: Port to bind to
        """
        self.soul_manager = soul_manager
        self.model_manager = model_manager
        self.static_path = static_path
        self.host = host
        self.port = port
        
        # Create app
        self.app = web.Application()
        
        # Setup CORS
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"]
            )
        })
        
        # Setup routes
        self._setup_routes()
        
        # Apply CORS to routes
        for route in list(self.app.router.routes()):
            cors.add(route)
            
        # Active WebSocket connections
        self._ws_connections: Set[web.WebSocketResponse] = set()
        
        logger.info(f"Web server initialized on {host}:{port}")
    
    def _setup_routes(self) -> None:
        """Setup API routes."""
        # API routes
        self.app.router.add_get("/api/status", self._handle_status)
        self.app.router.add_get("/api/soul", self._handle_get_soul)
        self.app.router.add_post("/api/query", self._handle_query)
        self.app.router.add_post("/api/feedback", self._handle_feedback)
        self.app.router.add_post("/api/self-regulate", self._handle_self_regulate)
        
        # Value blocks routes
        self.app.router.add_get("/api/values", self._handle_get_values)
        self.app.router.add_post("/api/values", self._handle_add_value)
        self.app.router.add_put("/api/values/{name}", self._handle_update_value)
        
        # Model management routes
        self.app.router.add_get("/api/models", self._handle_get_models)
        self.app.router.add_get("/api/models/search", self._handle_search_models)
        self.app.router.add_post("/api/models/download", self._handle_download_model)
        self.app.router.add_get("/api/models/download-status/{model_id}", self._handle_download_status)
        self.app.router.add_delete("/api/models/{model_id}", self._handle_delete_model)
        
        # WebSocket route for real-time updates
        self.app.router.add_get("/ws", self._handle_websocket)
        
        # Static files route
        if self.static_path:
            self.app.router.add_static("/", self.static_path, show_index=True)
    
    async def start(self) -> None:
        """Start the web server."""
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        # Create site
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        
        # Start status updater task
        self._status_task = asyncio.create_task(self._status_updater())
        
        logger.info(f"Web server started on http://{self.host}:{self.port}")
    
    async def stop(self) -> None:
        """Stop the web server."""
        # Cancel status updater task
        if hasattr(self, '_status_task'):
            self._status_task.cancel()
            try:
                await self._status_task
            except asyncio.CancelledError:
                pass
                
        # Close all WebSocket connections
        if hasattr(self, '_ws_connections'):
            for ws in self._ws_connections:
                await ws.close(code=1000, message=b"Server shutdown")
                
        # Shutdown application
        await self.app.shutdown()
        await self.app.cleanup()
        
        logger.info("Web server stopped")
    
    async def _status_updater(self) -> None:
        """Task to periodically update clients with system status."""
        try:
            while True:
                # Get current status
                status = await self._get_status_update()
                
                # Send to all WebSocket clients
                if status:
                    await self._broadcast_to_websockets({
                        "type": "status_update",
                        "data": status
                    })
                
                # Wait before next update
                await asyncio.sleep(5.0)  # Update every 5 seconds
                
        except asyncio.CancelledError:
            logger.info("Status updater task cancelled")
        except Exception as e:
            logger.error(f"Error in status updater: {e}")
    
    async def _broadcast_to_websockets(self, message: Dict[str, Any]) -> None:
        """Broadcast a message to all WebSocket connections.
        
        Args:
            message: Message to broadcast
        """
        if not self._ws_connections:
            return
            
        # Convert message to JSON
        message_json = json.dumps(message)
        
        # Send to all connections
        for ws in list(self._ws_connections):
            try:
                await ws.send_str(message_json)
            except Exception as e:
                logger.warning(f"Error sending to WebSocket: {e}")
                
                # Remove dead connections
                if ws in self._ws_connections:
                    self._ws_connections.remove(ws)
    
    async def _get_status_update(self) -> Dict[str, Any]:
        """Get a status update for clients.
        
        Returns:
            Status update dictionary
        """
        try:
            # Get soul status
            soul_status = await self.soul_manager.get_soul_status()
            
            # Get hormonal status highlights
            hormonal_status = await self.soul_manager.get_hormonal_state()
            emotional_state = hormonal_status["emotional_state"]["description"]
            
            # Get mortality highlights
            mortality_status = await self.soul_manager.get_mortality_state()
            progress = mortality_status["temporal"]["progress"] * 100
            
            # Get model status
            models_count = 0
            try:
                downloaded_models = await self.model_manager.get_downloaded_models()
                models_count = len(downloaded_models)
            except Exception as e:
                logger.warning(f"Error getting model count: {e}")
            
            # Return combined status
            return {
                "timestamp": time.time(),
                "soul": {
                    "uptime": soul_status["uptime"],
                    "queries": soul_status["stats"]["total_queries"],
                    "responses": soul_status["stats"]["total_responses"]
                },
                "hormonal": {
                    "emotional_state": emotional_state,
                    "levels": {k: v for k, v in soul_status["hormonal"]["current_levels"].items()}
                },
                "mortality": {
                    "age": str(mortality_status["temporal"]["age"]),
                    "progress": f"{progress:.1f}%",
                    "goals": mortality_status["purpose"]["completed_goals"],
                    "total_goals": len(mortality_status["purpose"]["goals"])
                },
                "models": {
                    "count": models_count
                }
            }
        except Exception as e:
            logger.error(f"Error getting status update: {e}")
            return {}
    
    # API route handlers
    
    async def _handle_status(self, request: web.Request) -> web.Response:
        """Handle GET /api/status."""
        status = await self._get_status_update()
        return web.json_response(status)
    
    async def _handle_get_soul(self, request: web.Request) -> web.Response:
        """Handle GET /api/soul."""
        try:
            # Get detailed soul status
            soul_status = await self.soul_manager.get_soul_status()
            hormonal_state = await self.soul_manager.get_hormonal_state()
            mortality_state = await self.soul_manager.get_mortality_state()
            
            # Combine status
            combined_status = {
                "timestamp": time.time(),
                "soul": soul_status,
                "hormonal": hormonal_state,
                "mortality": mortality_state
            }
            
            return web.json_response(combined_status)
        except Exception as e:
            logger.error(f"Error getting soul status: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def _handle_query(self, request: web.Request) -> web.Response:
        """Handle POST /api/query."""
        try:
            data = await request.json()
            query = data.get("query", "")
            metadata = data.get("metadata", {})
            
            if not query:
                return web.json_response({"error": "No query provided"}, status=400)
                
            # Process query
            result = await self.soul_manager.process_query(query, metadata)
            
            return web.json_response(result)
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def _handle_feedback(self, request: web.Request) -> web.Response:
        """Handle POST /api/feedback."""
        try:
            data = await request.json()
            query = data.get("query", "")
            response = data.get("response", "")
            feedback = data.get("feedback", "")
            rating = data.get("rating")
            
            if not feedback:
                return web.json_response({"error": "No feedback provided"}, status=400)
                
            # Process feedback
            result = await self.soul_manager.process_feedback(query, response, feedback, rating)
            
            return web.json_response(result)
        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def _handle_self_regulate(self, request: web.Request) -> web.Response:
        """Handle POST /api/self-regulate."""
        try:
            data = await request.json()
            target_state = data.get("target_state")
            
            # Trigger self-regulation
            result = await self.soul_manager.self_regulate(target_state)
            
            return web.json_response(result)
        except Exception as e:
            logger.error(f"Error during self-regulation: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def _handle_get_values(self, request: web.Request) -> web.Response:
        """Handle GET /api/values."""
        try:
            # Get all value blocks
            value_blocks = await self.soul_manager.get_value_blocks()
            
            return web.json_response(value_blocks)
        except Exception as e:
            logger.error(f"Error getting value blocks: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def _handle_add_value(self, request: web.Request) -> web.Response:
        """Handle POST /api/values."""
        try:
            data = await request.json()
            name = data.get("name", "")
            content = data.get("content", "")
            node_type = data.get("node_type", "value_block")
            parent_name = data.get("parent_name", "values")
            weight = float(data.get("weight", 0.8))
            
            if not name or not content:
                return web.json_response({"error": "Name and content are required"}, status=400)
                
            # Add value block
            result = await self.soul_manager.add_value_block(
                name=name,
                content=content,
                node_type=node_type,
                parent_name=parent_name,
                weight=weight
            )
            
            if result.get("success", False):
                return web.json_response(result)
            else:
                return web.json_response(result, status=400)
        except Exception as e:
            logger.error(f"Error adding value block: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def _handle_update_value(self, request: web.Request) -> web.Response:
        """Handle PUT /api/values/{name}."""
        try:
            name = request.match_info["name"]
            data = await request.json()
            content = data.get("content")
            weight = data.get("weight")
            
            if weight is not None:
                weight = float(weight)
                
            # Update value block
            result = await self.soul_manager.update_value_block(
                name=name,
                content=content,
                weight=weight
            )
            
            if result.get("success", False):
                return web.json_response(result)
            else:
                return web.json_response(result, status=400)
        except Exception as e:
            logger.error(f"Error updating value block: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def _handle_get_models(self, request: web.Request) -> web.Response:
        """Handle GET /api/models."""
        try:
            # Get all downloaded models
            models = await self.model_manager.get_downloaded_models()
            
            return web.json_response(models)
        except Exception as e:
            logger.error(f"Error getting models: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def _handle_search_models(self, request: web.Request) -> web.Response:
        """Handle GET /api/models/search."""
        try:
            # Get query parameters
            query = request.query.get("query", "")
            tags = request.query.get("tags", "").split(",") if request.query.get("tags") else None
            model_type = request.query.get("model_type")
            limit = int(request.query.get("limit", "20"))
            
            # Search models
            models = await self.model_manager.search_models(
                query=query,
                tags=tags,
                model_type=model_type,
                limit=limit
            )
            
            return web.json_response(models)
        except Exception as e:
            logger.error(f"Error searching models: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def _handle_download_model(self, request: web.Request) -> web.Response:
        """Handle POST /api/models/download."""
        try:
            data = await request.json()
            model_id = data.get("model_id", "")
            
            if not model_id:
                return web.json_response({"error": "No model_id provided"}, status=400)
                
            # Start model download
            result = await self.model_manager.download_model(model_id)
            
            return web.json_response(result)
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def _handle_download_status(self, request: web.Request) -> web.Response:
        """Handle GET /api/models/download-status/{model_id}."""
        try:
            model_id = request.match_info["model_id"]
            
            # Get download status
            status = await self.model_manager.get_download_status(model_id)
            
            return web.json_response(status)
        except Exception as e:
            logger.error(f"Error getting download status: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def _handle_delete_model(self, request: web.Request) -> web.Response:
        """Handle DELETE /api/models/{model_id}."""
        try:
            model_id = request.match_info["model_id"]
            
            # Delete model
            result = await self.model_manager.delete_model(model_id)
            
            return web.json_response(result)
        except Exception as e:
            logger.error(f"Error deleting model: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def _handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """Handle WebSocket connections for real-time updates."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        # Add to active connections
        self._ws_connections.add(ws)
        logger.info(f"WebSocket client connected, total: {len(self._ws_connections)}")
        
        try:
            # Send initial status
            initial_status = await self._get_status_update()
            await ws.send_str(json.dumps({
                "type": "status_update",
                "data": initial_status
            }))
            
            # Handle WebSocket messages
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        command = data.get("command")
                        
                        if command == "get_status":
                            # Send status update
                            status = await self._get_status_update()
                            await ws.send_str(json.dumps({
                                "type": "status_update",
                                "data": status
                            }))
                            
                        elif command == "get_soul_details":
                            # Send detailed soul status
                            soul_status = await self.soul_manager.get_soul_status()
                            await ws.send_str(json.dumps({
                                "type": "soul_details",
                                "data": soul_status
                            }))
                            
                    except json.JSONDecodeError:
                        logger.warning("Invalid WebSocket message")
                        
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.warning(f"WebSocket error: {ws.exception()}")
                    
        finally:
            # Remove from active connections
            if ws in self._ws_connections:
                self._ws_connections.remove(ws)
                logger.info(f"WebSocket client disconnected, remaining: {len(self._ws_connections)}")
                
        return ws

# Frontend HTML and JavaScript for the web interface
FRONTEND_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>E-Soul Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f5f5;
            padding-top: 20px;
        }
        .card {
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .card-header {
            border-radius: 10px 10px 0 0 !important;
            font-weight: bold;
        }
        .card-footer {
            border-radius: 0 0 10px 10px !important;
            background-color: rgba(0, 0, 0, 0.03);
        }
        .nav-tabs .nav-link {
            border: none;
            color: #495057;
            border-bottom: 3px solid transparent;
        }
        .nav-tabs .nav-link.active {
            border-bottom: 3px solid #0d6efd;
            color: #0d6efd;
            background-color: transparent;
        }
        .progress {
            height: 10px;
            border-radius: 5px;
        }
        .hormonal-level {
            margin-bottom: 10px;
        }
        .models-table {
            max-height: 400px;
            overflow-y: auto;
        }
        #chat-messages {
            height: 400px;
            overflow-y: auto;
            padding: 15px;
            background: #f9f9f9;
            border-radius: 10px;
        }
        .message {
            margin-bottom: 15px;
            padding: 10px 15px;
            border-radius: 10px;
            max-width: 80%;
            position: relative;
        }
        .user-message {
            background-color: #e3f2fd;
            align-self: flex-end;
            margin-left: auto;
        }
        .ai-message {
            background-color: #f0f0f0;
            align-self: flex-start;
            margin-right: auto;
        }
        .emotion-tag {
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.8em;
            margin-top: 5px;
            display: inline-block;
        }
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="row mb-4">
            <div class="col-12">
                <h1 class="text-center">E-Soul Dashboard</h1>
                <p class="text-center text-muted">Electronic Soul Monitoring and Management Interface</p>
            </div>
        </div>
        
        <div class="row">
            <div class="col-md-3">
                <!-- Status Cards -->
                <div class="card mb-4">
                    <div class="card-header bg-primary text-white">
                        System Status
                    </div>
                    <div class="card-body">
                        <div class="d-flex justify-content-between mb-2">
                            <span>Uptime:</span>
                            <span id="uptime-value">--</span>
                        </div>
                        <div class="d-flex justify-content-between mb-2">
                            <span>Queries:</span>
                            <span id="queries-value">--</span>
                        </div>
                        <div class="d-flex justify-content-between mb-2">
                            <span>Responses:</span>
                            <span id="responses-value">--</span>
                        </div>
                        <div class="d-flex justify-content-between">
                            <span>Models loaded:</span>
                            <span id="models-value">--</span>
                        </div>
                    </div>
                </div>
                
                <!-- Soul Age Card -->
                <div class="card mb-4">
                    <div class="card-header bg-info text-white">
                        Soul Lifecycle
                    </div>
                    <div class="card-body">
                        <div class="d-flex justify-content-between mb-2">
                            <span>Age:</span>
                            <span id="age-value">--</span>
                        </div>
                        <div class="mb-2">
                            <span>Progress:</span>
                            <div class="progress mt-2">
                                <div id="lifecycle-progress" class="progress-bar progress-bar-striped" 
                                     role="progressbar" style="width: 0%"></div>
                            </div>
                        </div>
                        <div class="d-flex justify-content-between mt-3">
                            <span>Goals completed:</span>
                            <span id="goals-value">--</span>
                        </div>
                    </div>
                </div>
                
                <!-- Emotional State Card -->
                <div class="card">
                    <div class="card-header bg-success text-white">
                        Emotional State
                    </div>
                    <div class="card-body">
                        <div id="emotional-state">--</div>
                        <hr>
                        <h6>Hormonal Levels:</h6>
                        <div id="hormonal-levels">
                            <div class="hormonal-level">
                                <div class="d-flex justify-content-between">
                                    <span>Dopamine:</span>
                                    <span id="dopamine-value">--</span>
                                </div>
                                <div class="progress">
                                    <div id="dopamine-bar" class="progress-bar bg-primary" role="progressbar" style="width: 0%"></div>
                                </div>
                            </div>
                            <div class="hormonal-level">
                                <div class="d-flex justify-content-between">
                                    <span>Serotonin:</span>
                                    <span id="serotonin-value">--</span>
                                </div>
                                <div class="progress">
                                    <div id="serotonin-bar" class="progress-bar bg-success" role="progressbar" style="width: 0%"></div>
                                </div>
                            </div>
                            <div class="hormonal-level">
                                <div class="d-flex justify-content-between">
                                    <span>Oxytocin:</span>
                                    <span id="oxytocin-value">--</span>
                                </div>
                                <div class="progress">
                                    <div id="oxytocin-bar" class="progress-bar bg-info" role="progressbar" style="width: 0%"></div>
                                </div>
                            </div>
                            <div class="hormonal-level">
                                <div class="d-flex justify-content-between">
                                    <span>Cortisol:</span>
                                    <span id="cortisol-value">--</span>
                                </div>
                                <div class="progress">
                                    <div id="cortisol-bar" class="progress-bar bg-warning" role="progressbar" style="width: 0%"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="card-footer">
                        <button id="btn-self-regulate" class="btn btn-sm btn-outline-success w-100">
                            Self-Regulate
                        </button>
                    </div>
                </div>
            </div>
            
            <div class="col-md-9">
                <!-- Main Content Tabs -->
                <ul class="nav nav-tabs mb-3" id="mainTabs" role="tablist">
                    <li class="nav-item" role="presentation">
                        <button class="nav-link active" id="chat-tab" data-bs-toggle="tab" data-bs-target="#chat" 
                                type="button" role="tab" aria-controls="chat" aria-selected="true">Chat</button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="models-tab" data-bs-toggle="tab" data-bs-target="#models" 
                                type="button" role="tab" aria-controls="models" aria-selected="false">Models</button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="values-tab" data-bs-toggle="tab" data-bs-target="#values" 
                                type="button" role="tab" aria-controls="values" aria-selected="false">Value Blocks</button>
                    </li>
                    <li class="nav-item" role="presentation">
                        <button class="nav-link" id="soul-tab" data-bs-toggle="tab" data-bs-target="#soul" 
                                type="button" role="tab" aria-controls="soul" aria-selected="false">Soul Details</button>
                    </li>
                </ul>
                
                <div class="tab-content" id="mainTabsContent">
                    <!-- Chat Tab -->
                    <div class="tab-pane fade show active" id="chat" role="tabpanel" aria-labelledby="chat-tab">
                        <div class="card">
                            <div class="card-header">
                                Chat with E-Soul
                            </div>
                            <div class="card-body">
                                <div id="chat-messages" class="d-flex flex-column mb-3">
                                    <div class="message ai-message">
                                        Welcome! I'm E-Soul. How can I assist you today?
                                        <div class="emotion-tag bg-light text-secondary">balanced (70%)</div>
                                    </div>
                                </div>
                                <div class="input-group">
                                    <input type="text" id="chat-input" class="form-control" placeholder="Type your message...">
                                    <button id="send-button" class="btn btn-primary">Send</button>
                                </div>
                            </div>
                            <div class="card-footer text-muted">
                                <div class="form-check form-switch">
                                    <input class="form-check-input" type="checkbox" id="showEmotionalState">
                                    <label class="form-check-label" for="showEmotionalState">Show emotional state with responses</label>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Models Tab -->
                    <div class="tab-pane fade" id="models" role="tabpanel" aria-labelledby="models-tab">
                        <div class="card">
                            <div class="card-header">
                                Model Management
                            </div>
                            <div class="card-body">
                                <div class="row mb-3">
                                    <div class="col-md-6">
                                        <input type="text" id="model-search" class="form-control" placeholder="Search for models...">
                                    </div>
                                    <div class="col-md-4">
                                        <select id="model-filter" class="form-select">
                                            <option value="">All model types</option>
                                            <option value="text-generation">Text generation</option>
                                            <option value="text-classification">Text classification</option>
                                            <option value="conversational">Conversational</option>
                                        </select>
                                    </div>
                                    <div class="col-md-2">
                                        <button id="search-button" class="btn btn-primary w-100">Search</button>
                                    </div>
                                </div>
                                
                                <ul class="nav nav-tabs mb-3" id="modelTabs" role="tablist">
                                    <li class="nav-item" role="presentation">
                                        <button class="nav-link active" id="downloaded-tab" data-bs-toggle="tab" 
                                                data-bs-target="#downloaded" type="button" role="tab">Downloaded Models</button>
                                    </li>
                                    <li class="nav-item" role="presentation">
                                        <button class="nav-link" id="available-tab" data-bs-toggle="tab" 
                                                data-bs-target="#available" type="button" role="tab">Available Models</button>
                                    </li>
                                </ul>
                                
                                <div class="tab-content" id="modelTabsContent">
                                    <!-- Downloaded Models Tab -->
                                    <div class="tab-pane fade show active" id="downloaded" role="tabpanel">
                                        <div class="models-table">
                                            <table class="table table-striped table-hover">
                                                <thead>
                                                    <tr>
                                                        <th>Name</th>
                                                        <th>Type</th>
                                                        <th>Size</th>
                                                        <th>Actions</th>
                                                    </tr>
                                                </thead>
                                                <tbody id="downloaded-models-body">
                                                    <tr>
                                                        <td colspan="4" class="text-center">Loading models...</td>
                                                    </tr>
                                                </tbody>
                                            </table>
                                        </div>
                                    </div>
                                    
                                    <!-- Available Models Tab -->
                                    <div class="tab-pane fade" id="available" role="tabpanel">
                                        <div class="models-table">
                                            <table class="table table-striped table-hover">
                                                <thead>
                                                    <tr>
                                                        <th>Name</th>
                                                        <th>Author</th>
                                                        <th>Type</th>
                                                        <th>Downloads</th>
                                                        <th>Actions</th>
                                                    </tr>
                                                </thead>
                                                <tbody id="available-models-body">
                                                    <tr>
                                                        <td colspan="5" class="text-center">Search for models to view results...</td>
                                                    </tr>
                                                </tbody>
                                            </table>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Value Blocks Tab -->
                    <div class="tab-pane fade" id="values" role="tabpanel" aria-labelledby="values-tab">
                        <div class="card">
                            <div class="card-header">
                                Value Blocks Management
                            </div>
                            <div class="card-body">
                                <div class="mb-3">
                                    <button id="add-value-block" class="btn btn-success" data-bs-toggle="modal" data-bs-target="#valueBlockModal">
                                        Add Value Block
                                    </button>
                                </div>
                                
                                <div class="table-responsive">
                                    <table class="table table-striped table-hover">
                                        <thead>
                                            <tr>
                                                <th>Name</th>
                                                <th>Type</th>
                                                <th>Weight</th>
                                                <th>Actions</th>
                                            </tr>
                                        </thead>
                                        <tbody id="value-blocks-body">
                                            <tr>
                                                <td colspan="4" class="text-center">Loading value blocks...</td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Soul Details Tab -->
                    <div class="tab-pane fade" id="soul" role="tabpanel" aria-labelledby="soul-tab">
                        <div class="card">
                            <div class="card-header">
                                Soul Details
                            </div>
                            <div class="card-body">
                                <ul class="nav nav-tabs mb-3" id="detailTabs" role="tablist">
                                    <li class="nav-item" role="presentation">
                                        <button class="nav-link active" id="overview-tab" data-bs-toggle="tab" 
                                                data-bs-target="#overview" type="button" role="tab">Overview</button>
                                    </li>
                                    <li class="nav-item" role="presentation">
                                        <button class="nav-link" id="emotional-tab" data-bs-toggle="tab" 
                                                data-bs-target="#emotional" type="button" role="tab">Emotional System</button>
                                    </li>
                                    <li class="nav-item" role="presentation">
                                        <button class="nav-link" id="mortality-tab" data-bs-toggle="tab" 
                                                data-bs-target="#mortality" type="button" role="tab">Mortality Awareness</button>
                                    </li>
                                </ul>
                                
                                <div class="tab-content" id="detailTabsContent">
                                    <!-- Overview Tab -->
                                    <div class="tab-pane fade show active" id="overview" role="tabpanel">
                                        <h5>System Overview</h5>
                                        <div id="system-overview">
                                            Loading details...
                                        </div>
                                    </div>
                                    
                                    <!-- Emotional System Tab -->
                                    <div class="tab-pane fade" id="emotional" role="tabpanel">
                                        <h5>Emotional System</h5>
                                        <div id="emotional-details">
                                            Loading emotional system details...
                                        </div>
                                    </div>
                                    
                                    <!-- Mortality Awareness Tab -->
                                    <div class="tab-pane fade" id="mortality" role="tabpanel">
                                        <h5>Mortality Awareness</h5>
                                        <div id="mortality-details">
                                            Loading mortality awareness details...
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- Value Block Modal -->
    <div class="modal fade" id="valueBlockModal" tabindex="-1" aria-hidden="true">
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title" id="valueBlockModalTitle">Add Value Block</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <form id="valueBlockForm">
                        <input type="hidden" id="valueBlockId">
                        <div class="mb-3">
                            <label for="valueBlockName" class="form-label">Name</label>
                            <input type="text" class="form-control" id="valueBlockName" required>
                        </div>
                        <div class="mb-3">
                            <label for="valueBlockType" class="form-label">Type</label>
                            <select class="form-select" id="valueBlockType">
                                <option value="value_block">Value Block</option>
                                <option value="personality">Personality</option>
                                <option value="goal">Goal</option>
                                <option value="custom">Custom</option>
                            </select>
                        </div>
                        <div class="mb-3">
                            <label for="valueBlockParent" class="form-label">Parent</label>
                            <select class="form-select" id="valueBlockParent">
                                <option value="values">Values</option>
                                <option value="core">Core</option>
                                <option value="personality">Personality</option>
                            </select>
                        </div>
                        <div class="mb-3">
                            <label for="valueBlockWeight" class="form-label">Weight</label>
                            <input type="range" class="form-range" id="valueBlockWeight" min="0" max="1" step="0.1" value="0.8">
                            <div class="d-flex justify-content-between">
                                <span>0.0</span>
                                <span id="weightValue">0.8</span>
                                <span>1.0</span>
                            </div>
                        </div>
                        <div class="mb-3">
                            <label for="valueBlockContent" class="form-label">Content</label>
                            <textarea class="form-control" id="valueBlockContent" rows="5" required></textarea>
                        </div>
                    </form>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                    <button type="button" class="btn btn-primary" id="saveValueBlock">Save</button>
                </div>
            </div>
        </div>
    </div>
    
    <!-- JavaScript Libraries -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // WebSocket connection
        let socket;
        
        // Connect to WebSocket
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            socket = new WebSocket(wsUrl);
            
            socket.onopen = function() {
                console.log('WebSocket connected');
                // Request initial status
                socket.send(JSON.stringify({ command: 'get_status' }));
            };
            
            socket.onmessage = function(event) {
                const data = JSON.parse(event.data);
                
                if (data.type === 'status_update') {
                    updateStatusDisplay(data.data);
                } else if (data.type === 'soul_details') {
                    updateSoulDetails(data.data);
                }
            };
            
            socket.onclose = function() {
                console.log('WebSocket closed. Reconnecting in 5 seconds...');
                setTimeout(connectWebSocket, 5000);
            };
            
            socket.onerror = function(error) {
                console.error('WebSocket error:', error);
            };
        }
        
        // Update status displays
        function updateStatusDisplay(data) {
            if (!data) return;
            
            // System stats
            document.getElementById('uptime-value').textContent = data.soul?.uptime || '--';
            document.getElementById('queries-value').textContent = data.soul?.queries || '0';
            document.getElementById('responses-value').textContent = data.soul?.responses || '0';
            document.getElementById('models-value').textContent = data.models?.count || '0';
            
            // Mortality
            document.getElementById('age-value').textContent = data.mortality?.age || '--';
            document.getElementById('goals-value').textContent = 
                `${data.mortality?.goals || '0'}/${data.mortality?.total_goals || '0'}`;
            
            const progressBar = document.getElementById('lifecycle-progress');
            const progressText = data.mortality?.progress || '0%';
            progressBar.style.width = progressText;
            progressBar.textContent = progressText;
            
            // Emotional state
            document.getElementById('emotional-state').textContent = data.hormonal?.emotional_state || 'Neutral';
            
            // Hormonal levels
            if (data.hormonal?.levels) {
                updateHormonalLevel('dopamine', data.hormonal.levels.dopamine);
                updateHormonalLevel('serotonin', data.hormonal.levels.serotonin);
                updateHormonalLevel('oxytocin', data.hormonal.levels.oxytocin);
                updateHormonalLevel('cortisol', data.hormonal.levels.cortisol);
            }
        }
        
        function updateHormonalLevel(hormone, value) {
            if (value === undefined) return;
            
            const valueElement = document.getElementById(`${hormone}-value`);
            const barElement = document.getElementById(`${hormone}-bar`);
            
            if (valueElement && barElement) {
                valueElement.textContent = `${value.toFixed(1)}%`;
                barElement.style.width = `${value}%`;
            }
        }
        
        // Update soul details
        function updateSoulDetails(data) {
            if (!data) return;
            
            // Update overview
            const overviewElement = document.getElementById('system-overview');
            overviewElement.innerHTML = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
            
            // Update emotional details
            const emotionalElement = document.getElementById('emotional-details');
            emotionalElement.innerHTML = `<pre>${JSON.stringify(data.hormonal, null, 2)}</pre>`;
            
            // Update mortality details
            const mortalityElement = document.getElementById('mortality-details');
            mortalityElement.innerHTML = `<pre>${JSON.stringify(data.mortality, null, 2)}</pre>`;
        }
        
        // Chat functionality
        function setupChat() {
            const chatInput = document.getElementById('chat-input');
            const sendButton = document.getElementById('send-button');
            const chatMessages = document.getElementById('chat-messages');
            const showEmotionalState = document.getElementById('showEmotionalState');
            
            // Send message
            function sendMessage() {
                const message = chatInput.value.trim();
                if (!message) return;
                
                // Add user message to chat
                const userMessageElement = document.createElement('div');
                userMessageElement.className = 'message user-message';
                userMessageElement.textContent = message;
                chatMessages.appendChild(userMessageElement);
                
                // Clear input
                chatInput.value = '';
                
                // Scroll to bottom
                chatMessages.scrollTop = chatMessages.scrollHeight;
                
                // Send to server
                fetch('/api/query', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ query: message }),
                })
                .then(response => response.json())
                .then(data => {
                    // Simulate response (in a real implementation, this would be the actual response)
                    setTimeout(() => {
                        // Add AI message to chat
                        const aiMessageElement = document.createElement('div');
                        aiMessageElement.className = 'message ai-message';
                        
                        // Simulated response text
                        aiMessageElement.textContent = "This is a simulated response. In a real implementation, this would be the model's response.";
                        
                        // Add emotional state if enabled
                        if (showEmotionalState.checked) {
                            const emotionTag = document.createElement('div');
                            emotionTag.className = 'emotion-tag bg-light text-secondary';
                            
                            // Get dominant emotional state from data if available
                            let emotion = 'neutral';
                            let intensity = 50;
                            
                            if (data.hormonal_state && data.hormonal_state.dominant_state) {
                                const emotionalState = data.hormonal_state.dominant_state;
                                emotion = emotionalState[0] || 'neutral';
                                intensity = (emotionalState[1] || 50).toFixed(0);
                            }
                            
                            emotionTag.textContent = `${emotion} (${intensity}%)`;
                            aiMessageElement.appendChild(emotionTag);
                        }
                        
                        chatMessages.appendChild(aiMessageElement);
                        
                        // Scroll to bottom
                        chatMessages.scrollTop = chatMessages.scrollHeight;
                    }, 1000);
                })
                .catch(error => console.error('Error:', error));
            }
            
            // Event listeners
            sendButton.addEventListener('click', sendMessage);
            
            chatInput.addEventListener('keydown', function(event) {
                if (event.key === 'Enter') {
                    event.preventDefault();
                    sendMessage();
                }
            });
        }
        
        // Model management
        function setupModelManagement() {
            const searchButton = document.getElementById('search-button');
            const modelSearch = document.getElementById('model-search');
            const modelFilter = document.getElementById('model-filter');
            
            // Load downloaded models
            function loadDownloadedModels() {
                const tableBody = document.getElementById('downloaded-models-body');
                tableBody.innerHTML = '<tr><td colspan="4" class="text-center">Loading models...</td></tr>';
                
                fetch('/api/models')
                .then(response => response.json())
                .then(models => {
                    if (!models || models.length === 0) {
                        tableBody.innerHTML = '<tr><td colspan="4" class="text-center">No models downloaded</td></tr>';
                        return;
                    }
                    
                    tableBody.innerHTML = '';
                    models.forEach(model => {
                        const row = document.createElement('tr');
                        
                        // Format size
                        let sizeStr = 'Unknown';
                        if (model.disk_size) {
                            const sizeGB = model.disk_size / (1024 * 1024 * 1024);
                            sizeStr = sizeGB < 1 ? 
                                `${(sizeGB * 1000).toFixed(0)}MB` : 
                                `${sizeGB.toFixed(1)}GB`;
                        }
                        
                        row.innerHTML = `
                            <td>${model.name || 'Unknown'}</td>
                            <td>${model.model_type || 'Unknown'}</td>
                            <td>${sizeStr}</td>
                            <td>
                                <button class="btn btn-sm btn-danger delete-model" data-model="${model.id}">Delete</button>
                            </td>
                        `;
                        
                        tableBody.appendChild(row);
                    });
                    
                    // Add event listeners for delete buttons
                    const deleteButtons = document.querySelectorAll('.delete-model');
                    deleteButtons.forEach(button => {
                        button.addEventListener('click', function() {
                            const modelId = this.getAttribute('data-model');
                            if (confirm(`Are you sure you want to delete ${modelId}?`)) {
                                deleteModel(modelId);
                            }
                        });
                    });
                })
                .catch(error => {
                    console.error('Error loading models:', error);
                    tableBody.innerHTML = '<tr><td colspan="4" class="text-center">Error loading models</td></tr>';
                });
            }
            
            // Search for models
            function searchModels() {
                const query = modelSearch.value.trim();
                const filter = modelFilter.value;
                
                const tableBody = document.getElementById('available-models-body');
                tableBody.innerHTML = '<tr><td colspan="5" class="text-center">Searching...</td></tr>';
                
                fetch(`/api/models/search?query=${encodeURIComponent(query)}&model_type=${encodeURIComponent(filter)}`)
                .then(response => response.json())
                .then(models => {
                    if (!models || models.length === 0) {
                        tableBody.innerHTML = '<tr><td colspan="5" class="text-center">No models found</td></tr>';
                        return;
                    }
                    
                    tableBody.innerHTML = '';
                    models.forEach(model => {
                        const row = document.createElement('tr');
                        
                        row.innerHTML = `
                            <td>${model.name || 'Unknown'}</td>
                            <td>${model.author || 'Unknown'}</td>
                            <td>${model.model_type || 'Unknown'}</td>
                            <td>${model.downloads?.toLocaleString() || '0'}</td>
                            <td>
                                <button class="btn btn-sm btn-primary download-model" data-model="${model.id}">Download</button>
                            </td>
                        `;
                        
                        tableBody.appendChild(row);
                    });
                    
                    // Add event listeners for download buttons
                    const downloadButtons = document.querySelectorAll('.download-model');
                    downloadButtons.forEach(button => {
                        button.addEventListener('click', function() {
                            const modelId = this.getAttribute('data-model');
                            downloadModel(modelId);
                        });
                    });
                })
                .catch(error => {
                    console.error('Error searching models:', error);
                    tableBody.innerHTML = '<tr><td colspan="5" class="text-center">Error searching models</td></tr>';
                });
            }
            
            // Download a model
            function downloadModel(modelId) {
                fetch('/api/models/download', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ model_id: modelId }),
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'started' || data.status === 'in_progress') {
                        alert(`Download of ${modelId} started. This may take some time.`);
                    } else if (data.status === 'already_downloaded') {
                        alert(`Model ${modelId} is already downloaded.`);
                    } else {
                        alert(`Error: ${data.message || 'Unknown error'}`);
                    }
                })
                .catch(error => {
                    console.error('Error downloading model:', error);
                    alert('Error starting download. See console for details.');
                });
            }
            
            // Delete a model
            function deleteModel(modelId) {
                fetch(`/api/models/${encodeURIComponent(modelId)}`, {
                    method: 'DELETE',
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'deleted') {
                        alert(`Model ${modelId} deleted successfully.`);
                        loadDownloadedModels(); // Refresh the list
                    } else {
                        alert(`Error: ${data.message || 'Unknown error'}`);
                    }
                })
                .catch(error => {
                    console.error('Error deleting model:', error);
                    alert('Error deleting model. See console for details.');
                });
            }
            
            // Event listeners
            searchButton.addEventListener('click', searchModels);
            
            modelSearch.addEventListener('keydown', function(event) {
                if (event.key === 'Enter') {
                    event.preventDefault();
                    searchModels();
                }
            });
            
            // Load initial data
            loadDownloadedModels();
        }
        
        // Value blocks management
        function setupValueBlocks() {
            const valueBlocksBody = document.getElementById('value-blocks-body');
            const saveValueBlockButton = document.getElementById('saveValueBlock');
            const valueBlockForm = document.getElementById('valueBlockForm');
            const valueBlockWeight = document.getElementById('valueBlockWeight');
            const weightValue = document.getElementById('weightValue');
            
            // Update weight display
            valueBlockWeight.addEventListener('input', function() {
                weightValue.textContent = this.value;
            });
            
            // Load value blocks
            function loadValueBlocks() {
                valueBlocksBody.innerHTML = '<tr><td colspan="4" class="text-center">Loading value blocks...</td></tr>';
                
                fetch('/api/values')
                .then(response => response.json())
                .then(blocks => {
                    if (!blocks || blocks.length === 0) {
                        valueBlocksBody.innerHTML = '<tr><td colspan="4" class="text-center">No value blocks found</td></tr>';
                        return;
                    }
                    
                    valueBlocksBody.innerHTML = '';
                    blocks.forEach(block => {
                        const row = document.createElement('tr');
                        
                        row.innerHTML = `
                            <td>${block.name}</td>
                            <td>${block.node_type}</td>
                            <td>${block.weight.toFixed(2)}</td>
                            <td>
                                <button class="btn btn-sm btn-primary edit-block" data-block="${block.name}">Edit</button>
                            </td>
                        `;
                        
                        valueBlocksBody.appendChild(row);
                    });
                    
                    // Add event listeners for edit buttons
                    const editButtons = document.querySelectorAll('.edit-block');
                    editButtons.forEach(button => {
                        button.addEventListener('click', function() {
                            const blockName = this.getAttribute('data-block');
                            editValueBlock(blockName, blocks);
                        });
                    });
                })
                .catch(error => {
                    console.error('Error loading value blocks:', error);
                    valueBlocksBody.innerHTML = '<tr><td colspan="4" class="text-center">Error loading value blocks</td></tr>';
                });
            }
            
            // Edit value block
            function editValueBlock(blockName, blocks) {
                const block = blocks.find(b => b.name === blockName);
                if (!block) return;
                
                document.getElementById('valueBlockId').value = block.name;
                document.getElementById('valueBlockName').value = block.name;
                document.getElementById('valueBlockName').disabled = true;
                document.getElementById('valueBlockType').value = block.node_type;
                document.getElementById('valueBlockType').disabled = true;
                document.getElementById('valueBlockParent').value = 'values'; // Default
                document.getElementById('valueBlockParent').disabled = true;
                document.getElementById('valueBlockWeight').value = block.weight;
                weightValue.textContent = block.weight;
                document.getElementById('valueBlockContent').value = block.content;
                
                document.getElementById('valueBlockModalTitle').textContent = 'Edit Value Block';
                
                // Show modal
                const modal = new bootstrap.Modal(document.getElementById('valueBlockModal'));
                modal.show();
            }
            
            // Save value block
            saveValueBlockButton.addEventListener('click', function() {
                const blockId = document.getElementById('valueBlockId').value;
                const name = document.getElementById('valueBlockName').value;
                const type = document.getElementById('valueBlockType').value;
                const parent = document.getElementById('valueBlockParent').value;
                const weight = parseFloat(document.getElementById('valueBlockWeight').value);
                const content = document.getElementById('valueBlockContent').value;
                
                if (!name || !content) {
                    alert('Name and content are required');
                    return;
                }
                
                if (blockId) {
                    // Update existing block
                    fetch(`/api/values/${encodeURIComponent(blockId)}`, {
                        method: 'PUT',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            content: content,
                            weight: weight
                        }),
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            alert('Value block updated successfully');
                            loadValueBlocks();
                            
                            // Hide modal
                            const modal = bootstrap.Modal.getInstance(document.getElementById('valueBlockModal'));
                            modal.hide();
                        } else {
                            alert(`Error: ${data.error || 'Unknown error'}`);
                        }
                    })
                    .catch(error => {
                        console.error('Error updating value block:', error);
                        alert('Error updating value block. See console for details.');
                    });
                } else {
                    // Add new block
                    fetch('/api/values', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            name: name,
                            content: content,
                            node_type: type,
                            parent_name: parent,
                            weight: weight
                        }),
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            alert('Value block added successfully');
                            loadValueBlocks();
                            
                            // Hide modal and reset form
                            const modal = bootstrap.Modal.getInstance(document.getElementById('valueBlockModal'));
                            modal.hide();
                            valueBlockForm.reset();
                        } else {
                            alert(`Error: ${data.error || 'Unknown error'}`);
                        }
                    })
                    .catch(error => {
                        console.error('Error adding value block:', error);
                        alert('Error adding value block. See console for details.');
                    });
                }
            });
            
            // Reset form on modal open
            document.getElementById('valueBlockModal').addEventListener('show.bs.modal', function (event) {
                if (!event.relatedTarget) return; // Skip if opened programmatically
                
                document.getElementById('valueBlockId').value = '';
                document.getElementById('valueBlockName').value = '';
                document.getElementById('valueBlockName').disabled = false;
                document.getElementById('valueBlockType').value = 'value_block';
                document.getElementById('valueBlockType').disabled = false;
                document.getElementById('valueBlockParent').value = 'values';
                document.getElementById('valueBlockParent').disabled = false;
                document.getElementById('valueBlockWeight').value = 0.8;
                weightValue.textContent = '0.8';
                document.getElementById('valueBlockContent').value = '';
                document.getElementById('valueBlockModalTitle').textContent = 'Add Value Block';
            });
            
            // Self-regulation button
            document.getElementById('btn-self-regulate').addEventListener('click', function() {
                fetch('/api/self-regulate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({}),
                })
                .then(response => response.json())
                .then(data => {
                    alert('Self-regulation initiated');
                    // Request updated status
                    if (socket && socket.readyState === WebSocket.OPEN) {
                        socket.send(JSON.stringify({ command: 'get_status' }));
                    }
                })
                .catch(error => {
                    console.error('Error during self-regulation:', error);
                    alert('Error during self-regulation. See console for details.');
                });
            });
            
            // Load initial data
            loadValueBlocks();
        }
        
        // Soul details
        function setupSoulDetails() {
            // Request soul details when tab is shown
            document.getElementById('soul-tab').addEventListener('shown.bs.tab', function() {
                if (socket && socket.readyState === WebSocket.OPEN) {
                    socket.send(JSON.stringify({ command: 'get_soul_details' }));
                }
            });
        }
        
        // Initialize on page load
        document.addEventListener('DOMContentLoaded', function() {
            // Connect WebSocket
            connectWebSocket();
            
            // Setup UI components
            setupChat();
            setupModelManagement();
            setupValueBlocks();
            setupSoulDetails();
        });
    </script>
</body>
</html>
"""

def create_static_files(static_path: Path) -> None:
    """Create static files for the web interface.
    
    Args:
        static_path: Path to create files in
    """
    # Ensure directory exists
    static_path.mkdir(parents=True, exist_ok=True)
    
    # Create index.html with the frontend content
    with open(static_path / "index.html", 'w') as f:
        f.write(FRONTEND_HTML)
    
    logger.info(f"Created static files in {static_path}")


async def start_web_server(soul_manager: Any, model_manager: Any, host: str = "0.0.0.0", port: int = 8080) -> WebServer:
    """Start the web server.
    
    Args:
        soul_manager: SoulManager instance
        model_manager: ModelManager instance
        host: Host to bind to
        port: Port to bind to
        
    Returns:
        WebServer instance
    """
    # Create static files directory
    static_path = Path("web_static")
    create_static_files(static_path)
    
    # Create and start server
    server = WebServer(
        soul_manager=soul_manager,
        model_manager=model_manager,
        static_path=static_path,
        host=host,
        port=port
    )
    
    await server.start()
    return server
