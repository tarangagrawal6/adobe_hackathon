import json
import logging
import base64
import asyncio
from typing import Dict, Any
from fastapi import WebSocket, WebSocketDisconnect
from .voice_service import voice_service, preload_models, cleanup_voice_service

# Ensure models are preloaded for faster response
# Note: preload_models is now async, so we'll call it when needed
from .gemini_utils import call_gemini_api, call_gemini_api_async
from .web_search_service import WebSearchService

logger = logging.getLogger(__name__)

class VoiceBotWebSocket:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.conversation_histories: Dict[str, str] = {}
        self.document_contexts: Dict[str, str] = {}
        self.call_states: Dict[str, bool] = {}  # Track if call is active for each client
        self.continuous_processing_tasks: Dict[str, asyncio.Task] = {}  # Track continuous processing tasks
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Handle WebSocket connection"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.conversation_histories[client_id] = ""
        self.call_states[client_id] = False
        logger.info(f"Client {client_id} connected to voice bot")
        
        # Preload models asynchronously if this is the first connection
        if not self.active_connections:
            try:
                await preload_models()
            except Exception as e:
                logger.error(f"Error preloading models: {str(e)}")
        
        # Send connection confirmation
        try:
            await websocket.send_text(json.dumps({
                "type": "connection_status",
                "status": "connected",
                "message": "Voice bot connected successfully"
            }))
        except Exception as e:
            logger.error(f"Error sending connection confirmation: {str(e)}")
    
    async def disconnect(self, client_id: str):
        """Handle WebSocket disconnection"""
        # Cancel continuous processing task if running
        if client_id in self.continuous_processing_tasks:
            task = self.continuous_processing_tasks[client_id]
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            del self.continuous_processing_tasks[client_id]
        
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.conversation_histories:
            del self.conversation_histories[client_id]
        if client_id in self.document_contexts:
            del self.document_contexts[client_id]
        if client_id in self.call_states:
            del self.call_states[client_id]
        logger.info(f"Client {client_id} disconnected from voice bot")
        
        # Cleanup voice service if no more active connections
        if not self.active_connections:
            try:
                await cleanup_voice_service()
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}")
    
    async def set_document_context(self, client_id: str, context: str):
        """Set the document context for a client"""
        self.document_contexts[client_id] = context
        logger.info(f"Document context set for client {client_id}")
    
    async def handle_voice_message(self, websocket: WebSocket, client_id: str, message: Dict[str, Any]):
        """Handle incoming voice message"""
        try:
            logger.info(f"Received message from client {client_id}: {message.get('type', 'unknown')}")
            message_type = message.get("type")
            
            if message_type == "voice_audio":
                # Handle continuous voice audio data
                logger.info("Processing voice_audio message")
                await self._process_continuous_voice_audio(websocket, client_id, message)
            elif message_type == "text_message":
                # Handle text message (fallback)
                logger.info("Processing text_message")
                await self._process_text_message(websocket, client_id, message)
            elif message_type == "set_context":
                # Set document context
                logger.info("Processing set_context message")
                context = message.get("context", "")
                await self.set_document_context(client_id, context)
                try:
                    await websocket.send_text(json.dumps({
                        "type": "context_set",
                        "status": "success",
                        "message": "Document context set successfully"
                    }))
                except Exception as e:
                    logger.error(f"Error sending context confirmation: {str(e)}")
            elif message_type == "start_call":
                # Start call and send greeting
                logger.info("Processing start_call message")
                await self._start_call(websocket, client_id)
            elif message_type == "end_call":
                # End call
                logger.info("Processing end_call message")
                await self._end_call(websocket, client_id)
            elif message_type == "ping":
                # Handle ping to keep connection alive
                logger.debug("Received ping from client")
                try:
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": message.get("timestamp", 0)
                    }))
                except Exception as e:
                    logger.error(f"Error sending pong: {str(e)}")
            else:
                logger.warning(f"Unknown message type: {message_type}")
                try:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Unknown message type: {message_type}"
                    }))
                except Exception as e:
                    logger.error(f"Error sending error message: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error handling voice message: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            try:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Error processing message: {str(e)}"
                }))
            except Exception as send_error:
                logger.error(f"Error sending error response: {str(send_error)}")
    
    async def _start_call(self, websocket: WebSocket, client_id: str):
        """Start call and send initial greeting"""
        try:
            self.call_states[client_id] = True
            
            # Send call started confirmation
            await websocket.send_text(json.dumps({
                "type": "call_status",
                "status": "started",
                "message": "Call started successfully"
            }))
            
            # Generate and send initial greeting
            context = self.document_contexts.get(client_id, "")
            greeting = await self._generate_greeting(context)
            
            # Convert greeting to speech
            audio_response = await voice_service.text_to_speech(greeting)
            audio_response_base64 = base64.b64encode(audio_response).decode('utf-8')
            
            # Send greeting response
            await websocket.send_text(json.dumps({
                "type": "voice_response",
                "text": greeting,
                "audio_data": audio_response_base64,
                "is_greeting": True
            }))
            
            logger.info(f"Call started for client {client_id} with greeting")
            
        except Exception as e:
            logger.error(f"Error starting call: {str(e)}")
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Error starting call: {str(e)}"
            }))
    
    async def _end_call(self, websocket: WebSocket, client_id: str):
        """End call"""
        try:
            self.call_states[client_id] = False
            
            # Cancel continuous processing task if running
            if client_id in self.continuous_processing_tasks:
                task = self.continuous_processing_tasks[client_id]
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                del self.continuous_processing_tasks[client_id]
            
            # Force process any remaining audio chunks
            try:
                remaining_text = await voice_service.force_process_remaining_chunks()
                if remaining_text:
                    logger.info(f"Processed remaining text: {remaining_text}")
                    # Generate response for remaining text
                    await self._generate_and_send_response(websocket, client_id, remaining_text)
            except Exception as e:
                logger.error(f"Error processing remaining chunks: {str(e)}")
            
            # Force process any accumulated text
            try:
                accumulated_text = await voice_service.force_process_accumulated_text()
                if accumulated_text:
                    logger.info(f"Processed accumulated text: {accumulated_text}")
                    # Generate response for accumulated text
                    await self._generate_and_send_response(websocket, client_id, accumulated_text)
            except Exception as e:
                logger.error(f"Error processing accumulated text: {str(e)}")
            
            # Reset audio processing state
            voice_service.reset_audio_processing()
            
            # Send call ended confirmation
            await websocket.send_text(json.dumps({
                "type": "call_status",
                "status": "ended",
                "message": "Call ended"
            }))
            
            logger.info(f"Call ended for client {client_id}")
            
        except Exception as e:
            logger.error(f"Error ending call: {str(e)}")
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Error ending call: {str(e)}"
            }))
    
    async def _generate_greeting(self, context: str) -> str:
        """Generate an appropriate greeting based on document context"""
        try:
            if context:
                # Ultra-fast contextual greeting
                greeting_prompt = f"""Generate a 1-sentence greeting for discussing this document: {context[:200]}..."""
                
                try:
                    greeting = await asyncio.wait_for(
                        call_gemini_api_async(greeting_prompt), 
                        timeout=8.0  # Increased timeout to 8 seconds
                    )
                    return greeting.strip()
                except (Exception, asyncio.TimeoutError) as e:
                    logger.error(f"Error calling Gemini API for greeting: {str(e)}")
                    # Fallback to contextual greeting
                    return f"Hello! I'm ready to help you with this document. What would you like to know?"
            else:
                # Default greeting
                return "Hello! I'm your voice assistant. How can I help you?"
                
        except Exception as e:
            logger.error(f"Error generating greeting: {str(e)}")
            return "Hello! I'm your voice assistant. How can I help you today?"
    
    async def _process_continuous_voice_audio(self, websocket: WebSocket, client_id: str, message: Dict[str, Any]):
        """Process continuous voice audio data with optimized processing for low latency"""
        try:
            logger.info(f"Received voice audio message from client {client_id}")
            
            # Check if call is active
            if not self.call_states.get(client_id, False):
                logger.warning(f"Call not active for client {client_id}")
                return  # Skip processing if call is not active
            
            # Extract audio data and timestamp
            audio_base64 = message.get("audio_data")
            timestamp = message.get("timestamp", asyncio.get_event_loop().time())
            
            if not audio_base64:
                logger.warning("No audio data provided")
                return  # Skip if no audio data
            
            logger.info(f"Processing audio chunk: {len(audio_base64)} chars base64, timestamp: {timestamp}")
            
            # Decode base64 audio data
            try:
                audio_data = base64.b64decode(audio_base64)
                logger.info(f"Decoded audio data: {len(audio_data)} bytes")
            except Exception as e:
                logger.error(f"Error decoding base64 audio: {str(e)}")
                return  # Skip processing on decode error
            
            # Process audio chunk with optimized system
            try:
                logger.info("Calling voice_service.process_audio_chunk...")
                transcribed_text = await voice_service.process_audio_chunk(audio_data, timestamp)
                logger.info(f"process_audio_chunk returned: {transcribed_text}")
                
                if transcribed_text:
                    logger.info(f"Complete sentence transcribed: {transcribed_text}")
                    
                    # Send transcribed text back to client with connection check
                    try:
                        await websocket.send_text(json.dumps({
                            "type": "transcription",
                            "text": transcribed_text
                        }))
                    except Exception as send_error:
                        logger.error(f"Error sending transcription: {str(send_error)}")
                        return  # Stop processing if connection is lost
                    
                    # Generate and send LLM response for the complete sentence
                    await self._generate_and_send_response(websocket, client_id, transcribed_text)
                else:
                    logger.debug("No transcribed text returned, continuing...")
                    
            except Exception as e:
                logger.error(f"Error processing audio chunk: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                
                # Fallback to direct processing if batching fails
                try:
                    logger.info("Attempting direct audio processing as fallback...")
                    direct_text = await voice_service.process_audio_directly(audio_data)
                    if direct_text:
                        logger.info(f"Direct processing succeeded: {direct_text}")
                        
                        # Send transcribed text back to client
                        try:
                            await websocket.send_text(json.dumps({
                                "type": "transcription",
                                "text": direct_text
                            }))
                        except Exception as send_error:
                            logger.error(f"Error sending direct transcription: {str(send_error)}")
                            return
                        
                        # Generate and send LLM response
                        await self._generate_and_send_response(websocket, client_id, direct_text)
                except Exception as fallback_error:
                    logger.error(f"Direct processing fallback also failed: {str(fallback_error)}")
                    # Don't send error to client for every chunk - just log it
                
        except Exception as e:
            logger.error(f"Error processing continuous voice audio: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Don't send error to client to avoid spam
    
    async def _generate_and_send_response(self, websocket: WebSocket, client_id: str, transcribed_text: str):
        """Generate LLM response and send it back to client with ultra-fast processing"""
        try:
            # Check if WebSocket is still connected
            if websocket.client_state.value == 3:  # WebSocketState.DISCONNECTED
                logger.warning(f"WebSocket disconnected for client {client_id}, skipping response")
                return
            
            # Get document context and conversation history
            context = self.document_contexts.get(client_id, "")
            conversation_history = self.conversation_histories.get(client_id, "")
            
            # Generate LLM response with timeout
            try:
                llm_response = await asyncio.wait_for(
                    self._generate_llm_response(transcribed_text, context, conversation_history),
                    timeout=1500.0  # Increased timeout to 15 seconds for LLM
                )
            except asyncio.TimeoutError:
                logger.warning("LLM response generation timed out, using fallback")
                llm_response = "I understand. What else can I help you with?"
            except Exception as e:
                logger.error(f"Error generating LLM response: {str(e)}")
                llm_response = "I'm having trouble processing your request right now. Please try again in a moment."
            
            # Update conversation history
            self.conversation_histories[client_id] = f"{conversation_history}\nUser: {transcribed_text}\nAssistant: {llm_response}"
            
            # Convert response to speech with timeout
            try:
                audio_response = await asyncio.wait_for(
                    voice_service.text_to_speech(llm_response),
                    timeout=10.0  # Increased timeout to 10 seconds for TTS
                )
                audio_response_base64 = base64.b64encode(audio_response).decode('utf-8')
            except asyncio.TimeoutError:
                logger.warning("TTS generation timed out, sending text only")
                audio_response_base64 = ""
            except Exception as e:
                logger.error(f"Error converting response to speech: {str(e)}")
                audio_response_base64 = ""
            
            # Send response back to client
            try:
                await websocket.send_text(json.dumps({
                    "type": "voice_response",
                    "text": llm_response,
                    "audio_data": audio_response_base64
                }))
                logger.info(f"Successfully sent response to client {client_id}")
            except Exception as send_error:
                logger.error(f"Error sending response to client {client_id}: {str(send_error)}")
                # Don't raise the exception to avoid breaking the flow
            
        except Exception as e:
            logger.error(f"Error generating and sending response: {str(e)}")
            try:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Error generating response: {str(e)}"
                }))
            except Exception as send_error:
                logger.error(f"Error sending error message: {str(send_error)}")
    
    async def _process_text_message(self, websocket: WebSocket, client_id: str, message: Dict[str, Any]):
        """Process text message (fallback)"""
        try:
            # Check if call is active
            if not self.call_states.get(client_id, False):
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Call is not active. Please start a call first."
                }))
                return
            
            text = message.get("text", "")
            if not text:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "No text provided"
                }))
                return
            
            # Generate and send response
            await self._generate_and_send_response(websocket, client_id, text)
            
        except Exception as e:
            logger.error(f"Error processing text message: {str(e)}")
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": f"Error processing text: {str(e)}"
            }))
    
    async def _generate_llm_response(self, user_message: str, context: str, conversation_history: str) -> str:
        """Generate LLM response using Gemini with enhanced research capabilities"""
        try:
            # Enhanced system prompt for voice assistant with research capabilities
            system_prompt = """You are a helpful voice assistant with access to both document content and web research. 
            Respond in 1-2 short sentences. Be direct, informative, and conversational. 
            When web research is available, briefly mention it naturally in your response."""
            
            enhanced_context = context
            research_links = []
            
            # Perform web research for voice interactions (limited to avoid delays)
            try:
                async with WebSearchService() as web_search:
                    research_data = await web_search.get_research_context(
                        query=user_message,
                        document_context=context
                    )
                    
                    enhanced_context = research_data['enhanced_context']
                    research_links = research_data['research_links']
                    
                    # For voice, we'll mention research availability but keep response concise
                    if research_links:
                        enhanced_context += f"\n\n[Web research found: {len(research_links)} sources]"
                        
            except Exception as search_error:
                logger.warning(f"Web search failed for voice interaction: {str(search_error)}")
                # Continue with document context only if search fails

            # Minimal prompt structure for fastest processing
            full_prompt = f"""Context: {enhanced_context[:200]}...

User: {user_message}

Quick response:"""

            # Generate response using Gemini with timeout
            try:
                response_text = await asyncio.wait_for(
                    call_gemini_api_async(full_prompt), 
                    timeout=1200.0  # Increased timeout to 12 seconds
                )
                return response_text.strip()
            except asyncio.TimeoutError:
                logger.warning("LLM response timed out, using fallback")
                return "I understand. What else can I help you with?"
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            return "Got it! What else would you like to know?"

# Global instance
voice_bot_websocket = VoiceBotWebSocket()
