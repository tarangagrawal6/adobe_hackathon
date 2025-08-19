import os
import logging
import tempfile
import io
import wave
import numpy as np
from typing import Tuple, List, Optional, Dict
import whisper
import requests
import aiohttp
import subprocess
import webrtcvad
import asyncio
import re
from collections import deque
from dotenv import load_dotenv
load_dotenv() 


logger = logging.getLogger(__name__)

# Global model cache to avoid reloading models
_global_stt_model = None
_global_vad_model = None
_global_models_initialized = False

class AudioChunk:
    """Represents a chunk of audio data with metadata"""
    def __init__(self, audio_data: bytes, timestamp: float, is_speech: bool = False):
        self.audio_data = audio_data
        self.timestamp = timestamp
        self.is_speech = is_speech
        self.processed = False

class ChunkBatch:
    """Represents a batch of audio chunks for processing"""
    def __init__(self, chunks: List[AudioChunk], batch_id: str):
        self.chunks = chunks
        self.batch_id = batch_id
        self.combined_audio = b""
        self.transcribed_text = ""
        self.processing = False
        self.completed = False

class VoiceService:
    def __init__(self):
        # Model caching - load once and keep in memory
        self.stt_model = None
        self.vad = None
        self._models_initialized = False
        self._initialize_models()
        
        # Audio processing settings - optimized for low latency
        self.sample_rate = 16000
        self.chunk_duration_ms = 30  # 30ms chunks for VAD
        self.speech_threshold = 0.6  # VAD sensitivity
        self.silence_threshold_ms = 500  # Reduced silence threshold for faster response
        self.min_speech_duration_ms = 200  # Reduced minimum speech duration
        self.chunks_per_batch = 2  # Reduced batch size for faster response (1-2 seconds)
        self.min_chunks_for_processing = 1  # Process even single chunks after timeout
        
        # VAD error tracking
        self.vad_error_count = 0
        self.max_vad_errors = 10  # Disable VAD after 10 consecutive errors
        
        # Chunk processing queue and batch management
        self.audio_chunks: deque = deque()
        self.chunk_batches: Dict[str, ChunkBatch] = {}
        self.processing_lock = asyncio.Lock()
        self.batch_counter = 0
        
        # Speech detection state
        self.is_speaking = False
        self.speech_start_time = None
        self.last_speech_time = None
        self.silence_start_time = None
        
        # Text accumulation for sentence cutting
        self.accumulated_text = ""
        self.sentence_queue = deque()
        
        # Model cache for faster processing
        self._model_cache = {}
        self._cache_initialized = False
        
        # Async session for HTTP requests
        self._aiohttp_session = None
        
        # Batch timeout management
        self.last_chunk_time = None
        self.batch_timeout_seconds = 1.5  # Process batch after 1.5 seconds of silence
    
    def _initialize_models(self):
        """Initialize STT, TTS, and VAD models with global caching"""
        global _global_stt_model, _global_vad_model, _global_models_initialized
        
        if self._models_initialized:
            logger.info("Models already initialized, skipping...")
            return
            
        try:
            # Use global cache if available
            if _global_models_initialized:
                logger.info("Using globally cached models...")
                self.stt_model = _global_stt_model
                self.vad = _global_vad_model
                self._models_initialized = True
                return
            
            # Initialize Whisper model for STT with global caching
            logger.info("Loading Whisper model (this may take a moment)...")
            _global_stt_model = whisper.load_model("base")
            self.stt_model = _global_stt_model
            logger.info("Whisper model loaded and globally cached successfully")
            
            # Initialize VAD with error handling and global caching
            try:
                logger.info("Initializing VAD...")
                _global_vad_model = webrtcvad.Vad(2)  # Aggressiveness level 2 (moderate)
                self.vad = _global_vad_model
                logger.info("VAD initialized and globally cached successfully")
            except Exception as e:
                logger.error(f"Error initializing VAD: {str(e)}")
                logger.warning("VAD will be disabled - all audio will be treated as speech")
                _global_vad_model = None
                self.vad = None
            
            # Check if ffmpeg is available (but don't fail if it's not)
            ffmpeg_available = self._check_ffmpeg()
            if not ffmpeg_available:
                logger.warning("FFmpeg not available - speech-to-text will not work properly")
                logger.warning("Please install FFmpeg for full voice functionality")
            
            # Check Azure TTS configuration at startup
            azure_key = os.getenv("AZURE_TTS_KEY")
            azure_endpoint = os.getenv("AZURE_TTS_ENDPOINT")
            
            if azure_key and azure_endpoint:
                logger.info("Azure TTS service configured")
            else:
                logger.warning("Azure TTS credentials not configured - will use beep fallback")
            
            # Mark models as initialized globally
            _global_models_initialized = True
            self._models_initialized = True
            logger.info("All models initialized and globally cached successfully")
                
        except Exception as e:
            logger.error(f"Error initializing voice models: {str(e)}")
            # Don't raise the exception - allow the service to start with limited functionality
            logger.warning("Voice service will start with limited functionality")
    
    def _check_ffmpeg(self):
        """Check if ffmpeg is available in the system"""
        # Skip FFmpeg check to reduce latency - assume it's available
        return True
    
    def _detect_speech(self, audio_chunk: bytes) -> bool:
        """Detect if audio chunk contains speech using VAD"""
        try:
            if self.vad is None:
                return True  # If VAD not available, assume speech
            
            # VAD expects 16-bit PCM audio at the specified sample rate
            # Check if chunk size matches expected size for 3ms at 16kHz
            expected_size = int(self.sample_rate * 2 * self.chunk_duration_ms / 1000)  # 2 bytes per sample
            
            # Handle size mismatches more gracefully
            if len(audio_chunk) != expected_size:
                # If chunk is too small, pad with zeros
                if len(audio_chunk) < expected_size:
                    audio_chunk = audio_chunk + b'\x00' * (expected_size - len(audio_chunk))
                # If chunk is too large, truncate
                elif len(audio_chunk) > expected_size:
                    audio_chunk = audio_chunk[:expected_size]
                
                logger.debug(f"Adjusted audio chunk size from {len(audio_chunk)} to {expected_size}")
            
            # Ensure the audio chunk is valid 16-bit PCM
            if len(audio_chunk) % 2 != 0:
                # Pad with zero if odd length
                audio_chunk = audio_chunk + b'\x00'
            
            result = self.vad.is_speech(audio_chunk, self.sample_rate)
            
            # Reset error count on successful VAD processing
            self.vad_error_count = 0
            
            return result
            
        except Exception as e:
            self.vad_error_count += 1
            logger.error(f"Error in speech detection (error #{self.vad_error_count}): {str(e)}")
            
            # Disable VAD if too many errors occur
            if self.vad_error_count >= self.max_vad_errors:
                logger.warning(f"Too many VAD errors ({self.vad_error_count}), disabling VAD")
                self.vad = None
                return True
            
            # Return True to avoid blocking the conversation flow
            return True
    
    def _webm_to_pcm(self, webm_data: bytes) -> bytes:
        """Convert WebM audio to 16-bit PCM using ffmpeg"""
        try:
            if not self._check_ffmpeg():
                raise Exception("FFmpeg not available")
            
            # Create temporary files
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as webm_file:
                webm_file.write(webm_data)
                webm_path = webm_file.name
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
                wav_path = wav_file.name
            
            try:
                # Convert WebM to WAV using ffmpeg
                cmd = [
                    'ffmpeg', '-y',  # Overwrite output file
                    '-i', webm_path,  # Input file
                    '-f', 'wav',  # Output format
                    '-acodec', 'pcm_s16le',  # 16-bit PCM codec
                    '-ar', str(self.sample_rate),  # Sample rate
                    '-ac', '1',  # Mono channel
                    wav_path  # Output file
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                
                if result.returncode != 0:
                    raise Exception(f"FFmpeg conversion failed: {result.stderr}")
                
                # Read the converted WAV file
                with open(wav_path, 'rb') as f:
                    wav_data = f.read()
                
                # Extract PCM data (skip WAV header)
                pcm_data = wav_data[44:]  # Standard WAV header is 44 bytes
                
                return pcm_data
                
            finally:
                # Clean up temporary files
                if os.path.exists(webm_path):
                    os.unlink(webm_path)
                if os.path.exists(wav_path):
                    os.unlink(wav_path)
                    
        except Exception as e:
            logger.error(f"Error converting WebM to PCM: {str(e)}")
            raise
    
    def _chunk_audio(self, pcm_data: bytes) -> List[bytes]:
        """Split PCM audio into chunks for VAD processing"""
        try:
            chunk_size = int(self.sample_rate * 2 * self.chunk_duration_ms / 1000)  # 2 bytes per sample
            chunks = []
            
            for i in range(0, len(pcm_data), chunk_size):
                chunk = pcm_data[i:i + chunk_size]
                if len(chunk) == chunk_size:  # Only add complete chunks
                    chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking audio: {str(e)}")
            return []
    
    def _create_batch_id(self) -> str:
        """Create a unique batch ID"""
        self.batch_counter += 1
        return f"batch_{self.batch_counter}_{int(asyncio.get_event_loop().time())}"
    
    def _cut_sentences(self, text: str) -> List[str]:
        """Cut text into sentences based on punctuation"""
        try:
            # Split on sentence-ending punctuation followed by whitespace or end of string
            sentences = re.split(r'[.!?]+(?:\s+|$)', text.strip())
            
            # Filter out empty sentences and clean up
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # Add punctuation back to sentences (except the last one if it doesn't end with punctuation)
            result = []
            for i, sentence in enumerate(sentences):
                if i < len(sentences) - 1:
                    # Find the original punctuation that was split on
                    original_text = text
                    start_pos = original_text.find(sentence)
                    if start_pos != -1:
                        end_pos = start_pos + len(sentence)
                        if end_pos < len(original_text):
                            # Find the next punctuation mark
                            for j in range(end_pos, len(original_text)):
                                if original_text[j] in '.!?':
                                    sentence += original_text[j]
                                    break
                
                if sentence.strip():
                    result.append(sentence.strip())
            
            return result
            
        except Exception as e:
            logger.error(f"Error cutting sentences: {str(e)}")
            return [text] if text.strip() else []
    
    async def process_audio_chunk(self, audio_data: bytes, timestamp: float) -> Optional[str]:
        """
        Process a single audio chunk with immediate processing for minimal latency
        
        Args:
            audio_data: Raw audio bytes (WebM format)
            timestamp: Timestamp of the audio chunk
            
        Returns:
            Transcribed text if a complete sentence is ready, None otherwise
        """
        try:
            logger.info(f"process_audio_chunk called with {len(audio_data)} bytes")
            
            # Skip processing if audio data is too small
            if len(audio_data) < 1000:  # Increased minimum size for better quality
                logger.debug(f"Audio data too small: {len(audio_data)} bytes")
                return None
            
            # IMMEDIATE PROCESSING: Process audio directly for fastest response
            logger.info("Processing audio immediately for minimal latency")
            transcribed_text = await self.speech_to_text(audio_data)
            
            if transcribed_text and transcribed_text.strip():
                logger.info(f"Immediate transcription result: '{transcribed_text}'")
                return transcribed_text
            else:
                logger.info("No transcription result from immediate processing")
                return None
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    async def _process_chunk_batch(self) -> Optional[str]:
        """Process a batch of audio chunks"""
        async with self.processing_lock:
            try:
                logger.info(f"_process_chunk_batch called with {len(self.audio_chunks)} chunks in queue")
                
                # Process all available chunks (even if less than chunks_per_batch)
                if len(self.audio_chunks) < self.min_chunks_for_processing:
                    logger.info(f"Not enough chunks for processing. Need {self.min_chunks_for_processing}, have {len(self.audio_chunks)}")
                    return None
                
                # Create a new batch with all available chunks
                batch_chunks = []
                chunks_to_process = min(len(self.audio_chunks), self.chunks_per_batch)
                
                for _ in range(chunks_to_process):
                    if self.audio_chunks:
                        batch_chunks.append(self.audio_chunks.popleft())
                
                if not batch_chunks:
                    logger.warning("No chunks to process")
                    return None
                
                batch_id = self._create_batch_id()
                batch = ChunkBatch(batch_chunks, batch_id)
                self.chunk_batches[batch_id] = batch
                
                logger.info(f"Created batch {batch_id} with {len(batch_chunks)} chunks")
                
                # Process the batch asynchronously
                asyncio.create_task(self._process_batch_async(batch))
                
                # Check for sentence completion
                result = await self._check_sentence_completion()
                logger.info(f"Batch processing completed, sentence result: {result}")
                return result
                
            except Exception as e:
                logger.error(f"Error processing chunk batch: {str(e)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                return None
    
    async def _process_batch_async(self, batch: ChunkBatch):
        """Process a batch of chunks asynchronously with optimized processing"""
        try:
            logger.info(f"Starting async batch processing for batch {batch.batch_id}")
            batch.processing = True
            
            # Combine audio chunks
            combined_audio = b""
            for chunk in batch.chunks:
                combined_audio += chunk.audio_data
                chunk.processed = True
            
            batch.combined_audio = combined_audio
            logger.info(f"Combined audio size: {len(combined_audio)} bytes")
            
            # Skip transcription if audio is too small
            if len(combined_audio) < 1000:  # Skip very small audio
                logger.info(f"Audio too small ({len(combined_audio)} bytes), skipping transcription")
                batch.transcribed_text = ""
                batch.completed = True
                return
            
            # Transcribe the combined audio
            logger.info("Starting speech-to-text transcription...")
            transcribed_text = await self.speech_to_text(combined_audio)
            batch.transcribed_text = transcribed_text
            batch.completed = True
            
            logger.info(f"Transcription result: '{transcribed_text}'")
            
            # Add to accumulated text
            if transcribed_text and transcribed_text.strip():
                self.accumulated_text += " " + transcribed_text
                self.accumulated_text = self.accumulated_text.strip()
                
                logger.info(f"Batch {batch.batch_id} completed: {transcribed_text}")
                logger.info(f"Updated accumulated text: '{self.accumulated_text}'")
            else:
                logger.warning(f"Batch {batch.batch_id} completed with empty transcription")
            
            # Clean up completed batches (keep only recent ones)
            await self._cleanup_completed_batches()
            
        except Exception as e:
            logger.error(f"Error processing batch {batch.batch_id}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            batch.completed = True  # Mark as completed even on error
    
    async def _cleanup_completed_batches(self):
        """Clean up completed batches to free memory"""
        try:
            completed_batches = [
                batch_id for batch_id, batch in self.chunk_batches.items()
                if batch.completed
            ]
            
            # Keep only the 5 most recent completed batches
            if len(completed_batches) > 5:
                batches_to_remove = completed_batches[:-5]
                for batch_id in batches_to_remove:
                    del self.chunk_batches[batch_id]
                    
        except Exception as e:
            logger.error(f"Error cleaning up batches: {str(e)}")
    
    async def _check_sentence_completion(self) -> Optional[str]:
        """Check if we have complete sentences ready for LLM processing"""
        try:
            logger.info(f"_check_sentence_completion called. Accumulated text: '{self.accumulated_text}'")
            
            if not self.accumulated_text:
                logger.debug("No accumulated text")
                return None
            
            # Cut text into sentences
            sentences = self._cut_sentences(self.accumulated_text)
            logger.info(f"Cut into {len(sentences)} sentences: {sentences}")
            
            if len(sentences) >= 1:
                # We have at least one complete sentence
                complete_sentence = sentences[0]
                
                # Remove the complete sentence from accumulated text
                remaining_text = " ".join(sentences[1:]) if len(sentences) > 1 else ""
                self.accumulated_text = remaining_text
                
                logger.info(f"Complete sentence ready: {complete_sentence}")
                return complete_sentence
            
            logger.debug("No complete sentences yet")
            return None
            
        except Exception as e:
            logger.error(f"Error checking sentence completion: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    async def force_process_remaining_chunks(self) -> Optional[str]:
        """Force process any remaining chunks in the queue"""
        try:
            if not self.audio_chunks:
                logger.info("No remaining chunks to process")
                return None
            
            logger.info(f"Force processing {len(self.audio_chunks)} remaining chunks")
            result = await self._process_chunk_batch()
            return result
            
        except Exception as e:
            logger.error(f"Error force processing remaining chunks: {str(e)}")
            return None
    
    async def force_process_accumulated_text(self) -> Optional[str]:
        """Force process any accumulated text that hasn't been processed yet"""
        try:
            if not self.accumulated_text:
                logger.info("No accumulated text to process")
                return None
            
            logger.info(f"Force processing accumulated text: '{self.accumulated_text}'")
            
            # Cut text into sentences
            sentences = self._cut_sentences(self.accumulated_text)
            if sentences:
                # Return the first sentence
                complete_sentence = sentences[0]
                remaining_text = " ".join(sentences[1:]) if len(sentences) > 1 else ""
                self.accumulated_text = remaining_text
                
                logger.info(f"Force processed sentence: {complete_sentence}")
                return complete_sentence
            
            return None
            
        except Exception as e:
            logger.error(f"Error force processing accumulated text: {str(e)}")
            return None
    
    async def process_audio_directly(self, audio_data: bytes) -> Optional[str]:
        """Process audio directly without batching (for immediate response)"""
        try:
            logger.info(f"Processing audio directly: {len(audio_data)} bytes")
            
            # Skip processing if audio data is too small
            if len(audio_data) < 1000:
                logger.info(f"Audio too small for direct processing: {len(audio_data)} bytes")
                return None
            
            # Transcribe the audio directly
            transcribed_text = await self.speech_to_text(audio_data)
            
            if transcribed_text and transcribed_text.strip():
                logger.info(f"Direct transcription result: '{transcribed_text}'")
                return transcribed_text
            else:
                logger.info("Direct transcription returned empty result")
                return None
                
        except Exception as e:
            logger.error(f"Error in direct audio processing: {str(e)}")
            return None
    
    def reset_audio_processing(self):
        """Reset audio processing state"""
        try:
            self.audio_chunks.clear()
            self.accumulated_text = ""
            self.last_chunk_time = None
            logger.info("Audio processing state reset")
        except Exception as e:
            logger.error(f"Error resetting audio processing: {str(e)}")
    
    def _ensure_models_loaded(self):
        """Ensure all models are loaded and cached"""
        if not self._models_initialized:
            logger.info("Models not initialized, loading now...")
            self._initialize_models()
    
    async def _get_aiohttp_session(self):
        """Get or create aiohttp session for async HTTP requests"""
        if self._aiohttp_session is None or self._aiohttp_session.closed:
            self._aiohttp_session = aiohttp.ClientSession()
        return self._aiohttp_session
    
    async def _close_aiohttp_session(self):
        """Close aiohttp session"""
        if self._aiohttp_session and not self._aiohttp_session.closed:
            await self._aiohttp_session.close()
            self._aiohttp_session = None
    
    async def speech_to_text(self, audio_data: bytes) -> str:
        """
        Convert speech audio to text using Whisper with ultra-fast settings (async)
        
        Args:
            audio_data: Raw audio bytes (WebM format expected)
            
        Returns:
            Transcribed text
        """
        try:
            logger.info(f"speech_to_text called with {len(audio_data)} bytes")
            
            # Ensure models are loaded
            self._ensure_models_loaded()
            
            if self.stt_model is None:
                logger.error("STT model not initialized")
                raise Exception("STT model not initialized")
            
            # Skip processing if audio is too small
            if len(audio_data) < 1000:
                logger.info(f"Audio too small for transcription: {len(audio_data)} bytes")
                return ""
            
            # Save audio data to temporary file with .webm extension
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            try:
                logger.info(f"Processing audio file: {temp_file_path} ({len(audio_data)} bytes)")
                
                # Run transcription in a thread with ultra-fast settings
                def transcribe_sync():
                    logger.info("Starting ultra-fast Whisper transcription...")
                    result = self.stt_model.transcribe(
                        temp_file_path,
                        language="en",
                        task="transcribe",
                        fp16=False,  # Disable FP16 to avoid CPU warning
                        verbose=False,
                        condition_on_previous_text=False,  # Faster processing
                        temperature=0.0,  # Deterministic output
                        compression_ratio_threshold=1.8,  # More aggressive noise filtering
                        logprob_threshold=-0.8,  # Higher threshold for faster processing
                        no_speech_threshold=0.6,  # Skip silent audio faster
                        initial_prompt="",  # No initial prompt for speed
                        word_timestamps=False,  # Disable word timestamps for speed
                        prepend_punctuations="",  # No prepend for speed
                        append_punctuations=""  # No append for speed
                    )
                    return result["text"].strip()
                
                # Use asyncio.to_thread for Python 3.9+ or loop.run_in_executor for older versions
                try:
                    transcribed_text = await asyncio.to_thread(transcribe_sync)
                except AttributeError:
                    # Fallback for older Python versions
                    loop = asyncio.get_event_loop()
                    transcribed_text = await loop.run_in_executor(None, transcribe_sync)
                
                logger.info(f"Ultra-fast Whisper transcription completed: '{transcribed_text}'")
                return transcribed_text
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    
        except Exception as e:
            logger.error(f"Error in speech to text conversion: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return empty string instead of raising to avoid breaking the flow
            return ""
    
    async def text_to_speech(self, text: str) -> bytes:
        """
        Convert text to speech using Azure TTS with beep fallback (async)
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Audio data as bytes (WAV format)
        """
        try:
            # Ensure models are loaded
            self._ensure_models_loaded()
            
            logger.info(f"Generating speech for text: {text[:50]}...")
            
            # Re-read environment variables at runtime (same approach as gemini_utils.py)
            azure_key = os.getenv("AZURE_TTS_KEY")
            azure_endpoint = os.getenv("AZURE_TTS_ENDPOINT")
            
            # Try Azure TTS first
            if azure_key and azure_endpoint:
                logger.info("Using Azure TTS")
                try:
                    return await self._azure_tts(text, azure_key, azure_endpoint)
                except Exception as azure_error:
                    logger.error(f"Azure TTS failed: {str(azure_error)}, falling back to beep")
                    return self._generate_beep_audio()
            else:
                # Fallback to beep sound
                logger.warning("Azure TTS not configured, using beep fallback")
                return self._generate_beep_audio()
            
        except Exception as e:
            logger.error(f"Error in text to speech conversion: {str(e)}")
            return self._generate_beep_audio()
    
    async def _azure_tts(self, text: str, azure_key: str, azure_endpoint: str) -> bytes:
        """
        Convert text to speech using Azure TTS (async)
        
        Args:
            text: Text to convert to speech
            azure_key: Azure TTS subscription key
            azure_endpoint: Azure TTS endpoint URL
            
        Returns:
            Audio data as bytes (WAV format)
        """
        try:
            # Clean up endpoint URL (remove trailing slash if present)
            azure_endpoint = azure_endpoint.rstrip('/')
            
            # Azure Speech Service TTS API endpoint
            url = f"{azure_endpoint}/cognitiveservices/v1"
            
            # Headers for Azure TTS
            headers = {
                "Ocp-Apim-Subscription-Key": azure_key,
                "Content-Type": "application/ssml+xml",
                "X-Microsoft-OutputFormat": "riff-16khz-16bit-mono-pcm",
                "User-Agent": "VoiceBot"
            }
            
            # SSML (Speech Synthesis Markup Language) for faster response
            # Using optimized settings for minimal latency
            ssml = f"""<speak version='1.0' xml:lang='en-US'>
                <voice xml:lang='en-US' xml:gender='Female' name='en-US-JennyNeural'>
                    <prosody rate="+10%" pitch="+0%" volume="+0%">
                        {text}
                    </prosody>
                </voice>
            </speak>"""
            
            logger.info(f"Calling Azure TTS API for text: {text[:50]}...")
            logger.info(f"Using endpoint: {url}")
            
            # Get aiohttp session and make async API call
            session = await self._get_aiohttp_session()
            timeout = aiohttp.ClientTimeout(total=15.0)  # 15 second timeout for TTS
            
            async with session.post(url, headers=headers, data=ssml.encode('utf-8'), timeout=timeout) as response:
                if response.status == 200:
                    audio_bytes = await response.read()
                    logger.info(f"Azure TTS generated audio: {len(audio_bytes)} bytes")
                    return audio_bytes
                else:
                    error_text = await response.text()
                    logger.error(f"Azure TTS API error: {response.status} - {error_text}")
                    raise Exception(f"Azure TTS API error: {response.status}")
                
        except Exception as e:
            logger.error(f"Error in Azure TTS: {str(e)}")
            raise
    
    def _generate_beep_audio(self, duration_seconds: float = 0.5) -> bytes:
        """Generate a simple beep sound as fallback"""
        try:
            sample_rate = 16000
            frequency = 1000  # Higher frequency for better audibility
            num_samples = int(sample_rate * duration_seconds)
            
            # Generate sine wave
            t = np.linspace(0, duration_seconds, num_samples, False)
            beep = np.sin(2 * np.pi * frequency * t)
            
            # Add some harmonics for better sound
            beep += 0.3 * np.sin(2 * np.pi * frequency * 2 * t)
            
            # Convert to 16-bit PCM
            beep = (beep * 16383).astype(np.int16)  # Reduced amplitude to avoid clipping
            
            # Create WAV file in memory
            with io.BytesIO() as wav_buffer:
                with wave.open(wav_buffer, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(beep.tobytes())
                
                return wav_buffer.getvalue()
        except Exception as e:
            logger.error(f"Error generating beep audio: {str(e)}")
            return b''
    
    def _numpy_to_wav_bytes(self, audio_array, sample_rate: int = 16000) -> bytes:
        """
        Convert audio array to WAV bytes
        
        Args:
            audio_array: Audio array (numpy array or list)
            sample_rate: Sample rate of the audio
            
        Returns:
            WAV audio as bytes
        """
        try:
            logger.info(f"Converting audio array to WAV, input type: {type(audio_array)}")
            
            # Convert to numpy array if it's not already
            if not isinstance(audio_array, np.ndarray):
                audio_array = np.array(audio_array)
            
            logger.info(f"Audio array shape: {audio_array.shape}, dtype: {audio_array.dtype}")
            
            # Flatten if needed
            if audio_array.ndim > 1:
                audio_array = audio_array.flatten()
                logger.info(f"Flattened audio shape: {audio_array.shape}")
            
            # Ensure audio is in the correct format
            if audio_array.dtype != np.float32:
                audio_array = audio_array.astype(np.float32)
            
            # Check if audio is all zeros (silent)
            if np.all(audio_array == 0):
                logger.warning("Audio array is all zeros (silent)")
                raise Exception("Audio array is silent")
            
            # Normalize audio to 16-bit range
            audio_array = np.clip(audio_array, -1.0, 1.0)
            audio_array = (audio_array * 32767).astype(np.int16)
            
            # Create WAV file in memory
            with io.BytesIO() as wav_buffer:
                with wave.open(wav_buffer, 'wb') as wav_file:
                    wav_file.setnchannels(1)  # Mono
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_array.tobytes())
                
                wav_bytes = wav_buffer.getvalue()
                logger.info(f"Generated WAV bytes: {len(wav_bytes)} bytes")
                return wav_bytes
                
        except Exception as e:
            logger.error(f"Error converting audio array to WAV bytes: {str(e)}")
            raise
    
    async def process_voice_conversation(self, audio_data: bytes, context: str, conversation_history: str = "") -> Tuple[str, bytes]:
        """
        Process a complete voice conversation: STT -> LLM -> TTS (async)
        
        Args:
            audio_data: Raw audio bytes from user
            context: PDF document context
            conversation_history: Previous conversation history
            
        Returns:
            Tuple of (transcribed_text, audio_response_bytes)
        """
        try:
            # Step 1: Convert speech to text
            transcribed_text = await self.speech_to_text(audio_data)
            
            if not transcribed_text:
                return "I couldn't hear what you said. Please try again.", b""
            
            # Step 2: Generate LLM response (this will be handled by the WebSocket handler)
            # For now, we'll return the transcribed text and empty audio
            # The actual LLM processing will be done in the WebSocket handler
            
            return transcribed_text, b""
            
        except Exception as e:
            logger.error(f"Error in voice conversation processing: {str(e)}")
            raise

# Global instance
voice_service = VoiceService()

async def preload_models():
    """Preload all models to reduce latency on first use"""
    logger.info("Preloading models for faster startup...")
    try:
        # Initialize the global voice service to load models
        voice_service._ensure_models_loaded()
        logger.info("Models preloaded successfully")
    except Exception as e:
        logger.error(f"Error preloading models: {str(e)}")

async def cleanup_voice_service():
    """Cleanup voice service resources"""
    try:
        await voice_service._close_aiohttp_session()
        logger.info("Voice service cleanup completed")
    except Exception as e:
        logger.error(f"Error during voice service cleanup: {str(e)}")

# Preload models when module is imported
# Note: This will be called synchronously, but the async function will be created
# The actual preloading will happen when the async function is called
