import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Button } from './ui/button';
import { Mic, MicOff, Loader2, X, Phone, PhoneOff } from 'lucide-react';

interface VoiceBotProps {
  documentContext: string;
  onClose: () => void;
}

interface WebSocketMessage {
  type: string;
  status?: string;
  message?: string;
  text?: string;
  audio_data?: string;
  is_greeting?: boolean;
}

const VoiceBot: React.FC<VoiceBotProps> = ({ documentContext, onClose }) => {
  const [isConnected, setIsConnected] = useState(false);
  const [isCallActive, setIsCallActive] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [transcribedText, setTranscribedText] = useState('');
  const [lastResponse, setLastResponse] = useState('');
  const [callDuration, setCallDuration] = useState(0);
  const [clientId] = useState(() => `voice-bot-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`);
  
  const websocketRef = useRef<WebSocket | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const callTimerRef = useRef<NodeJS.Timeout | null>(null);
  // const audioContextRef = useRef<AudioContext | null>(null);

  // Start call timer
  const startCallTimer = useCallback(() => {
    setCallDuration(0);
    callTimerRef.current = setInterval(() => {
      setCallDuration(prev => prev + 1);
    }, 1000);
  }, []);

  // Stop call timer
  const stopCallTimer = useCallback(() => {
    if (callTimerRef.current) {
      clearInterval(callTimerRef.current);
      callTimerRef.current = null;
    }
  }, []);

  // Initialize WebSocket connection
  const connectWebSocket = useCallback(() => {
    try {
      // Fix the WebSocket URL - use the current host for WebSocket
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const host = window.location.host;
      const wsUrl = `${protocol}//${host}/api/v1/ws/voice-bot/${clientId}`;
      console.log('Connecting to WebSocket:', wsUrl);
      const ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        
        // Set document context
        ws.send(JSON.stringify({
          type: 'set_context',
          context: documentContext
        }));
        
      };
      
      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          handleWebSocketMessage(message);
        } catch (error) {
          console.error('Error parsing WebSocket message:', error);
        }
      };
      
      ws.onclose = () => {
        console.log('WebSocket disconnected');
        setIsConnected(false);
        setIsCallActive(false);
        setIsRecording(false);
        setIsProcessing(false);
        stopCallTimer();
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setIsConnected(false);
      };
      
      websocketRef.current = ws;
    } catch (error) {
      console.error('Error connecting to WebSocket:', error);
    }
  }, [clientId, documentContext, stopCallTimer]);

  // Handle WebSocket messages
  const handleWebSocketMessage = useCallback((message: WebSocketMessage) => {
    switch (message.type) {
      case 'connection_status':
        if (message.status === 'connected') {
          setIsConnected(true);
        }
        break;
        
      case 'call_status':
        if (message.status === 'started') {
          setIsCallActive(true);
          startCallTimer();
        } else if (message.status === 'ended') {
          setIsCallActive(false);
          stopCallTimer();
          setCallDuration(0);
        }
        break;
        
      case 'processing_status':
        if (message.status === 'processing') {
          setIsProcessing(true);
        }
        break;
        
      case 'transcription':
        setTranscribedText(message.text || '');
        break;
        
      case 'voice_response':
        setLastResponse(message.text || '');
        setIsProcessing(false);
        
        console.log('Received voice response:', {
          text: message.text,
          hasAudio: !!message.audio_data,
          audioLength: message.audio_data?.length || 0,
          isGreeting: message.is_greeting || false
        });
        
        // Play audio response
        if (message.audio_data) {
          playAudioResponse(message.audio_data);
        } else {
          console.warn('No audio data received in voice response');
        }
        break;
        
      case 'error':
        setIsProcessing(false);
        break;
        
      case 'context_set':
        console.log('Document context set successfully');
        break;
        
      default:
        console.log('Unknown message type:', message.type);
    }
  }, [startCallTimer, stopCallTimer]);

  // Play audio response
  const playAudioResponse = useCallback(async (audioBase64: string) => {
    try {
      console.log('Playing audio response, base64 length:', audioBase64.length);
      
      // Decode base64 audio
      const audioData = atob(audioBase64);
      const audioArray = new Uint8Array(audioData.length);
      for (let i = 0; i < audioData.length; i++) {
        audioArray[i] = audioData.charCodeAt(i);
      }
      
      console.log('Audio array length:', audioArray.length);
      
      // Create audio blob
      const audioBlob = new Blob([audioArray], { type: 'audio/wav' });
      const audioUrl = URL.createObjectURL(audioBlob);
      
      console.log('Audio URL created:', audioUrl);
      
      // Play audio with user interaction
      const audio = new Audio(audioUrl);
      
      // Set volume and other properties
      audio.volume = 1.0;
      audio.preload = 'auto';
      
      // Add event listeners for debugging
      audio.onloadstart = () => console.log('Audio loading started');
      audio.oncanplay = () => console.log('Audio can play');
      audio.onplay = () => console.log('Audio started playing');
      audio.onended = () => {
        console.log('Audio finished playing');
        URL.revokeObjectURL(audioUrl);
      };
      audio.onerror = (e) => console.error('Audio error:', e);
      
      // Try to play audio
      const playPromise = audio.play();
      if (playPromise !== undefined) {
        playPromise
          .then(() => {
            console.log('Audio playback started successfully');
          })
          .catch((error) => {
            console.error('Audio playback failed:', error);
          });
      }
      
    } catch (error) {
      console.error('Error playing audio response:', error);
    }
  }, []);

  // Format call duration
  const formatCallDuration = useCallback((seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }, []);

  // Start recording
  const startRecording = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true
        } 
      });
      
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });
      
      const audioChunks: Blob[] = [];
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.push(event.data);
        }
      };
      
      mediaRecorder.onstop = async () => {
        try {
          const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
          
          // Convert to base64 directly without WAV conversion
          const reader = new FileReader();
          reader.onload = () => {
            const base64Data = (reader.result as string).split(',')[1];
            
            // Send to WebSocket
            if (websocketRef.current && websocketRef.current.readyState === WebSocket.OPEN) {
              websocketRef.current.send(JSON.stringify({
                type: 'voice_audio',
                audio_data: base64Data
              }));
            }
          };
          reader.readAsDataURL(audioBlob);
          
        } catch (error) {
          console.error('Error processing recorded audio:', error);
        }
        
        // Stop all tracks
        stream.getTracks().forEach(track => track.stop());
      };
      
      mediaRecorder.start();
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = audioChunks;
      setIsRecording(true);
      
    } catch (error) {
      console.error('Error starting recording:', error);
    }
  }, []);

  // Stop recording
  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  }, [isRecording]);

  // Start call
  const startCall = useCallback(() => {
    if (websocketRef.current && websocketRef.current.readyState === WebSocket.OPEN) {
      websocketRef.current.send(JSON.stringify({
        type: 'start_call'
      }));
    } else {
    }
  }, []);

  // End call
  const endCall = useCallback(() => {
    if (websocketRef.current && websocketRef.current.readyState === WebSocket.OPEN) {
      websocketRef.current.send(JSON.stringify({
        type: 'end_call'
      }));
    }
    setIsCallActive(false);
    setIsRecording(false);
    stopCallTimer();
    setCallDuration(0);
    setTranscribedText('');
    setLastResponse('');
  }, [stopCallTimer]);

  // Connect on mount
  useEffect(() => {
    connectWebSocket();
    
    return () => {
      if (websocketRef.current) {
        websocketRef.current.close();
      }
      stopCallTimer();
    };
  }, [connectWebSocket, stopCallTimer]);

  // Reconnect when document context changes
  useEffect(() => {
    if (websocketRef.current && websocketRef.current.readyState === WebSocket.OPEN) {
      websocketRef.current.send(JSON.stringify({
        type: 'set_context',
        context: documentContext
      }));
    }
  }, [documentContext]);

  return (
    <div className="fixed bottom-4 right-4 z-50">
      <div className="bg-white rounded-lg shadow-lg border p-4 w-80">
        {/* Call Header */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-2">
            <div className={`w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
            <h3 className="text-lg font-semibold text-gray-900">Voice Bot</h3>
            {isCallActive && (
              <span className="text-sm text-green-600 font-medium">
                {formatCallDuration(callDuration)}
              </span>
            )}
          </div>
          <Button
            onClick={onClose}
            variant="ghost"
            size="sm"
            className="h-6 w-6 p-0 hover:bg-gray-100"
          >
            <X className="w-4 h-4" />
          </Button>
        </div>

        {/* Connection Status */}
        {!isConnected && (
          <div className="text-center mb-4 p-3 bg-yellow-50 rounded border">
            <p className="text-sm text-yellow-700">Connecting to voice bot...</p>
          </div>
        )}

        {/* Call Controls */}
        {isConnected && !isCallActive && (
          <div className="text-center mb-4">
            <Button
              onClick={startCall}
              className="bg-green-500 hover:bg-green-600 text-white rounded-full w-16 h-16 shadow-lg"
              disabled={!isConnected}
            >
              <Phone className="w-6 h-6" />
            </Button>
            <p className="text-sm text-gray-600 mt-2">Click to start call</p>
          </div>
        )}

        {/* Active Call Interface */}
        {isConnected && isCallActive && (
          <>
            {/* Call Status */}
            <div className="text-center mb-4">
              <div className="flex items-center justify-center space-x-2 mb-2">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                <span className="text-sm text-green-600 font-medium">Call Active</span>
              </div>
              <p className="text-xs text-gray-500">Duration: {formatCallDuration(callDuration)}</p>
            </div>

            {/* Recording Button */}
            <div className="flex justify-center mb-4">
              <Button
                onClick={isRecording ? stopRecording : startRecording}
                disabled={isProcessing}
                className={`w-16 h-16 rounded-full ${
                  isRecording 
                    ? 'bg-red-500 hover:bg-red-600' 
                    : 'bg-blue-500 hover:bg-blue-600'
                }`}
              >
                {isProcessing ? (
                  <Loader2 className="w-6 h-6 animate-spin" />
                ) : isRecording ? (
                  <MicOff className="w-6 h-6" />
                ) : (
                  <Mic className="w-6 h-6" />
                )}
              </Button>
            </div>

            {/* Recording Status */}
            {isRecording && (
              <div className="text-center mb-3">
                <div className="flex items-center justify-center space-x-2">
                  <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
                  <span className="text-sm text-red-600">Recording...</span>
                </div>
              </div>
            )}

            {isProcessing && (
              <div className="text-center mb-3">
                <div className="flex items-center justify-center space-x-2">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span className="text-sm text-blue-600">Processing...</span>
                </div>
              </div>
            )}

            {/* Conversation History */}
            <div className="max-h-32 overflow-y-auto mb-4 space-y-2">
              {transcribedText && (
                <div className="p-2 bg-blue-50 rounded border">
                  <p className="text-xs text-gray-700">
                    <strong>You:</strong> {transcribedText}
                  </p>
                </div>
              )}

              {lastResponse && (
                <div className="p-2 bg-green-50 rounded border">
                  <p className="text-xs text-gray-700">
                    <strong>Assistant:</strong> {lastResponse}
                  </p>
                </div>
              )}
            </div>

            {/* End Call Button */}
            <div className="text-center">
              <Button
                onClick={endCall}
                className="bg-red-500 hover:bg-red-600 text-white rounded-full w-12 h-12 p-0 shadow-lg"
              >
                <PhoneOff className="w-5 h-5" />
              </Button>
              <p className="text-xs text-gray-500 mt-1">End Call</p>
            </div>
          </>
        )}

        {/* Instructions */}
        <div className="mt-3 text-xs text-gray-500">
          {isCallActive ? (
            <p>Click the microphone to speak, or use the end call button to hang up</p>
          ) : (
            <p>Start a call to begin voice conversation with the document assistant</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default VoiceBot;
