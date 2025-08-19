# 📑 Adobe PDF Document Analysis with Voice Bot

A powerful and modern PDF document analysis platform that seamlessly integrates **Adobe PDF Embed API** with **AI-powered voice interaction**, **text-based chatbot**, and **web research capabilities**. Upload PDFs, navigate content using AI-extracted headings, and interact through intuitive voice or text interfaces. 🚀

## 🛠️ Quick Setup

### 📋 Prerequisites
- 🐳 **Docker** installed on your system
- 🔑 **Adobe PDF Embed API** credentials
- ☁️ **Google Cloud** credentials for Gemini API
- 🗣️ **Azure Speech Service** credentials (optional, for TTS)

### ⚙️ Environment Variables

Create a `.env` file or set these environment variables:

```bash
# Adobe PDF Embed API
ADOBE_EMBED_API_KEY=daf3f003d39241d48eaa29500f3ee516

# LLM Configuration
LLM_PROVIDER=gemini
GOOGLE_APPLICATION_CREDENTIALS=/credentials/service-account-key.json
GEMINI_MODEL=gemini-2.5-flash

# Text-to-Speech Configuration
TTS_PROVIDER=azure
AZURE_TTS_KEY=your_azure_tts_key
AZURE_TTS_ENDPOINT=https://your-region.tts.speech.microsoft.com
```

### 🏃‍♂️ Installation & Running

1. **Clone the Repository**  
   ```bash
   git clone <repository-url>
   cd adobe-pdf-voice-bot
   ```

2. **Add Google Cloud Service Account Key**  
   ```bash
   cp /path/to/your/service-account-key.json ./service-account-key.json
   ```

3. **Build the Docker Image**  
   ```bash
   docker build --platform linux/amd64 -t adobe .
   ```

4. **Run the Application**  
   ```bash
   docker run -v $(pwd)/service-account-key.json:/credentials/service-account-key.json \
     -e ADOBE_EMBED_API_KEY=your_adobe_client_id \
     -e LLM_PROVIDER=gemini \
     -e GOOGLE_APPLICATION_CREDENTIALS=/credentials/service-account-key.json \
     -e GEMINI_MODEL=gemini-2.5-flash \
     -e TTS_PROVIDER=azure \
     -e AZURE_TTS_KEY=your_azure_tts_key \
     -e AZURE_TTS_ENDPOINT=https://your-region.tts.speech.microsoft.com \
     -p 8080:8080 adobe
   ```

5. **Access the Application**  
   Open your browser and visit 🌐 `http://localhost:8080`

## ✨ Features

### 📄 PDF Document Management
- 🖼️ **Adobe PDF Embed API Integration**: High-quality PDF rendering with native Adobe features
- 📤 **Document Upload & Processing**: Seamlessly upload and extract structured content from PDFs
- 🧠 **AI-Powered Heading Extraction**: Automatically identify headings with page positions
- 🗺️ **Smart Navigation**: Jump to specific sections by clicking on headings

### 💡 AI-Powered Chatbot (Insights Bulb)
- 🗨️ **Contextual Conversations**: Engage with your document using AI that understands its content
- 🌐 **Web Research Integration**: Get enriched responses with real-time web searches
- 📜 **Conversation Memory**: Maintains context for smooth, multi-turn interactions
- 📝 **Markdown Support**: Rich text formatting with code highlighting
- 🔗 **Research Links**: Direct access to relevant, reliable web sources

### 🎙️ Voice Bot Assistant
- 🗣️ **Real-time Voice Interaction**: Speak naturally to interact with your document
- ⚡ **WebSocket Communication**: Low-latency audio streaming
- 🎤 **Speech-to-Text**: Whisper-based transcription with Voice Activity Detection (VAD)
- 🔊 **Text-to-Speech**: Azure TTS with fallback beep support
- 🔄 **Continuous Processing**: Stream audio for seamless conversations
- 📞 **Call Management**: Start/end calls with duration tracking

### 🔍 Web Research Service
- 🔎 **Intelligent Search**: Context-aware queries for accurate results
- ✅ **Quality Filtering**: Prioritizes educational and technical content
- 🌍 **Multiple Sources**: Aggregates data from trusted domains
- 🧩 **Enhanced Context**: Combines document content with web insights

### 🎨 Modern UI/UX
- 📱 **Responsive Design**: Optimized for desktop and mobile
- 🖥️ **Tabbed Interface**: Organized sidebar for headings and chatbot
- ⏳ **Real-time Updates**: Live transcription and response display
- 🛡️ **Error Handling**: Graceful recovery with user-friendly feedback
- 🔄 **Loading States**: Smooth animations and progress indicators

## 🏗️ Architecture

### 🖼️ Frontend (React + TypeScript)
- 📄 `PDFViewerPage.tsx`: Core PDF viewer with Adobe integration and voice bot
- 🎙️ `VoiceBot.tsx`: Voice interaction interface with WebSocket communication
- 💬 `ChatbotSidebar.tsx`: Text-based AI assistant with research capabilities
- 📑 `TabbedSidebar.tsx`: Document navigation with AI-extracted headings

### ⚙️ Backend (Python + FastAPI)
- 🗣️ `voice_service.py`: Speech processing, STT/TTS, and audio management
- 🌐 `websocket_handler.py`: Real-time WebSocket communication for voice bot
- 🔍 `web_search_service.py`: Web research and content aggregation
- 🤖 `gemini_utils.py`: LLM integration with Google Gemini
- 📜 `process_pdf.py`: PDF processing and heading extraction

## 🔧 Configuration

### 🔑 Adobe PDF Embed API
- Obtain your client ID from [Adobe PDF Embed API](https://www.adobe.com/go/dcsdks_credentials)
- Configure allowed domains in your Adobe dashboard

### ☁️ Google Cloud Setup
1. Create a Google Cloud project
2. Enable the Gemini API
3. Create a service account and download the JSON key
4. Place the key file in the project root

### 🗣️ Azure Speech Service (Optional)
1. Create an Azure Speech-Service resource
2. Retrieve the key and endpoint from the Azure portal
3. Configure the environment variables

## 🎯 Usage

### 📤 Uploading Documents
1. Navigate to the upload page
2. Select a PDF file
3. Wait for processing to complete
4. View the document with extracted headings

### 💬 Using the Chatbot
1. Open the "Insights Bulb" tab in the sidebar
2. Ask questions about your document
3. Enable web research for enhanced answers
4. Explore research links and sources

### 🎤 Using the Voice Bot
1. Click the microphone button in the bottom-right corner
2. Start a call to begin voice interaction
3. Speak naturally about your document
4. Listen to AI-generated responses

### 🗺️ Navigating Documents
1. Use the headings tab to view document structure
2. Click any heading to jump to that section
3. Utilize Adobe PDF controls for zoom, search, and more

## 🔒 Security Considerations
- 🔐 API keys stored securely as environment variables
- 🔗 WebSocket connections are encrypted
- 🛡️ File uploads are validated and sanitized
- 🌐 CORS configured for production safety

## 🚀 Performance Optimizations
- 🧠 **Model Caching**: Global caching for faster AI model startup
- 🎙️ **Audio Processing**: Optimized chunking and VAD for low latency
- ⚡ **WebSocket Management**: Efficient connection handling and cleanup
- 📜 **PDF Processing**: Asynchronous analysis for quick heading extraction

## 🐛 Troubleshooting

### ⚠️ Common Issues
1. **Adobe PDF Not Loading**  
   - Verify your Adobe client ID  
   - Ensure the domain is allowed in the Adobe dashboard  

2. **Voice Bot Not Working**  
   - Check microphone permissions  
   - Confirm WebSocket connection status  
   - Validate Azure TTS credentials  

3. **Chatbot Not Responding**  
   - Verify Google Cloud credentials  
   - Ensure Gemini API is enabled  
   - Check service account permissions  

### 📜 Logs
View detailed error information:  
```bash
docker logs <container_id>
```

## 🌟 Contributing
Contributions are welcome! Fork the repository, make your changes, and submit a pull request. Let's build something amazing together! 🚀