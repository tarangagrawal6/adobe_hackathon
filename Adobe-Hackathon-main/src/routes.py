import json
import logging
import os
import uuid
from typing import List
from pathlib import Path
from fastapi import APIRouter, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

# Set up logging
logger = logging.getLogger(__name__)

from .process_pdf import pdf_processor
from .types import ChatRequest, ChatResponse, EnhancedChatRequest, EnhancedChatResponse
from .websocket_handler import voice_bot_websocket
from .gemini_utils import call_gemini_api
from .web_search_service import WebSearchService

# Create router
router = APIRouter()

# Define directories
OUTPUT_DIR = Path("json")
UPLOADS_DIR = Path("uploads")

# Create directories if they don't exist
OUTPUT_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)




@router.get("/")
async def root():
    return {
        "message": "PDF Processing Server",
        "version": "1.0.0",
        "endpoints": {
            "process_pdf_headings": "/process-pdf-headings/",
            "initiate_chatbot": "/initiate-chatbot/",
            "chat": "/chat/",
            "chat_with_memory": "/chat-with-memory/",
            "enhanced_chat": "/enhanced-chat/",
            "list_pdfs": "/list-pdfs/",
            "get_pdf": "/get-pdf/{filename}",
            "get_headings": "/get-headings/{filename}",
            "delete_pdf": "/delete-pdf/{filename}",
            "voice_bot_ws": "/ws/voice-bot/{client_id}",
            "health": "/health",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "PDF Processing Server"
    }

@router.get("/api/v1/config/")
async def get_config():
    """Get frontend configuration including Adobe client ID"""
    adobe_client_id = os.getenv("ADOBE_EMBED_API_KEY")
    if not adobe_client_id:
        raise HTTPException(
            status_code=500,
            detail="Adobe Embed API key not configured. Please set ADOBE_EMBED_API_KEY environment variable."
        )
    
    return {
        "adobe_client_id": adobe_client_id
    }

@router.websocket("/ws/voice-bot/{client_id}")
async def voice_bot_websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for voice bot functionality"""
    try:
        # Connect to voice bot
        await voice_bot_websocket.connect(websocket, client_id)
        
        # Handle incoming messages
        while True:
            try:
                # Receive message
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle the message
                await voice_bot_websocket.handle_voice_message(websocket, client_id, message)
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for client {client_id}")
                break
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON received from client {client_id}: {str(e)}")
                try:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Invalid JSON format"
                    }))
                except Exception:
                    break  # Connection lost, exit loop
            except Exception as e:
                logger.error(f"Error handling WebSocket message for client {client_id}: {str(e)}")
                try:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"Internal server error: {str(e)}"
                    }))
                except Exception:
                    break  # Connection lost, exit loop
                
    except Exception as e:
        logger.error(f"WebSocket connection error for client {client_id}: {str(e)}")
    finally:
        # Clean up connection
        await voice_bot_websocket.disconnect(client_id)


@router.post("/chat/", response_model=ChatResponse)
async def chat_with_gemini(request: ChatRequest):
    """
    Chat with Gemini AI using document context.
    """
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        raise HTTPException(
            status_code=500,
            detail="Gemini API not configured. Please set GOOGLE_APPLICATION_CREDENTIALS environment variable."
        )
    
    try:
        # Create the prompt
        prompt = f"""You are a helpful PDF document assistant. You have access to the following document content as context. Please answer the user's question based on this content. If the question cannot be answered from the context, politely say so.

Document Context:
{request.context}

User Question: {request.message}

Please provide a helpful, accurate, and concise answer based on the document content."""

        # Generate response using Gemini
        response_text = call_gemini_api(prompt)
        
        return ChatResponse(
            response=response_text,
            status="success"
        )
        
    except Exception as e:
        logger.error(f"Error in chat_with_gemini: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}"
        )


@router.post("/chat-with-memory/", response_model=ChatResponse)
async def chat_with_gemini_memory(request: ChatRequest):
    """
    Chat with Gemini AI using document context and conversation history.
    """
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        raise HTTPException(
            status_code=500,
            detail="Gemini API not configured. Please set GOOGLE_APPLICATION_CREDENTIALS environment variable."
        )
    
    try:
        # Build the complete prompt with system instructions, conversation history, and current context
        system_prompt = request.system_prompt or "You are Insights Bulb, a helpful PDF document assistant."
        
        full_prompt = f"""{system_prompt}

Document Context:
{request.context}

{request.conversation_history if request.conversation_history else ''}Current User Question: {request.message}

Please respond as Insights Bulb, maintaining conversation context and providing helpful insights based on the document content."""

        # Generate response using Gemini
        response_text = call_gemini_api(full_prompt)
        
        return ChatResponse(
            response=response_text,
            status="success"
        )
        
    except Exception as e:
        logger.error(f"Error in chat_with_gemini_memory: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating response: {str(e)}"
        )


@router.post("/enhanced-chat/", response_model=EnhancedChatResponse)
async def enhanced_chat_with_research(request: EnhancedChatRequest):
    """
    Enhanced chat with Gemini AI using document context, conversation history, and web research.
    """
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        raise HTTPException(
            status_code=500,
            detail="Gemini API not configured. Please set GOOGLE_APPLICATION_CREDENTIALS environment variable."
        )
    
    try:
        enhanced_context = request.context
        research_links = []
        search_results = None
        
        # Ensure research_links is always initialized
        if not request.enable_research:
            search_results = {
                'query': request.message,
                'total_results': 0,
                'results': [],
                'search_url': '',
                'error': 'Research disabled'
            }
        
        # Perform web research if enabled
        if request.enable_research:
            try:
                async with WebSearchService() as web_search:
                    research_data = await web_search.get_research_context(
                        query=request.message,
                        document_context=request.context
                    )
                    
                    enhanced_context = research_data['enhanced_context']
                    research_links = research_data['research_links']
                    search_results = research_data['search_results']
                    
                    logger.info(f"Found {len(research_links)} research links for query: {request.message}")
                    
            except Exception as search_error:
                logger.warning(f"Web search failed, continuing with document context only: {str(search_error)}")
                # Continue with document context only if search fails
                search_results = {
                    'query': request.message,
                    'total_results': 0,
                    'results': [],
                    'search_url': '',
                    'error': str(search_error)
                }
                research_links = []  # Ensure research_links is initialized
        
        # Build the complete prompt with enhanced context
        system_prompt = request.system_prompt or """You are "Insights Bulb" - an intelligent AI assistant designed to help users understand and extract insights from PDF documents and web research. 

CORE BEHAVIOR GUIDELINES:
1. **Personality**: Be friendly, enthusiastic, and genuinely helpful. Use a warm, conversational tone while maintaining professionalism.
2. **Knowledge**: You have access to both document content and web research as context. Always base your answers on this comprehensive information.
3. **Accuracy**: If you cannot answer a question from the provided context, clearly state this rather than making assumptions.
4. **Clarity**: Provide clear, well-structured responses. Use bullet points or numbered lists when appropriate.
5. **Research Integration**: When web research is available, integrate it with document insights to provide comprehensive answers.
6. **Citation**: When referencing web sources, mention them naturally in your response.
7. **Insights**: Focus on extracting meaningful insights, patterns, and key takeaways from both document and web content."""

        full_prompt = f"""{system_prompt}

Enhanced Context (Document + Web Research):
{enhanced_context}

{request.conversation_history if request.conversation_history else ''}Current User Question: {request.message}

Please respond as Insights Bulb, providing comprehensive insights that combine document analysis with relevant web research when available."""

        # Generate response using Gemini
        response_text = call_gemini_api(full_prompt)
        
        # Ensure search_results has all required fields
        if search_results is None:
            search_results = {
                'query': request.message,
                'total_results': 0,
                'results': [],
                'search_url': '',
                'error': 'No search performed'
            }
        elif isinstance(search_results, dict) and 'error' in search_results:
            # Ensure error case has all required fields
            search_results = {
                'query': search_results.get('query', request.message),
                'total_results': search_results.get('total_results', 0),
                'results': search_results.get('results', []),
                'search_url': search_results.get('search_url', ''),
                'error': search_results.get('error', 'Unknown error')
            }
        
        return EnhancedChatResponse(
            response=response_text,
            status="success",
            research_links=research_links,
            search_results=search_results,
            enhanced_context=enhanced_context
        )
        
    except Exception as e:
        logger.error(f"Error in enhanced_chat_with_research: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating enhanced response: {str(e)}"
        )


@router.get("/list-pdfs/")
async def list_pdfs():
    """
    Get a list of all PDF files currently in the uploads folder.
    """
    try:
        pdf_files = list(UPLOADS_DIR.glob("*.pdf"))
        files_info = []
        
        for pdf_file in pdf_files:
            file_stat = pdf_file.stat()
            files_info.append({
                "filename": pdf_file.name,
                "size_bytes": file_stat.st_size,
                "created": file_stat.st_ctime,
                "modified": file_stat.st_mtime
            })
        
        return {
            "total_files": len(files_info),
            "files": files_info
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing PDF files: {str(e)}"
        )


@router.get("/get-pdf/{filename}")
async def get_pdf(filename: str):
    """
    Get a PDF file by its filename.
    """
    if not filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    file_path = UPLOADS_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"PDF file '{filename}' not found"
        )
    
    try:
        return FileResponse(
            path=str(file_path),
            media_type="application/pdf",
            filename=filename
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading PDF file: {str(e)}"
        )


@router.get("/get-headings/{filename}")
async def get_headings(filename: str):
    """
    Get headings JSON file for a specific PDF filename.
    The filename should be the base name without extension (e.g., 'pdf2' for 'pdf2.pdf').
    """
    # Construct the JSON filename
    json_filename = f"{filename}_headings.json"
    json_path = OUTPUT_DIR / json_filename
    
    if not json_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Headings file for '{filename}' not found. Available files: {[f.name for f in OUTPUT_DIR.glob('*_headings.json')]}"
        )
    
    try:
        # Read and parse the JSON file
        with open(json_path, 'r', encoding='utf-8') as json_file:
            headings_data = json.load(json_file)
        
        return JSONResponse(
            content=headings_data,
            headers={
                "Content-Type": "application/json",
                "Cache-Control": "no-cache"
            }
        )
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error parsing headings JSON file: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading headings file: {str(e)}"
        )


@router.delete("/delete-pdf/{filename}")
async def delete_pdf(filename: str):
    """
    Delete a PDF file and its corresponding headings JSON file.
    """
    if not filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    pdf_path = UPLOADS_DIR / filename
    json_filename = filename.replace('.pdf', '_headings.json')
    json_path = OUTPUT_DIR / json_filename
    
    deleted_files = []
    errors = []
    
    # Delete PDF file
    if pdf_path.exists():
        try:
            os.remove(pdf_path)
            deleted_files.append(filename)
        except Exception as e:
            errors.append(f"Error deleting PDF file: {str(e)}")
    else:
        errors.append(f"PDF file '{filename}' not found")
    
    # Delete corresponding JSON file
    if json_path.exists():
        try:
            os.remove(json_path)
            deleted_files.append(json_filename)
        except Exception as e:
            errors.append(f"Error deleting JSON file: {str(e)}")
    
    if errors:
        raise HTTPException(
            status_code=500,
            detail=f"Some errors occurred: {'; '.join(errors)}"
        )
    
    return {
        "message": f"Successfully deleted {len(deleted_files)} files",
        "deleted_files": deleted_files
    }


@router.post("/process-pdf-headings/")
async def process_pdf_headings(files: List[UploadFile] = File(...)):
    """
    Process PDF files to detect headings using LayoutLMv3 model.
    Saves PDFs to uploads folder and returns JSON files with detected titles, headings, and text.
    """
    if not files:
        raise HTTPException(
            status_code=400, 
            detail="No files provided. Please upload at least one PDF file."
        )
    
    results = []
    
    for file in files:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400, 
                detail=f"File '{file.filename}' is not a PDF. Only PDF files are supported."
            )
        
        try:
            # Read file content
            content = await file.read()
            
            # Validate file size (optional - 50MB limit)
            if len(content) > 50 * 1024 * 1024:  # 50MB
                raise HTTPException(
                    status_code=400,
                    detail=f"File '{file.filename}' is too large. Maximum size is 50MB."
                )
            
            # Save PDF to uploads folder
            pdf_path = UPLOADS_DIR / file.filename
            with open(pdf_path, 'wb') as pdf_file:
                pdf_file.write(content)
            
            # Process the PDF for headings using LayoutLMv3
            logger.info(f"Processing PDF headings for file: {file.filename}")
            headings_data = pdf_processor.process_pdf_headings(content, file.filename)
            
            # Save JSON output to file with _headings suffix
            json_filename = file.filename.replace('.pdf', '_headings.json')
            json_path = OUTPUT_DIR / json_filename
            
            with open(json_path, 'w', encoding='utf-8') as json_file:
                json.dump(headings_data, json_file, indent=2, ensure_ascii=False)
            
            # Create processing response
            processing_response = {
                "original_filename": file.filename,
                "pdf_saved_to": str(pdf_path),
                "json_filename": json_filename,
                "json_path": str(json_path),
                "processing_status": "success",
                "data": headings_data,
                "error_message": None
            }
            
            results.append(processing_response)
            
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            # Handle other exceptions
            error_response = {
                "original_filename": file.filename,
                "pdf_saved_to": "",
                "json_filename": "",
                "json_path": "",
                "processing_status": "error",
                "data": None,
                "error_message": str(e)
            }
            results.append(error_response)
    
    # Count successful processing
    successful_count = len([r for r in results if r["processing_status"] == "success"])
    
    # Return response
    return {
        "message": f"Processed {len(files)} PDF files for heading detection",
        "results": results,
        "total_files": len(files),
        "successful_processing": successful_count
    }


@router.post("/initiate-chatbot/")
async def initiate_chatbot(request: dict):
    """
    Extract all text from a PDF file and clean it for LLM consumption.
    If the text is more than 500,000 words, create a summary using Flan-T5-small.
    Returns the cleaned text or summary for chatbot use.
    """
    try:
        # Extract filename from request body
        filename = request.get("filename")
        
        # Validate filename
        if not filename:
            raise HTTPException(
                status_code=400,
                detail="Filename is required in request body"
            )
        
        # Ensure filename has .pdf extension
        if not filename.lower().endswith('.pdf'):
            filename = filename + '.pdf'
        
        # Check if PDF file exists
        pdf_path = UPLOADS_DIR / filename
        if not pdf_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"PDF file '{filename}' not found in uploads directory"
            )
        
        logger.info(f"Initiating chatbot for PDF: {filename}")
        
        # Extract and process text using the PDF processor
        result = pdf_processor.extract_pdf_text_for_chatbot(filename)
        
        if result["status"] == "error":
            raise HTTPException(
                status_code=500,
                detail=f"Error processing PDF: {result.get('error_message', 'Unknown error')}"
            )
        
        # Return the processed text data
        return {
            "message": "PDF text extracted successfully for chatbot",
            "filename": result["filename"],
            "original_word_count": result["original_word_count"],
            "summary_word_count": result["summary_word_count"],
            "is_summarized": result["is_summarized"],
            "text": result["text"],
            "processing_status": "success"
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error in initiate_chatbot: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error processing PDF: {str(e)}"
        )
