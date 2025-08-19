from typing import List, Union, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class ContentType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


class TableContent(BaseModel):
    columns: List[str] = Field(..., description="Column headers of the table")
    rows: List[List[str]] = Field(..., description="Table data rows")


class ContentItem(BaseModel):
    id: int = Field(..., description="Unique identifier for the content item")
    type: ContentType = Field(..., description="Type of content (text, table, or image)")
    content: Union[str, TableContent, str] = Field(..., description="Content data - string for text/image, TableContent for tables")
    page: int = Field(..., description="Page number where this content was found")


class PDFProcessingResult(BaseModel):
    filename: str = Field(..., description="Original PDF filename")
    data: List[ContentItem] = Field(..., description="List of extracted content items")


class ProcessingStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    PROCESSING = "processing"


class ProcessingResponse(BaseModel):
    original_filename: str = Field(..., description="Original PDF filename")
    json_filename: str = Field(..., description="Generated JSON filename")
    json_path: str = Field(..., description="Path to the generated JSON file")
    processing_status: ProcessingStatus = Field(..., description="Status of processing")
    data: PDFProcessingResult = Field(None, description="Processed data (if successful)")
    error_message: str = Field(None, description="Error message (if failed)")


class BatchProcessingResponse(BaseModel):
    message: str = Field(..., description="Summary message")
    results: List[ProcessingResponse] = Field(..., description="Results for each processed file")
    total_files: int = Field(..., description="Total number of files processed")
    successful_processing: int = Field(..., description="Number of successfully processed files")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")


class RootResponse(BaseModel):
    message: str = Field(..., description="Service message")
    version: str = Field(..., description="API version")
    endpoints: Dict[str, str] = Field(..., description="Available endpoints")


# Chat-related models
class ChatRequest(BaseModel):
    message: str = Field(..., description="User's message/question")
    context: str = Field(..., description="Document context for the chat")
    conversation_history: Optional[str] = Field(None, description="Previous conversation history")
    system_prompt: Optional[str] = Field(None, description="System prompt for the AI")


class ChatResponse(BaseModel):
    response: str = Field(..., description="AI's response to the user's message")
    status: str = Field(..., description="Status of the chat request")
    error_message: Optional[str] = Field(None, description="Error message if any")


# Enhanced chat with research capabilities
class SearchResult(BaseModel):
    title: str = Field(..., description="Title of the search result")
    url: str = Field(..., description="URL of the search result")
    snippet: str = Field(..., description="Brief snippet of the content")
    content: str = Field(..., description="Extracted content from the page")
    relevance_score: float = Field(..., description="Relevance score of the result")


class WebSearchResults(BaseModel):
    query: str = Field(..., description="Original search query")
    total_results: int = Field(..., description="Total number of results found")
    results: List[SearchResult] = Field(..., description="List of search results")
    search_url: str = Field(..., description="URL of the search page")
    error: Optional[str] = Field(None, description="Error message if search failed")


class EnhancedChatRequest(BaseModel):
    message: str = Field(..., description="User's message/question")
    context: str = Field(..., description="Document context for the chat")
    conversation_history: Optional[str] = Field(None, description="Previous conversation history")
    system_prompt: Optional[str] = Field(None, description="System prompt for the AI")
    enable_research: bool = Field(True, description="Whether to enable web research")
    max_search_results: int = Field(6, description="Maximum number of search results to include")


class EnhancedChatResponse(BaseModel):
    response: str = Field(..., description="AI's response to the user's message")
    status: str = Field(..., description="Status of the chat request")
    research_links: List[str] = Field(..., description="List of research links found")
    search_results: Optional[WebSearchResults] = Field(None, description="Detailed search results")
    enhanced_context: str = Field(..., description="Enhanced context including web research")
    error_message: Optional[str] = Field(None, description="Error message if any")
