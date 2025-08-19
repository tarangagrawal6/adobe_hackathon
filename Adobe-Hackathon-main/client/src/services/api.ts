import { API_BASE_URL } from '../main';

export interface PDFData {
  filename: string;
  original_word_count: number;
  summary_word_count: number;
  text: string;
  is_summarized: boolean;
  processing_status: string;
}

export interface ChatResponse {
  response: string;
  status: string;
  error_message?: string;
}

export interface SearchResult {
  title: string;
  url: string;
  snippet: string;
  content: string;
  relevance_score: number;
}

export interface WebSearchResults {
  query: string;
  total_results: number;
  results: SearchResult[];
  search_url: string;
  error?: string;
}

export interface EnhancedChatResponse {
  response: string;
  status: string;
  research_links: string[];
  search_results?: WebSearchResults;
  enhanced_context: string;
  error_message?: string;
}

export const api = {
  async initiateChatbot(filename: string): Promise<PDFData> {
    const response = await fetch(`${API_BASE_URL}/initiate-chatbot/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ filename }),
    });

    if (!response.ok) {
      throw new Error(`Failed to initiate chatbot: ${response.statusText}`);
    }

    const data = await response.json();
    return data;
  },

  async chatWithGemini(message: string, context: string): Promise<string> {
    const response = await fetch(`${API_BASE_URL}/chat/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message,
        context,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(`Failed to get response from chat API: ${errorData.detail || response.statusText}`);
    }

    const data: ChatResponse = await response.json();
    
    if (data.status !== 'success') {
      throw new Error(data.error_message || 'Chat request failed');
    }
    
    return data.response;
  },

  async chatWithGeminiWithMemory(
    message: string, 
    context: string, 
    conversationHistory: string, 
    systemPrompt: string
  ): Promise<string> {
    const response = await fetch(`${API_BASE_URL}/chat-with-memory/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message,
        context,
        conversation_history: conversationHistory,
        system_prompt: systemPrompt,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(`Failed to get response from chat API: ${errorData.detail || response.statusText}`);
    }

    const data: ChatResponse = await response.json();
    
    if (data.status !== 'success') {
      throw new Error(data.error_message || 'Chat request failed');
    }
    
    return data.response;
  },

  async enhancedChatWithResearch(
    message: string, 
    context: string, 
    conversationHistory: string, 
    systemPrompt: string,
    enableResearch: boolean = true,
    maxSearchResults: number = 6
  ): Promise<EnhancedChatResponse> {
    const response = await fetch(`${API_BASE_URL}/enhanced-chat/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message,
        context,
        conversation_history: conversationHistory,
        system_prompt: systemPrompt,
        enable_research: enableResearch,
        max_search_results: maxSearchResults,
      }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(`Failed to get response from enhanced chat API: ${errorData.detail || response.statusText}`);
    }

    const data: EnhancedChatResponse = await response.json();
    
    if (data.status !== 'success') {
      throw new Error(data.error_message || 'Enhanced chat request failed');
    }
    
    return data;
  }
};
