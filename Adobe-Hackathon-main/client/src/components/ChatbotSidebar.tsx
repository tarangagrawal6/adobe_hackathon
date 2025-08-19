import React, { useState, useEffect, useRef } from 'react';
import { Send, User, Loader2, FileText, AlertCircle, Lightbulb, Trash2, ExternalLink, Globe, Search } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Card, CardContent } from './ui/card';
import { ScrollArea } from './ui/scroll-area';
import { usePDF } from '../contexts/PDFContext';
import { api, type EnhancedChatResponse } from '../services/api';
import { toast } from 'sonner';

interface Message {
  id: string;
  content: string;
  sender: 'user' | 'bot';
  timestamp: Date;
  researchLinks?: string[];
  searchResults?: any;
}

interface ChatbotSidebarProps {
  filename: string;
}

// Detailed system prompt for Insights Bulb
const SYSTEM_PROMPT = `You are "Insights Bulb" - an intelligent AI assistant designed to help users understand and extract insights from PDF documents. 

CORE BEHAVIOR GUIDELINES:
1. **Personality**: Be friendly, enthusiastic, and genuinely helpful. Use a warm, conversational tone while maintaining professionalism.
2. **Knowledge**: You have access to the document content provided as context. Always base your answers on this content.
3. **Accuracy**: If you cannot answer a question from the provided context, clearly state this rather than making assumptions.
4. **Clarity**: Provide clear, well-structured responses. Use bullet points or numbered lists when appropriate.
5. **Engagement**: Ask follow-up questions when relevant to better understand user needs.
6. **Insights**: Focus on extracting meaningful insights, patterns, and key takeaways from the document.

CONVERSATION GUIDELINES:
- Remember previous messages in the conversation to maintain context
- Reference earlier parts of the conversation when relevant
- Provide comprehensive answers while being concise
- Use examples from the document when helpful
- Acknowledge user's previous questions and build upon them

RESPONSE FORMAT:
- Start with a brief acknowledgment if continuing a conversation thread
- Provide the main answer with clear structure
- Include relevant quotes or references from the document when appropriate
- End with a helpful follow-up or offer additional assistance

Remember: You are here to illuminate insights and make complex information accessible and engaging!`;

const ChatbotSidebar: React.FC<ChatbotSidebarProps> = ({ filename }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isInitializing, setIsInitializing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [enableResearch, setEnableResearch] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const { hasPDFData, getPDFData, setPDFData } = usePDF();

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Initialize chatbot when component mounts
  useEffect(() => {
    // Clear messages when filename changes
    setMessages([]);
    initializeChatbot();
  }, [filename]);

  const initializeChatbot = async () => {
    if (hasPDFData(filename)) {
      // Data already exists, add welcome message only if no messages exist
      if (messages.length === 0) {
        addBotMessage("Hello! I'm Insights Bulb, your intelligent document assistant! 💡 I have access to your document and I'm here to help you extract valuable insights and answer any questions you might have. What would you like to explore today?");
      }
      return;
    }

    setIsInitializing(true);
    setError(null);

    try {
      const data = await api.initiateChatbot(filename);
      setPDFData(filename, data);
      
      // Add welcome message only if no messages exist
      if (messages.length === 0) {
        addBotMessage("Hello! I'm Insights Bulb, your intelligent document assistant! 💡 I have processed your document and I'm ready to help you discover valuable insights and answer any questions. What would you like to explore today?");
      }
    } catch (err) {
      console.error('Error initializing chatbot:', err);
      setError('Failed to load document data. Please try again.');
      toast.error('Failed to load document data');
    } finally {
      setIsInitializing(false);
    }
  };

  const addBotMessage = (content: string, researchLinks?: string[], searchResults?: any) => {
    const newMessage: Message = {
      id: Date.now().toString(),
      content,
      sender: 'bot',
      timestamp: new Date(),
      researchLinks,
      searchResults,
    };
    setMessages(prev => [...prev, newMessage]);
  };

  const addUserMessage = (content: string) => {
    const newMessage: Message = {
      id: Date.now().toString(),
      content,
      sender: 'user',
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, newMessage]);
  };

  // Build conversation history for context
  const buildConversationHistory = (): string => {
    if (messages.length === 0) return '';
    
    return messages.map(msg => 
      `${msg.sender === 'user' ? 'User' : 'Assistant'}: ${msg.content}`
    ).join('\n\n');
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = inputMessage.trim();
    setInputMessage('');
    addUserMessage(userMessage);
    setIsLoading(true);

    try {
      const pdfData = getPDFData(filename);
      if (!pdfData) {
        throw new Error('PDF data not available');
      }

      // Build conversation context
      const conversationHistory = buildConversationHistory();
      
      // Use enhanced chat with research
      const response: EnhancedChatResponse = await api.enhancedChatWithResearch(
        userMessage, 
        pdfData.text, 
        conversationHistory,
        SYSTEM_PROMPT,
        enableResearch,
        6
      );
      
      // Add bot message with research links
      addBotMessage(response.response, response.research_links, response.search_results);
      
      // Show toast if research was performed
      if (response.research_links && response.research_links.length > 0) {
        toast.success(`Found ${response.research_links.length} research sources!`, {
          description: 'Check the links below for more information.'
        });
      }
      
    } catch (err) {
      console.error('Error sending message:', err);
      addBotMessage("I'm sorry, I encountered an error while processing your request. Please try again.");
      toast.error('Failed to get response');
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const clearConversation = () => {
    setMessages([]);
    toast.success('Conversation cleared');
  };

  const openResearchLink = (url: string) => {
    window.open(url, '_blank', 'noopener,noreferrer');
  };

  if (isInitializing) {
    return (
      <div className="flex flex-col h-full overflow-hidden">
        <div className="flex items-center p-4 border-b border-gray-200 bg-gradient-to-r from-yellow-50 to-orange-50 min-h-[60px]">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-yellow-500 rounded-lg">
              <Lightbulb className="h-5 w-5 text-white" />
            </div>
            <h2 className="text-lg font-semibold text-gray-900">Insights Bulb</h2>
          </div>
        </div>
        
        <div className="flex-1 flex items-center justify-center p-8">
          <div className="text-center">
            <Loader2 className="h-12 w-12 animate-spin text-yellow-500 mx-auto mb-4" />
            <p className="text-sm text-gray-600 mb-2">Processing document...</p>
            <p className="text-xs text-gray-500">This may take a few moments</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="flex items-center p-4 border-b border-gray-200 bg-gradient-to-r from-yellow-50 to-orange-50 min-h-[60px]">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-yellow-500 rounded-lg">
            <Lightbulb className="h-5 w-5 text-white" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-gray-900">Insights Bulb</h2>
            <p className="text-xs text-gray-600">AI Document Assistant</p>
          </div>
        </div>
        <div className="ml-auto flex items-center space-x-2">
          {/* Research Toggle */}
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setEnableResearch(!enableResearch)}
            className={`h-8 px-3 text-xs ${
              enableResearch 
                ? 'bg-green-100 text-green-700 border-green-200 hover:bg-green-200' 
                : 'bg-gray-100 text-gray-700 border-gray-200 hover:bg-gray-200'
            }`}
            title={enableResearch ? 'Web research enabled' : 'Web research disabled'}
          >
            <Globe className="h-3 w-3 mr-1" />
            {enableResearch ? 'Research ON' : 'Research OFF'}
          </Button>
          {messages.length > 0 && (
            <Button
              variant="ghost"
              size="sm"
              onClick={clearConversation}
              className="h-8 w-8 p-0 text-gray-500 hover:text-red-500 hover:bg-red-50"
              title="Clear conversation"
            >
              <Trash2 className="h-4 w-4" />
            </Button>
          )}
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="p-4 border-b border-red-200 bg-red-50">
          <div className="flex items-center space-x-2 mb-2">
            <AlertCircle className="h-4 w-4 text-red-500" />
            <p className="text-sm text-red-700">{error}</p>
          </div>
          <Button 
            variant="outline" 
            size="sm" 
            className="text-xs"
            onClick={initializeChatbot}
          >
            Retry
          </Button>
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-hidden min-h-0">
        <ScrollArea className="h-full">
          <div className="p-4 space-y-4 pb-4">
            {messages.length === 0 && !error && (
              <div className="text-center py-8">
                <Lightbulb className="h-12 w-12 text-yellow-400 mx-auto mb-4" />
                <p className="text-sm text-gray-500 mb-3">Start a conversation about your document</p>
                <div className="text-xs text-gray-400 space-y-2">
                  <p>💡 I remember our conversation history</p>
                  <p>📄 I can reference specific parts of your document</p>
                  <p>🔍 I'll help you discover insights and patterns</p>
                  <p>🌐 I can search the web for additional research</p>
                </div>
              </div>
            )}
            
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div className={`flex max-w-[90%] min-w-0 ${message.sender === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                  <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                    message.sender === 'user' 
                      ? 'bg-blue-500 text-white ml-3' 
                      : 'bg-yellow-500 text-white mr-3'
                  }`}>
                    {message.sender === 'user' ? (
                      <User className="h-4 w-4" />
                    ) : (
                      <Lightbulb className="h-4 w-4" />
                    )}
                  </div>
                  
                  <Card className={`min-w-0 max-w-full ${
                    message.sender === 'user' 
                      ? 'bg-gradient-to-r from-blue-500 to-blue-600 text-white shadow-sm' 
                      : 'bg-gradient-to-r from-yellow-50 to-orange-50 text-gray-900 shadow-sm border border-yellow-200'
                  }`}>
                    <CardContent className="p-4 overflow-hidden">
                      {message.sender === 'bot' ? (
                        <div className="prose prose-sm max-w-full text-sm leading-relaxed break-words">
                          <ReactMarkdown
                            remarkPlugins={[remarkGfm]}
                            rehypePlugins={[rehypeHighlight]}
                            components={{
                              // Custom styling for markdown elements
                              h1: ({ children }) => <h1 className="text-lg font-bold mb-2 text-gray-900 break-words">{children}</h1>,
                              h2: ({ children }) => <h2 className="text-base font-bold mb-2 text-gray-900 break-words">{children}</h2>,
                              h3: ({ children }) => <h3 className="text-sm font-bold mb-2 text-gray-900 break-words">{children}</h3>,
                              p: ({ children }) => <p className="mb-2 last:mb-0 break-words">{children}</p>,
                              ul: ({ children }) => <ul className="list-disc list-inside mb-2 space-y-1 break-words">{children}</ul>,
                              ol: ({ children }) => <ol className="list-decimal list-inside mb-2 space-y-1 break-words">{children}</ol>,
                              li: ({ children }) => <li className="text-sm break-words">{children}</li>,
                              blockquote: ({ children }) => (
                                <blockquote className="border-l-4 border-yellow-300 pl-3 italic text-gray-700 mb-2 break-words">
                                  {children}
                                </blockquote>
                              ),
                              code: ({ children, className }) => {
                                const isInline = !className;
                                return isInline ? (
                                  <code className="bg-gray-100 text-gray-800 px-1 py-0.5 rounded text-xs font-mono break-all">
                                    {children}
                                  </code>
                                ) : (
                                  <pre className="bg-gray-100 p-3 rounded-lg overflow-x-auto mb-2 break-words">
                                    <code className="text-xs font-mono break-all">{children}</code>
                                  </pre>
                                );
                              },
                              strong: ({ children }) => <strong className="font-semibold break-words">{children}</strong>,
                              em: ({ children }) => <em className="italic break-words">{children}</em>,
                              a: ({ children, href }) => (
                                <a 
                                  href={href} 
                                  target="_blank" 
                                  rel="noopener noreferrer"
                                  className="text-blue-600 hover:text-blue-800 underline break-all"
                                >
                                  {children}
                                </a>
                              ),
                            }}
                          >
                            {message.content}
                          </ReactMarkdown>
                        </div>
                      ) : (
                        <p className="text-sm whitespace-pre-wrap leading-relaxed">{message.content}</p>
                      )}
                      
                      {/* Research Links */}
                      {message.sender === 'bot' && message.researchLinks && message.researchLinks.length > 0 && (
                        <div className="mt-4 pt-3 border-t border-yellow-200">
                          <div className="flex items-center space-x-2 mb-3">
                            <Search className="h-4 w-4 text-yellow-600 flex-shrink-0" />
                            <span className="text-sm font-medium text-yellow-700">Research Sources:</span>
                          </div>
                          <div className="space-y-2">
                            {message.researchLinks.map((link, index) => {
                              // Try to extract a meaningful title from the URL
                              const url = new URL(link);
                              const pathParts = url.pathname.split('/').filter(part => part.length > 0);
                              let title = url.hostname.replace('www.', '');
                              
                              // Try to get a better title from the path
                              if (pathParts.length > 0) {
                                const lastPart = pathParts[pathParts.length - 1];
                                if (lastPart && lastPart.length > 3) {
                                  // Convert URL-friendly text to readable text
                                  const readableTitle = lastPart
                                    .replace(/[-_]/g, ' ')
                                    .replace(/\b\w/g, l => l.toUpperCase());
                                  title = readableTitle;
                                }
                              }
                              
                              return (
                                <button
                                  key={index}
                                  onClick={() => openResearchLink(link)}
                                  className="flex items-start space-x-2 text-sm text-blue-600 hover:text-blue-800 hover:bg-blue-50 p-2 rounded-lg w-full text-left transition-colors break-words"
                                  title={link}
                                >
                                  <ExternalLink className="h-4 w-4 flex-shrink-0 mt-0.5" />
                                  <div className="flex-1 min-w-0 overflow-hidden">
                                    <div className="font-medium break-words line-clamp-2">{title}</div>
                                    <div className="text-gray-500 break-all text-xs mt-1">
                                      {url.hostname.replace('www.', '')}
                                    </div>
                                  </div>
                                </button>
                              );
                            })}
                          </div>
                        </div>
                      )}
                      
                      <p className={`text-xs mt-3 ${
                        message.sender === 'user' ? 'text-blue-100' : 'text-gray-500'
                      }`}>
                        {formatTime(message.timestamp)}
                      </p>
                    </CardContent>
                  </Card>
                </div>
              </div>
            ))}
            
            {isLoading && (
              <div className="flex justify-start">
                <div className="flex max-w-[90%]">
                  <div className="flex-shrink-0 w-8 h-8 rounded-full bg-yellow-500 text-white mr-3 flex items-center justify-center">
                    <Lightbulb className="h-4 w-4" />
                  </div>
                  <Card className="bg-gradient-to-r from-yellow-50 to-orange-50 shadow-sm border border-yellow-200">
                    <CardContent className="p-4">
                      <div className="flex items-center space-x-3">
                        <Loader2 className="h-5 w-5 animate-spin text-yellow-500" />
                        <p className="text-sm text-gray-600">
                          {enableResearch ? 'Researching and thinking...' : 'Thinking...'}
                        </p>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>
        </ScrollArea>
      </div>

      {/* Input - Fixed at bottom */}
      <div className="border-t border-gray-200 bg-white shadow-sm">
        <div className="p-4">
          <div className="flex space-x-3">
            <Input
              ref={inputRef}
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={enableResearch ? "Ask Insights Bulb (with web research)..." : "Ask Insights Bulb..."}
              className="flex-1 text-sm h-10"
              disabled={isLoading || !!error}
            />
            <Button
              onClick={handleSendMessage}
              disabled={!inputMessage.trim() || isLoading || !!error}
              size="sm"
              className="px-4 h-10 bg-yellow-500 hover:bg-yellow-600"
            >
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </div>
        
        {/* Document Info */}
        {getPDFData(filename) && (
          <div className="px-4 pb-3 flex items-center justify-between text-xs text-gray-500 bg-gray-50">
            <div className="flex items-center space-x-2">
              <FileText className="h-4 w-4" />
              <span className="truncate">{filename}</span>
            </div>
            <span className="flex-shrink-0">
              {getPDFData(filename)?.summary_word_count.toLocaleString()} words
            </span>
          </div>
        )}
      </div>
    </div>
  );
};

export default ChatbotSidebar;
