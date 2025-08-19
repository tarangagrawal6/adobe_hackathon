import asyncio
import logging
import re
from typing import List, Dict, Any
from crawl4ai import AsyncWebCrawler
from urllib.parse import quote_plus, urlparse

logger = logging.getLogger(__name__)

class WebSearchService:
    def __init__(self):
        self.crawler = None
        self.search_engines = {
            'google': 'https://www.google.com/search?q={}',
            'bing': 'https://www.bing.com/search?q={}',
            'duckduckgo': 'https://duckduckgo.com/?q={}'
        }
    
    async def __aenter__(self):
        self.crawler = AsyncWebCrawler()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.crawler = None
    
    def _build_simple_query(self, query: str) -> str:
        """
        Build a simple, clean search query that focuses on educational content.
        """
        # Clean the query - remove quotes and extra spaces
        clean_query = query.strip().replace('"', '').replace("'", "")
        
        # Add educational keywords to improve results
        educational_keywords = "tutorial guide explanation article"
        
        # Build a simple query with educational focus
        simple_query = f'"{clean_query}" {educational_keywords}'
        
        # Add some quality domains but keep it simple
        quality_domains = [
            "site:towardsdatascience.com",
            "site:medium.com",
            "site:analyticsvidhya.com",
            "site:machinelearningmastery.com",
            "site:distill.pub"
        ]
        
        # Combine query with domains
        final_query = f'{simple_query} ({" OR ".join(quality_domains)})'
        
        return final_query
    
    async def search_and_extract(self, query: str, max_results: int = 6) -> Dict[str, Any]:
        """
        Simple web search that extracts relevant URLs and content.
        """
        try:
            if not self.crawler:
                self.crawler = AsyncWebCrawler()
            
            # Build simple query
            search_query = self._build_simple_query(query)
            encoded_query = quote_plus(search_query)
            search_url = self.search_engines['google'].format(encoded_query)
            
            logger.info(f"Searching for: {search_query}")
            
            # Crawl search results
            result = await asyncio.wait_for(
                self.crawler.arun(
                    url=search_url,
                    strategy="adaptive"
                ),
                timeout=20.0
            )
            
            # Extract URLs from the search results
            urls = self._extract_urls_from_search(result.markdown)
            
            # Filter and validate URLs
            valid_urls = self._filter_valid_urls(urls, query)
            
            # Limit results
            valid_urls = valid_urls[:max_results]
            
            # Create results structure
            results = []
            for url in valid_urls:
                results.append({
                    'title': self._extract_title_from_url(url, query),
                    'url': url,
                    'snippet': f"Article about {query}",
                    'content': f"Content about {query} from {urlparse(url).netloc}",
                    'relevance_score': 0.8
                })
            
            return {
                'query': query,
                'total_results': len(results),
                'results': results,
                'search_url': search_url
            }
            
        except Exception as e:
            logger.error(f"Error in search_and_extract: {str(e)}")
            return {
                'query': query,
                'total_results': 0,
                'results': [],
                'search_url': '',
                'error': str(e)
            }
    
    def _extract_urls_from_search(self, markdown_content: str) -> List[str]:
        """
        Extract URLs from search results markdown content.
        """
        urls = []
        
        # Find all URLs in the content
        url_pattern = r'https?://[^\s\)\]]+'
        found_urls = re.findall(url_pattern, markdown_content)
        
        # Clean and validate URLs
        for url in found_urls:
            # Remove trailing punctuation
            clean_url = url.rstrip('.,;:!?')
            
            # Basic URL validation
            if self._is_valid_url(clean_url):
                urls.append(clean_url)
        
        return urls
    
    def _is_valid_url(self, url: str) -> bool:
        """
        Basic URL validation.
        """
        try:
            parsed = urlparse(url)
            return bool(parsed.netloc and parsed.scheme)
        except:
            return False
    
    def _filter_valid_urls(self, urls: List[str], query: str) -> List[str]:
        """
        Filter URLs to keep only relevant, article-like URLs.
        """
        valid_urls = []
        
        # Keywords that indicate good content
        good_keywords = [
            'blog', 'article', 'tutorial', 'guide', 'explanation', 
            'introduction', 'understanding', 'what-is', 'how-to',
            'machine-learning', 'deep-learning', 'ai', 'neural',
            'transformer', 'attention', 'self-attention'
        ]
        
        # Domains known for good content
        good_domains = [
            'towardsdatascience.com', 'medium.com', 'analyticsvidhya.com',
            'machinelearningmastery.com', 'distill.pub', 'arxiv.org',
            'dev.to', 'realpython.com', 'geeksforgeeks.org', 'tutorialspoint.com',
            'pytorch.org', 'tensorflow.org', 'huggingface.co', 'openai.com'
        ]
        
        for url in urls:
            url_lower = url.lower()
            
            # Skip search engines and social media
            if any(skip in url_lower for skip in [
                'google.com/search', 'bing.com/search', 'duckduckgo.com',
                'youtube.com', 'facebook.com', 'twitter.com', 'linkedin.com',
                'reddit.com', 'stackoverflow.com', 'github.com'
            ]):
                continue
            
            # Check if URL has meaningful path
            parsed = urlparse(url)
            if not parsed.path or parsed.path == '/' or len(parsed.path) < 5:
                continue
            
            # Check if URL contains good keywords
            has_good_keywords = any(keyword in url_lower for keyword in good_keywords)
            
            # Check if URL is from good domain
            is_good_domain = any(domain in url_lower for domain in good_domains)
            
            # Include if it has good keywords or is from good domain
            if has_good_keywords or is_good_domain:
                valid_urls.append(url)
        
        return valid_urls
    
    def _extract_title_from_url(self, url: str, query: str) -> str:
        """
        Extract a meaningful title from URL.
        """
        try:
            parsed = urlparse(url)
            path_parts = parsed.path.split('/')
            
            # Find the most meaningful part of the path
            for part in reversed(path_parts):
                if part and len(part) > 3:
                    # Convert URL-friendly text to readable text
                    title = part.replace('-', ' ').replace('_', ' ')
                    title = ' '.join(word.capitalize() for word in title.split())
                    
                    # If it's too long, truncate
                    if len(title) > 60:
                        title = title[:57] + "..."
                    
                    return title
            
            # Fallback to domain-based title
            domain = parsed.netloc.replace('www.', '')
            return f"Article about {query} from {domain}"
            
        except:
            return f"Article about {query}"
    
    async def get_research_context(self, query: str, document_context: str = "") -> Dict[str, Any]:
        """
        Get research context by combining document context with web search results.
        """
        try:
            # Search for additional information
            search_results = await self.search_and_extract(query)
            
            # Combine document context with web search results
            combined_context = document_context
            
            if search_results.get('results'):
                combined_context += "\n\n=== ADDITIONAL RESEARCH FROM THE WEB ===\n\n"
                
                for i, result in enumerate(search_results['results'], 1):
                    combined_context += f"Source {i}: {result['title']}\n"
                    combined_context += f"URL: {result['url']}\n"
                    combined_context += f"Content: {result.get('content', result.get('snippet', 'No content available'))}\n\n"
            
            return {
                'enhanced_context': combined_context,
                'search_results': search_results,
                'research_links': [result['url'] for result in search_results.get('results', [])]
            }
            
        except Exception as e:
            logger.error(f"Error in get_research_context: {str(e)}")
            return {
                'enhanced_context': document_context,
                'search_results': {'error': str(e)},
                'research_links': []
            }
