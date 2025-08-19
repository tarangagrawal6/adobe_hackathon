import os
import logging
import requests
import ssl
import urllib3
import asyncio
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from dotenv import load_dotenv
load_dotenv() 

logger = logging.getLogger(__name__)

# Disable SSL warnings for development
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Initialize Gemini API with service account
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

def get_gemini_access_token():
    """Get access token from service account credentials"""
    try:
        # Re-read environment variable at runtime
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not credentials_path:
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
        
        # Load service account credentials
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=["https://www.googleapis.com/auth/generative-language"]
        )
        
        # Get access token with better error handling
        try:
            credentials.refresh(Request())
            token = credentials.token
            return token
        except Exception as auth_error:
            logger.error(f"Authentication error: {str(auth_error)}")
            # Check if it's an SSL issue
            if "SSL" in str(auth_error) or "EOF" in str(auth_error):
                logger.error("SSL/TLS connection issue detected. This might be due to:")
                logger.error("1. Network connectivity issues")
                logger.error("2. Corporate firewall/proxy settings")
                logger.error("3. SSL certificate issues")
                logger.error("4. Outdated SSL libraries")
            raise
        
    except Exception as e:
        logger.error(f"Error getting access token: {str(e)}")
        raise

def call_gemini_api(prompt):
    """Call Gemini API with proper authentication and SSL handling"""
    try:
        access_token = get_gemini_access_token()
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
        
        data = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": 0.4,
                "topK": 40,
                "topP": 0.95,
                "maxOutputTokens": 1500,
            }
        }
        
        # Configure requests session with better SSL handling
        session = requests.Session()
        session.verify = True  # Enable SSL verification
        
        # Add retry mechanism for network issues
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        response = session.post(url, headers=headers, json=data, timeout=30)
        
        if not response.ok:
            error_data = response.json()
            raise Exception(f"Gemini API error: {error_data.get('error', {}).get('message', response.text)}")
        
        result = response.json()
        
        if not result.get("candidates") or not result["candidates"][0].get("content"):
            raise Exception("Invalid response format from Gemini API")
        
        content = result["candidates"][0]["content"]
        if "parts" in content and len(content["parts"]) > 0:
            return content["parts"][0]["text"]
        elif "text" in content:
            return content["text"]
        else:
            raise Exception("No text found in Gemini API response")
        
    except requests.exceptions.SSLError as ssl_error:
        logger.error(f"SSL Error calling Gemini API: {str(ssl_error)}")
        logger.error("This might be due to SSL certificate issues or network configuration")
        raise
    except requests.exceptions.ConnectionError as conn_error:
        logger.error(f"Connection Error calling Gemini API: {str(conn_error)}")
        logger.error("This might be due to network connectivity issues or firewall settings")
        raise
    except requests.exceptions.Timeout as timeout_error:
        logger.error(f"Timeout Error calling Gemini API: {str(timeout_error)}")
        logger.error("The request timed out. Please check your network connection")
        raise
    except Exception as e:
        logger.error(f"Error calling Gemini API: {str(e)}")
        raise

async def call_gemini_api_async(prompt):
    """Async wrapper for Gemini API call"""
    try:
        # Use asyncio.to_thread for Python 3.9+ or loop.run_in_executor for older versions
        try:
            return await asyncio.to_thread(call_gemini_api, prompt)
        except AttributeError:
            # Fallback for older Python versions
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, call_gemini_api, prompt)
    except Exception as e:
        logger.error(f"Error in async Gemini API call: {str(e)}")
        raise
