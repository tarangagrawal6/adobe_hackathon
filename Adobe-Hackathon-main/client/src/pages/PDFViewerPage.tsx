import { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { FileText, Loader2, Mic } from 'lucide-react';
import { Button } from '../components/ui/button';
import { Card, CardContent } from '../components/ui/card';
import TabbedSidebar from '../components/TabbedSidebar';
import VoiceBot from '../components/VoiceBot';
import { API_BASE_URL } from '../main';

// Adobe PDF Embed API types
declare global {
  interface Window {
    AdobeDC: {
      View: new (config: { clientId: string; divId: string }) => {
        previewFile: (
          content: { content: { location: { url: string } }; metaData: { fileName: string } },
          options: {
            defaultViewMode?: string;
            showDownloadPDF?: boolean;
            showPrintPDF?: boolean;
            showLeftHandPanel?: boolean;
            showAnnotationTools?: boolean;
            enableFormFilling?: boolean;
          }
        ) => Promise<any>;  // Note: previewFile returns a Promise
      };
    };
  }
}

const PDFViewerPage = () => {
  const { filename } = useParams<{ filename: string }>();
  const navigate = useNavigate();

  // Enhanced error suppression for Adobe internal errors
  useEffect(() => {
    const originalError = console.error;
    const originalWarn = console.warn;
    const originalLog = console.log;

    // Suppress Adobe SDK internal errors
    console.error = (...args) => {
      const message = args[0]?.toString() || '';
      const shouldSuppress = [
        'safeSessionStorage',
        'GET_FEATURE_FLAG',
        'ERR_BLOCKED_BY_CLIENT',
        'mobx.array',
        'Error in Logging serverHandler',
        'Failed to fetch',
        'about:blank',
        'No callback registered by viewer',
        'Cannot redefine property',
        'dc-api.adobe.io',
        'typekit.net',
        'AdobeDCViewApp.js',
        'ViewSDKInterface.js',
        'dc-core.js',
        'bootstrap.js'
      ].some(keyword => message.includes(keyword));

      if (shouldSuppress) {
        return;
      }
      originalError.apply(console, args);
    };

    console.warn = (...args) => {
      const message = args[0]?.toString() || '';
      const shouldSuppress = [
        'mobx.array',
        'Attempt to read an array index',
        'preloaded using link preload',
        'Slow network is detected',
        'Fallback font will be used',
        'Adobe',
        'ViewSDK'
      ].some(keyword => message.includes(keyword));

      if (shouldSuppress) {
        return;
      }
      originalWarn.apply(console, args);
    };

    // Suppress excessive Adobe logging
    console.log = (...args) => {
      const message = args[0]?.toString() || '';
      const shouldSuppress = [
        'rendering worker created',
        'Adobe',
        'ViewSDK',
        'dc-core'
      ].some(keyword => message.includes(keyword));

      if (shouldSuppress) {
        return;
      }
      originalLog.apply(console, args);
    };

    return () => {
      console.error = originalError;
      console.warn = originalWarn;
      console.log = originalLog;
    };
  }, []);

  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pdfUrl, setPdfUrl] = useState<string | null>(null);
  const [adobeScriptLoaded, setAdobeScriptLoaded] = useState(false);
  const [initAttempts, setInitAttempts] = useState(0);
  const [documentContext, setDocumentContext] = useState<string>('');
  const [showVoiceBot, setShowVoiceBot] = useState(false);
  const [adobeClientId, setAdobeClientId] = useState<string | null>(null);
  const adobeViewRef = useRef<any>(null);
  const apisRef = useRef<any>(null);  // NEW: Ref to store the viewer APIs for navigation
  const containerRef = useRef<HTMLDivElement>(null);
  const scriptLoadedRef = useRef(false);

  // Validate Adobe Client ID format
  const isValidClientId = (clientId: string) => {
    return clientId && clientId.length === 32 && /^[a-f0-9]+$/i.test(clientId);
  };

  // Load Adobe PDF Embed API script with better error handling
  const loadAdobeScript = (): Promise<void> => {
    return new Promise((resolve, reject) => {
      if (scriptLoadedRef.current) {
        resolve();
        return;
      }

      // Check if script is already loaded
      const existingScript = document.querySelector('script[src*="view-sdk"]');
      if (existingScript) {
        scriptLoadedRef.current = true;
        setAdobeScriptLoaded(true);
        resolve();
        return;
      }

      const script = document.createElement('script');
      script.src = 'https://documentcloud.adobe.com/view-sdk/main.js';
      script.async = true;
      script.crossOrigin = 'anonymous';

      const timeout = setTimeout(() => {
        reject(new Error('Adobe script load timeout'));
      }, 10000);

      script.onload = () => {
        clearTimeout(timeout);
        // Wait for AdobeDC to be available
        const checkAdobeDC = () => {
          if (window.AdobeDC && window.AdobeDC.View) {
            scriptLoadedRef.current = true;
            setAdobeScriptLoaded(true);
            resolve();
          } else {
            setTimeout(checkAdobeDC, 100);
          }
        };
        checkAdobeDC();
      };

      script.onerror = () => {
        clearTimeout(timeout);
        // Try fallback URL
        const fallbackScript = document.createElement('script');
        fallbackScript.src = 'https://documentcloud.adobe.com/view-sdk/2.12/main.js';
        fallbackScript.async = true;
        fallbackScript.crossOrigin = 'anonymous';

        const fallbackTimeout = setTimeout(() => {
          reject(new Error('Adobe script load timeout (fallback)'));
        }, 10000);

        fallbackScript.onload = () => {
          clearTimeout(fallbackTimeout);
          const checkAdobeDC = () => {
            if (window.AdobeDC && window.AdobeDC.View) {
              scriptLoadedRef.current = true;
              setAdobeScriptLoaded(true);
              resolve();
            } else {
              setTimeout(checkAdobeDC, 100);
            }
          };
          checkAdobeDC();
        };

        fallbackScript.onerror = () => {
          clearTimeout(fallbackTimeout);
          reject(new Error('Failed to load Adobe PDF Embed API script from both URLs'));
        };

        document.head.appendChild(fallbackScript);
      };

      document.head.appendChild(script);
    });
  };

  // Prevent Adobe SDK from redefining properties
  useEffect(() => {
    // Store original Object.defineProperty
    const originalDefineProperty = Object.defineProperty;

    // Override Object.defineProperty to prevent redefinition of safeSessionStorage
    Object.defineProperty = function (obj: any, prop: PropertyKey, descriptor: PropertyDescriptor) {
      if (prop === 'safeSessionStorage' && obj === window) {
        // Skip redefinition of safeSessionStorage
        return obj;
      }
      return originalDefineProperty.call(Object, obj, prop, descriptor);
    } as typeof Object.defineProperty;

    // Prevent Adobe from accessing blocked resources
    const originalFetch = window.fetch;
    window.fetch = function (input: RequestInfo | URL, init?: RequestInit) {
      const url = typeof input === 'string' ? input : input.toString();
      if (url.includes('dc-api.adobe.io') || url.includes('typekit.net')) {
        // Return a rejected promise for blocked resources
        return Promise.reject(new Error('Resource blocked by client'));
      }
      return originalFetch.call(this, input, init);
    };

    return () => {
      Object.defineProperty = originalDefineProperty;
      window.fetch = originalFetch;
    };
  }, []);

  // Create a safe Adobe SDK wrapper
  const createSafeAdobeViewer = () => {
    if (!window.AdobeDC?.View) {
      throw new Error('AdobeDC.View not available');
    }

    try {
      // Create the viewer with error handling
      const viewer = new window.AdobeDC.View({
        clientId: adobeClientId!,
        divId: "adobe-dc-view",
      });

      // Wrap the previewFile method to handle errors
      const originalPreviewFile = viewer.previewFile;
      viewer.previewFile = function (content: any, options: any) {
        try {
          return originalPreviewFile.call(this, content, options);
        } catch (error) {
          console.error('Adobe previewFile error:', error);
          throw error;
        }
      };

      return viewer;
    } catch (error) {
      console.error('Error creating Adobe viewer:', error);
      throw error;
    }
  };

  // Fetch Adobe client ID from API
  useEffect(() => {
    const fetchAdobeClientId = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/v1/config/`);
        if (!response.ok) {
          throw new Error('Failed to fetch configuration');
        }
        const config = await response.json();
        setAdobeClientId(config.adobe_client_id);
      } catch (error) {
        console.error('Error fetching Adobe client ID:', error);
        setError('Failed to load Adobe configuration');
      }
    };

    fetchAdobeClientId();
  }, []);

  useEffect(() => {
    if (!filename) {
      setError('No filename provided');
      setIsLoading(false);
      return;
    }

    const loadResources = async () => {
      try {
        // Load PDF first
        const response = await fetch(`${API_BASE_URL}/get-pdf/${decodeURIComponent(filename!)}`);

        if (!response.ok) {
          throw new Error('Failed to load PDF');
        }

        const blob = await response.blob();

        // Verify it's actually a PDF
        if (blob.type !== 'application/pdf') {
          console.warn('Response is not a PDF:', blob.type);
          // Still try to use it, as some servers don't set the correct content type
        }

        const url = URL.createObjectURL(blob);
        setPdfUrl(url);

        // Load Adobe PDF Embed API script
        await loadAdobeScript();
      } catch (error) {
        console.error('Error loading resources:', error);
        setError('Failed to load PDF file or Adobe viewer');
      } finally {
        setIsLoading(false);
      }
    };

    loadResources();

    return () => {
      if (pdfUrl) {
        window.URL.revokeObjectURL(pdfUrl);
      }
    };
  }, [filename]);

  // Initialize Adobe viewer when script is loaded, PDF URL is available, and Adobe client ID is loaded
  useEffect(() => {
    if (adobeScriptLoaded && pdfUrl && adobeClientId && window.AdobeDC && containerRef.current) {
      // Add a delay to ensure everything is ready
      const timer = setTimeout(async () => {
        await initializeAdobeViewer();
      }, 500);

      return () => clearTimeout(timer);
    }
  }, [adobeScriptLoaded, pdfUrl, adobeClientId, initAttempts]);

  const initializeAdobeViewer = async () => {
    if (!window.AdobeDC || !pdfUrl || !containerRef.current) {
      console.error('AdobeDC, pdfUrl, or containerRef is missing');
      return;
    }

    if (!adobeClientId || !isValidClientId(adobeClientId)) {
      console.error('Invalid Adobe Client ID format');
      setError('Invalid Adobe Client ID configuration');
      return;
    }

    // Validate PDF URL
    if (!pdfUrl || !pdfUrl.startsWith('blob:')) {
      console.error('Invalid PDF URL:', pdfUrl);
      setError('Invalid PDF URL format');
      return;
    }

    // Test if the PDF URL is accessible
    try {
      const testResponse = await fetch(pdfUrl);
      if (!testResponse.ok) {
        console.error('PDF URL not accessible:', testResponse.status);
        setError('PDF file not accessible');
        return;
      }
    } catch (error) {
      console.error('Error testing PDF URL:', error);
      setError('PDF file not accessible');
      return;
    }

    // Additional validation for Adobe configuration
    if (!window.AdobeDC?.View) {
      console.error('AdobeDC.View not available');
      setError('Adobe PDF Embed API not properly loaded');
      return;
    }

    // Add timeout for initialization
    const initTimeout = setTimeout(() => {
      setError('PDF viewer initialization timed out');
    }, 15000); // 15 second timeout

    try {
      // Ensure the container div exists and is properly set up
      const containerDiv = document.getElementById("adobe-dc-view");
      if (!containerDiv) {
        console.error('Container div not found');
        setError('PDF viewer container not found');
        return;
      }

      // Ensure the container has proper dimensions
      const rect = containerDiv.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) {
        console.error('Container div has no dimensions:', rect);
        setError('PDF viewer container has no dimensions');
        return;
      }

      // Clear any existing content
      containerDiv.innerHTML = '';

      // Initialize AdobeDC.View with safe wrapper
      adobeViewRef.current = createSafeAdobeViewer();

      // Preview the PDF file with configuration to fit the entire PDF
      // CHANGED: Capture the previewFile promise
      const previewFilePromise = adobeViewRef.current.previewFile(
        {
          content: { location: { url: pdfUrl } },
          metaData: { fileName: decodeURIComponent(filename!) },
        },
        {
          defaultViewMode: "FIT_PAGE", // Changed from FIT_WIDTH to FIT_PAGE
          showDownloadPDF: true,
          showPrintPDF: true,
          showLeftHandPanel: false, // Disable to avoid feature flag errors
          showAnnotationTools: false, // Disable to avoid feature flag errors
          enableFormFilling: false,
          showZoomControl: true,
          showPageControls: true,
          showFullScreen: true,
        }
      );

      // NEW: Get the viewer APIs after previewFile resolves
      previewFilePromise.then((adobeViewer: any) => {
        adobeViewer.getAPIs().then((apis: any) => {
          apisRef.current = apis;
          console.log('Adobe viewer APIs ready for navigation');
        }).catch((err: Error) => {
          console.error('Error getting Adobe APIs:', err);
        });
      }).catch((err: Error) => {
        console.error('Error in previewFile:', err);
      });

      clearTimeout(initTimeout);
      setInitAttempts(0); // Reset attempts on success
    } catch (error) {
      clearTimeout(initTimeout);
      console.error('Error initializing Adobe PDF viewer:', error);

      // Retry up to 3 times
      if (initAttempts < 3) {
        setInitAttempts(prev => prev + 1);
      } else {
        setError('Failed to initialize PDF viewer after multiple attempts');
      }
    }
  };

  const handleRetry = async () => {
    setError(null);
    setInitAttempts(0);
    cleanupAdobeViewer(); // Clean up before retrying
    if (adobeScriptLoaded && pdfUrl && window.AdobeDC && containerRef.current) {
      await initializeAdobeViewer();
    }
  };

  // UPDATED: Handle heading click to navigate to specific location in PDF using APIs
  const handleHeadingClick = (heading: any) => {
    try {
      if (apisRef.current && heading) {
        // Validate and convert position values
        const pageNumber = parseInt(heading.page);
        const left = parseFloat(heading.x_position);
        const top = parseFloat(heading.y_position);

        // Check if values are valid numbers
        if (isNaN(pageNumber) || isNaN(left) || isNaN(top)) {
          console.error('Invalid position values:', { page: heading.page, x: heading.x_position, y: heading.y_position });
          return;
        }

        // Ensure values are within reasonable bounds
        if (pageNumber < 1 || left < 0 || top < 0) {
          console.error('Position values out of bounds:', { pageNumber, left, top });
          return;
        }

        console.log('Heading data:', heading);
        console.log('Attempting navigation with:', { pageNumber, left, top });

        // Debug: Log available API methods
        console.log('Available API methods:', Object.keys(apisRef.current));

        // Check if the API has the gotoLocation method
        if (!apisRef.current.gotoLocation) {
          console.error('gotoLocation method not available');
          return;
        }

        // Try navigating to just the page first - use only pageNumber
        apisRef.current.gotoLocation(pageNumber)
          .then(() => {
            console.log('Successfully navigated to heading:', heading);
          })
          .catch((err: Error) => {
            console.error('Navigation error:', err);
            console.error('Failed navigation attempt with:', { pageNumber, left, top });
          });
      } else {
      }
    } catch (error) {
      console.error('Error navigating to heading:', error);
    }
  };

  // Load document context for voice bot
  const loadDocumentContext = async () => {
    if (!filename) return;

    try {
      const response = await fetch(`${API_BASE_URL}/initiate-chatbot/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ filename: filename.replace('.pdf', '') }),
      });

      if (response.ok) {
        const data = await response.json();
        setDocumentContext(data.text || '');
        console.log('Document context loaded for voice bot');
      } else {
        console.error('Failed to load document context');
      }
    } catch (error) {
      console.error('Error loading document context:', error);
    }
  };

  // Cleanup function to properly dispose of Adobe viewer
  const cleanupAdobeViewer = () => {
    if (adobeViewRef.current) {
      try {
        // Clear the container
        const containerDiv = document.getElementById("adobe-dc-view");
        if (containerDiv) {
          containerDiv.innerHTML = '';
        }
        adobeViewRef.current = null;
        apisRef.current = null;  // NEW: Reset APIs ref on cleanup
      } catch (error) {
        console.error('Error cleaning up Adobe viewer:', error);
      }
    }
  };

  // Load document context when PDF is loaded
  useEffect(() => {
    if (filename && !documentContext) {
      console.log('Loading document context for voice bot...');
      loadDocumentContext();
    }
  }, [filename, documentContext]);

  // Handle window resize to ensure PDF viewer adapts
  useEffect(() => {
    const handleResize = () => {
      if (adobeViewRef.current && containerRef.current) {
        // Trigger a re-render of the PDF viewer
        const containerDiv = document.getElementById("adobe-dc-view");
        if (containerDiv) {
          // Force Adobe viewer to recalculate dimensions
          setTimeout(() => {
            if (adobeViewRef.current) {
              try {
                const refreshPromise = adobeViewRef.current.previewFile(
                  {
                    content: { location: { url: pdfUrl } },
                    metaData: { fileName: decodeURIComponent(filename!) },
                  },
                  {
                    defaultViewMode: "FIT_PAGE",
                    showDownloadPDF: true,
                    showPrintPDF: true,
                    showLeftHandPanel: false,
                    showAnnotationTools: false,
                    enableFormFilling: false,
                    showZoomControl: true,
                    showPageControls: true,
                    showFullScreen: true,
                  }
                );

                // NEW: Re-fetch APIs after refresh
                refreshPromise.then((adobeViewer: any) => {
                  adobeViewer.getAPIs().then((apis: any) => {
                    apisRef.current = apis;
                  });
                });
              } catch (error) {
                console.error('Error refreshing PDF viewer on resize:', error);
              }
            }
          }, 100);
        }
      }
    };

    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      cleanupAdobeViewer();
      if (pdfUrl) {
        window.URL.revokeObjectURL(pdfUrl);
      }
    };
  }, [pdfUrl, filename]);

  if (isLoading) {
    return (
      <div className="h-screen flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="h-12 w-12 animate-spin text-blue-500 mx-auto mb-4" />
          <p className="text-lg text-gray-600">Loading PDF...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-screen flex items-center justify-center">
        <Card className="w-full max-w-md">
          <CardContent className="text-center py-8">
            <FileText className="h-12 w-12 text-red-500 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-gray-900 mb-2">Error Loading PDF</h3>
            <p className="text-gray-600 mb-4">{error}</p>
            <div className="flex gap-2">
              <Button onClick={handleRetry} variant="outline" className="flex-1">
                Retry
              </Button>
              <Button onClick={() => navigate('/')} className="flex-1">
                Back to Dashboard
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 top-16 flex bg-white overflow-hidden">
      {/* Tabbed Sidebar */}
      <TabbedSidebar
        filename={filename || ''}
        onHeadingClick={handleHeadingClick}
      />

      {/* PDF Viewer - Takes remaining space */}
      <div
        id="adobe-dc-view"
        ref={containerRef}
        className="flex-1 h-full relative"
      >
        {/* Adobe PDF Embed API will render here */}
        {!adobeScriptLoaded && (
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <Loader2 className="h-8 w-8 animate-spin text-blue-500 mx-auto mb-2" />
              <p className="text-gray-600">Loading Adobe PDF viewer...</p>
            </div>
          </div>
        )}
      </div>

      {/* Voice Bot Toggle Button - Outside PDF viewer */}
      {!showVoiceBot && (
        <div className="fixed bottom-8 right-8 z-50">
          <Button
            onClick={() => {
              console.log('Voice bot button clicked, current state:', { showVoiceBot, documentContext: !!documentContext });
              if (!showVoiceBot) {
                loadDocumentContext();
              }
              setShowVoiceBot(!showVoiceBot);
            }}
            className="bg-blue-500 hover:bg-blue-600 text-white rounded-full px-4 py-2 shadow-lg border-2 border-white"
            title="Toggle Voice Bot"
          >
            <Mic className="w-4 h-4 mr-2" />
            Voice Bot
          </Button>
        </div>
      )}
      {/* Voice Bot Component */}
      {showVoiceBot && documentContext && (
        <VoiceBot
          documentContext={documentContext}
          onClose={() => setShowVoiceBot(false)}
        />
      )}
    </div>
  );
};

export default PDFViewerPage;