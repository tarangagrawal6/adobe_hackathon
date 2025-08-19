import React, { useState, useEffect } from 'react';
import { ChevronRight, FileText, BookOpen, Search } from 'lucide-react';
import { Input } from './ui/input';
import { Card, CardContent } from './ui/card';
import { Badge } from './ui/badge';
import { ScrollArea } from './ui/scroll-area';
import { cn } from '../lib/utils';
import { API_BASE_URL } from '../main';

interface Heading {
  id: number;
  text: string;
  page: number;
  bbox: number[];
  y_position: number;
  x_position: number;
  confidence: number;
  content: string;
}

interface HeadingsData {
  filename: string;
  ordered_content: Heading[];
  summary: {
    total_items: number;
    headings: number;
    original_headings: number;
    filtered_out: number;
  };
}

interface HeadingsSidebarProps {
  filename: string;
  onHeadingClick: (heading: Heading) => void;
}

const HeadingsSidebar: React.FC<HeadingsSidebarProps> = ({
  filename,
  onHeadingClick
}) => {
  const [headingsData, setHeadingsData] = useState<HeadingsData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedHeadingId, setSelectedHeadingId] = useState<number | null>(null);

  // Fetch headings data when filename changes
  useEffect(() => {
    if (!filename) return;

    const fetchHeadings = async () => {
      setLoading(true);
      setError(null);
      
      try {
        // Extract base filename without extension
        const baseFilename = filename.replace(/\.pdf$/i, '');
        const response = await fetch(`${API_BASE_URL}/get-headings/${baseFilename}`);
        
        if (!response.ok) {
          throw new Error(`Failed to fetch headings: ${response.statusText}`);
        }
        
        const data = await response.json();
        setHeadingsData(data);
      } catch (err) {
        console.error('Error fetching headings:', err);
        setError(err instanceof Error ? err.message : 'Failed to fetch headings');
      } finally {
        setLoading(false);
      }
    };

    fetchHeadings();
  }, [filename]);

  // Filter headings based on search term
  const filteredHeadings = headingsData?.ordered_content.filter(heading =>
    heading.text.toLowerCase().includes(searchTerm.toLowerCase()) ||
    heading.content.toLowerCase().includes(searchTerm.toLowerCase())
  ) || [];

  const handleHeadingClick = (heading: Heading) => {
    setSelectedHeadingId(heading.id);
    onHeadingClick(heading);
  };

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="flex items-center p-4 border-b border-gray-200 bg-gradient-to-r from-green-50 to-emerald-50 min-h-[60px]">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-green-500 rounded-lg">
            <BookOpen className="h-5 w-5 text-white" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-gray-900">Document Headings</h2>
            {headingsData && (
              <p className="text-xs text-gray-600">
                {headingsData.summary.headings} headings found
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Search */}
      <div className="p-4 border-b border-gray-200 bg-white">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
          <Input
            placeholder="Search headings..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10 h-10 text-sm border-gray-300 focus:border-green-500 focus:ring-green-500"
          />
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-hidden min-h-0">
        {loading && (
          <div className="flex items-center justify-center h-full p-8">
            <div className="text-center">
              <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-green-600 mx-auto mb-4"></div>
              <p className="text-sm text-gray-600">Loading headings...</p>
              <p className="text-xs text-gray-500 mt-2">This may take a moment</p>
            </div>
          </div>
        )}

        {error && (
          <div className="p-4">
            <Card className="border-red-200 bg-red-50">
              <CardContent className="pt-4">
                <p className="text-sm text-red-700">{error}</p>
              </CardContent>
            </Card>
          </div>
        )}

        {headingsData && !loading && (
          <div className="h-full flex flex-col">
            {/* Headings List */}
            <div className="flex-1 min-h-0">
              <ScrollArea className="h-full">
                <div className="p-4 space-y-3">
                  {filteredHeadings.length === 0 ? (
                    <div className="text-center py-12">
                      <FileText className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                      <p className="text-sm text-gray-500 mb-2">
                        {searchTerm ? 'No headings match your search' : 'No headings found'}
                      </p>
                      {searchTerm && (
                        <p className="text-xs text-gray-400">Try adjusting your search terms</p>
                      )}
                    </div>
                  ) : (
                    filteredHeadings.map((heading) => (
                      <Card
                        key={heading.id}
                        className={cn(
                          "cursor-pointer transition-all duration-200 hover:shadow-md border-l-4",
                          selectedHeadingId === heading.id
                            ? "border-l-green-500 bg-green-50 shadow-md"
                            : "border-l-transparent hover:border-l-green-300 hover:bg-gray-50"
                        )}
                        onClick={() => handleHeadingClick(heading)}
                      >
                        <CardContent className="p-4">
                          <div className="flex items-start justify-between mb-3">
                            <h3 className="font-semibold text-gray-900 text-sm leading-tight flex-1 pr-2">
                              {heading.text}
                            </h3>
                            <Badge variant="outline" className="text-xs ml-2 flex-shrink-0 bg-green-50 text-green-700 border-green-200">
                              Pg {heading.page}
                            </Badge>
                          </div>
                          
                          <p className="text-xs text-gray-700 leading-5 mb-3 line-clamp-3">
                            {heading.content}
                          </p>
                          
                          <div className="flex items-center justify-between text-xs text-gray-500">
                            <span>ID: {heading.id}</span>
                            <div className="flex items-center space-x-1">
                              <ChevronRight className="h-3 w-3" />
                              <span>Click to navigate</span>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    ))
                  )}
                </div>
              </ScrollArea>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default HeadingsSidebar;
