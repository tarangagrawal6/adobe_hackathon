import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Upload, FileText, Trash2, Eye, Download, Loader2, CheckCircle } from 'lucide-react';
import { Button } from '../components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card';
import { Input } from '../components/ui/input';
import { Progress } from '../components/ui/progress';
import { toast } from 'sonner';
import { formatFileSize, formatDate } from '../lib/utils';
import { API_BASE_URL } from '../main';

interface PDFFile {
  filename: string;
  size_bytes: number;
  created: number;
  modified: number;
}

interface UploadResponse {
  total_files: number;
  successful_processing: number;
  results: Array<{
    original_filename: string;
    processing_status: string;
    error_message?: string;
  }>;
}

const UploadPage = () => {
  const [files, setFiles] = useState<File[]>([]);
  const [existingPDFs, setExistingPDFs] = useState<PDFFile[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    fetchExistingPDFs();
  }, []);

  const fetchExistingPDFs = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/list-pdfs/`);
      if (response.ok) {
        const data = await response.json();
        setExistingPDFs(data.files || []);
      } else {
        toast.error("Failed to fetch existing PDFs");
      }
    } catch (error) {
      toast.error("Failed to connect to server");
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(event.target.files || []);
    const pdfFiles = selectedFiles.filter(file => file.type === 'application/pdf');
    
    if (pdfFiles.length !== selectedFiles.length) {
      toast.error("Only PDF files are allowed");
    }
    
    // Check for duplicate filenames
    const existingFilenames = existingPDFs.map(pdf => pdf.filename);
    const duplicateFiles = pdfFiles.filter(file => existingFilenames.includes(file.name));
    
    if (duplicateFiles.length > 0) {
      const duplicateNames = duplicateFiles.map(file => file.name).join(', ');
      toast.error(`File(s) already exist: ${duplicateNames}. Please use a different filename.`);
      // Filter out duplicate files
      const uniqueFiles = pdfFiles.filter(file => !existingFilenames.includes(file.name));
      setFiles(uniqueFiles);
      return;
    }
    
    setFiles(pdfFiles);
  };

  const handleUpload = async () => {
    if (files.length === 0) {
      toast.error("Please select at least one PDF file");
      return;
    }

    setIsUploading(true);
    setUploadProgress(0);

    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
    });

    try {
      const response = await fetch(`${API_BASE_URL}/process-pdf-headings/`, {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const data: UploadResponse = await response.json();
        setUploadProgress(100);
        
        toast.success(`Successfully processed ${data.successful_processing} out of ${data.total_files} files`);

        // Refresh the list of existing PDFs
        await fetchExistingPDFs();
        setFiles([]);
        
        // Show detailed results
        data.results.forEach(result => {
          if (result.processing_status === 'error') {
            toast.error(`Error processing ${result.original_filename}: ${result.error_message || 'Unknown error'}`);
          }
        });
      } else {
        const errorData = await response.json();
        toast.error(errorData.detail || 'Failed to upload files');
      }
    } catch (error) {
      toast.error("Network error occurred");
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
    }
  };

  const handleDeletePDF = async (filename: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/delete-pdf/${filename}`, {
        method: 'DELETE',
      });

      if (response.ok) {
        toast.success(`Successfully deleted ${filename}`);
        await fetchExistingPDFs();
      } else {
        toast.error("Failed to delete file");
      }
    } catch (error) {
      toast.error("Network error occurred");
    }
  };

  const handleViewPDF = (filename: string) => {
    navigate(`/pdf/${encodeURIComponent(filename)}`);
  };

  const handleDownloadPDF = async (filename: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/get-pdf/${filename}`);
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      } else {
        toast.error("Failed to download file");
      }
    } catch (error) {
      toast.error("Network error occurred");
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Processing Overlay */}
      {isUploading && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center">
          <div className="bg-white rounded-2xl p-8 max-w-md w-full mx-4 shadow-2xl">
            <div className="text-center space-y-6">
              <div className="relative">
                <div className="w-20 h-20 mx-auto bg-blue-100 rounded-full flex items-center justify-center">
                  <Loader2 className="h-10 w-10 text-blue-600 animate-spin" />
                </div>
                <div className="absolute -top-1 -right-1 w-6 h-6 bg-blue-600 rounded-full flex items-center justify-center">
                  <div className="w-3 h-3 bg-white rounded-full animate-pulse"></div>
                </div>
              </div>
              
              <div className="space-y-2">
                <h3 className="text-xl font-semibold text-gray-900">Processing Documents</h3>
                <p className="text-gray-600 text-sm leading-relaxed">
                  This might take a while depending on the size and complexity of your PDF files. 
                  Please don't close this window.
                </p>
              </div>

              <div className="space-y-3">
                <div className="flex justify-between text-sm text-gray-500">
                  <span>Processing files...</span>
                  <span>{uploadProgress}%</span>
                </div>
                <Progress value={uploadProgress} className="h-2" />
              </div>

              <div className="flex items-center justify-center gap-2 text-xs text-gray-500">
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                <span>AI is analyzing your documents</span>
              </div>
            </div>
          </div>
        </div>
      )}

      <div className="container mx-auto px-4 py-8 max-w-7xl">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-blue-100 rounded-2xl mb-6">
            <FileText className="h-8 w-8 text-blue-600" />
          </div>
          <h1 className="text-4xl font-bold text-gray-900 mb-3">PDF Processing Dashboard</h1>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto">
            Upload PDFs and extract headings using advanced AI technology. 
            Get instant insights from your documents.
          </p>
        </div>

        <div className="grid gap-8 lg:grid-cols-3">
          {/* Upload Section */}
          <div className="lg:col-span-1">
            <Card className="border-0 shadow-lg bg-white/80 backdrop-blur-sm">
              <CardHeader className="pb-4">
                <CardTitle className="flex items-center gap-3 text-xl">
                  <div className="p-2 bg-blue-100 rounded-lg">
                    <Upload className="h-5 w-5 text-blue-600" />
                  </div>
                  Upload PDFs
                </CardTitle>
                <CardDescription className="text-gray-600">
                  Select one or more PDF files to process and extract headings
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className={`relative border-2 border-dashed rounded-xl p-8 text-center transition-all duration-200 ${
                  isUploading 
                    ? 'border-gray-200 bg-gray-50 opacity-50 cursor-not-allowed' 
                    : 'border-blue-300 bg-blue-50/50 hover:border-blue-400 hover:bg-blue-50 cursor-pointer'
                }`}>
                  <Input
                    type="file"
                    multiple
                    accept=".pdf"
                    onChange={handleFileSelect}
                    className="hidden"
                    id="file-upload"
                    disabled={isUploading}
                  />
                  <label htmlFor="file-upload" className={`cursor-pointer ${isUploading ? 'pointer-events-none' : ''}`}>
                    <div className="mb-4">
                      <Upload className={`h-12 w-12 mx-auto ${isUploading ? 'text-gray-400' : 'text-blue-500'}`} />
                    </div>
                    <p className="text-lg font-semibold text-gray-900 mb-2">
                      {files.length > 0 ? `${files.length} file(s) selected` : 'Click to select PDF files'}
                    </p>
                    <p className="text-sm text-gray-500">or drag and drop PDF files here</p>
                  </label>
                </div>

                {files.length > 0 && (
                  <div className="space-y-3">
                    <h4 className="font-semibold text-gray-900 flex items-center gap-2">
                      <CheckCircle className="h-4 w-4 text-green-500" />
                      Selected files ({files.length})
                    </h4>
                    <div className="space-y-2 max-h-48 overflow-y-auto">
                      {files.map((file, index) => (
                        <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg border">
                          <div className="flex items-center gap-3 min-w-0 flex-1">
                            <FileText className="h-5 w-5 text-blue-500 flex-shrink-0" />
                            <div className="min-w-0 flex-1">
                              <p className="font-medium text-sm truncate">{file.name}</p>
                              <p className="text-xs text-gray-500">{formatFileSize(file.size)}</p>
                            </div>
                          </div>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => setFiles(files.filter((_, i) => i !== index))}
                            disabled={isUploading}
                            className="text-red-500 hover:text-red-700 hover:bg-red-50"
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <Button
                  onClick={handleUpload}
                  disabled={files.length === 0 || isUploading}
                  className="w-full h-12 text-base font-semibold bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 disabled:text-gray-500"
                  size="lg"
                >
                  {isUploading ? (
                    <>
                      <Loader2 className="h-5 w-5 mr-2 animate-spin" />
                      Processing...
                    </>
                  ) : (
                    <>
                      <Upload className="h-5 w-5 mr-2" />
                      Upload and Process PDFs
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>
          </div>

          {/* Existing PDFs Section */}
          <div className="lg:col-span-2">
            <Card className="border-0 shadow-lg bg-white/80 backdrop-blur-sm">
              <CardHeader className="pb-4">
                <CardTitle className="flex items-center gap-3 text-xl">
                  <div className="p-2 bg-green-100 rounded-lg">
                    <FileText className="h-5 w-5 text-green-600" />
                  </div>
                  Your Documents
                </CardTitle>
                <CardDescription className="text-gray-600">
                  View and manage your uploaded PDF files
                </CardDescription>
              </CardHeader>
              <CardContent>
                {isLoading ? (
                  <div className="text-center py-12">
                    <div className="inline-flex items-center justify-center w-12 h-12 bg-blue-100 rounded-full mb-4">
                      <Loader2 className="h-6 w-6 text-blue-600 animate-spin" />
                    </div>
                    <p className="text-gray-500 font-medium">Loading your documents...</p>
                  </div>
                ) : existingPDFs.length === 0 ? (
                  <div className="text-center py-12">
                    <div className="inline-flex items-center justify-center w-16 h-16 bg-gray-100 rounded-full mb-4">
                      <FileText className="h-8 w-8 text-gray-400" />
                    </div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-2">No documents yet</h3>
                    <p className="text-gray-500">Upload your first PDF to get started</p>
                  </div>
                ) : (
                  <div className="grid gap-4 sm:grid-cols-2">
                    {existingPDFs.map((pdf) => (
                      <Card key={pdf.filename} className="group hover:shadow-lg transition-all duration-200 border-0 bg-gradient-to-br from-white to-gray-50/50">
                        <CardContent className="p-6">
                          <div className="flex items-start justify-between mb-4">
                            <div className="flex items-center gap-3 flex-1 min-w-0">
                              <div className="p-2 bg-blue-100 rounded-lg group-hover:bg-blue-200 transition-colors">
                                <FileText className="h-5 w-5 text-blue-600" />
                              </div>
                              <div className="min-w-0 flex-1">
                                <p className="font-semibold text-gray-900 truncate">{pdf.filename}</p>
                                <p className="text-sm text-gray-500">{formatFileSize(pdf.size_bytes)}</p>
                              </div>
                            </div>
                          </div>
                          
                          <div className="space-y-2 mb-6">
                            <div className="flex justify-between text-sm">
                              <span className="text-gray-500">Uploaded:</span>
                              <span className="font-medium text-gray-900">{formatDate(pdf.created)}</span>
                            </div>
                          </div>

                          <div className="flex gap-2">
                            <Button
                              size="sm"
                              onClick={() => handleViewPDF(pdf.filename)}
                              className="flex-1 bg-blue-600 hover:bg-blue-700 text-white"
                            >
                              <Eye className="h-4 w-4 mr-2" />
                              View
                            </Button>
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => handleDownloadPDF(pdf.filename)}
                              className="border-gray-300 hover:bg-gray-50"
                            >
                              <Download className="h-4 w-4" />
                            </Button>
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => handleDeletePDF(pdf.filename)}
                              className="border-red-200 text-red-600 hover:bg-red-50 hover:border-red-300"
                            >
                              <Trash2 className="h-4 w-4" />
                            </Button>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
};

export default UploadPage;
