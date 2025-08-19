import React, { createContext, useContext, useState, useCallback } from 'react';

interface PDFData {
  filename: string;
  original_word_count: number;
  summary_word_count: number;
  text: string;
  is_summarized: boolean;
  processing_status: string;
}

interface PDFContextType {
  pdfData: Record<string, PDFData>;
  setPDFData: (filename: string, data: PDFData) => void;
  getPDFData: (filename: string) => PDFData | null;
  hasPDFData: (filename: string) => boolean;
  clearPDFData: (filename: string) => void;
}

const PDFContext = createContext<PDFContextType | undefined>(undefined);

export function PDFProvider({ children }: { children: React.ReactNode }) {
  const [pdfData, setPdfDataState] = useState<Record<string, PDFData>>({});

  const setPDFData = useCallback((filename: string, data: PDFData) => {
    setPdfDataState(prev => ({
      ...prev,
      [filename]: data
    }));
  }, []);

  const getPDFData = useCallback((filename: string) => {
    return pdfData[filename] || null;
  }, [pdfData]);

  const hasPDFData = useCallback((filename: string) => {
    return filename in pdfData;
  }, [pdfData]);

  const clearPDFData = useCallback((filename: string) => {
    setPdfDataState(prev => {
      const newState = { ...prev };
      delete newState[filename];
      return newState;
    });
  }, []);

  return (
    <PDFContext.Provider value={{ 
      pdfData, 
      setPDFData, 
      getPDFData, 
      hasPDFData, 
      clearPDFData 
    }}>
      {children}
    </PDFContext.Provider>
  );
}

export function usePDF() {
  const context = useContext(PDFContext);
  if (context === undefined) {
    throw new Error('usePDF must be used within a PDFProvider');
  }
  return context;
}
