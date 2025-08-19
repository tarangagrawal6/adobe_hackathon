import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Toaster } from './components/ui/sonner';
import Navbar from '@/components/Navbar';
import UploadPage from '@/pages/UploadPage';
import PDFViewerPage from '@/pages/PDFViewerPage';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        <Navbar />
        <Routes>
          <Route path="/" element={
            <main className="container mx-auto px-4 py-8">
              <UploadPage />
            </main>
          } />
          <Route path="/pdf/:filename" element={<PDFViewerPage />} />
        </Routes>
        <Toaster />
      </div>
    </Router>
  );
}

export default App;
