import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { PDFProvider } from './contexts/PDFContext'

export const API_BASE_URL = '/api/v1';

createRoot(document.getElementById('root')!).render(
    <PDFProvider>
      <App />
    </PDFProvider>
)
