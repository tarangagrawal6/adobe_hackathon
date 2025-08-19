import React, { useState } from 'react';
import { BookOpen, Lightbulb } from 'lucide-react';
import { cn } from '../lib/utils';
import HeadingsSidebar from './HeadingsSidebar';
import ChatbotSidebar from './ChatbotSidebar';

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

interface TabbedSidebarProps {
  filename: string;
  onHeadingClick: (heading: Heading) => void;
}

type TabType = 'headings' | 'chatbot';

const TabbedSidebar: React.FC<TabbedSidebarProps> = ({ filename, onHeadingClick }) => {
  const [activeTab, setActiveTab] = useState<TabType>('headings');

  const tabs = [
    {
      id: 'headings' as TabType,
      label: 'Headings',
      icon: BookOpen,
      description: 'Document structure'
    },
    {
      id: 'chatbot' as TabType,
      label: 'Insights Bulb',
      icon: Lightbulb,
      description: 'AI Assistant'
    }
  ];

  return (
    <div className="w-[480px] min-w-[480px] bg-white border-r border-gray-200 shadow-lg flex flex-col h-full overflow-hidden">
      {/* Tab Navigation */}
      <div className="flex border-b border-gray-200 bg-gradient-to-r from-gray-50 to-gray-100 min-h-[60px]">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={cn(
              "flex-1 flex flex-col items-center justify-center py-3 px-2 transition-all duration-200 relative",
              activeTab === tab.id
                ? "bg-white text-blue-600 shadow-sm"
                : "text-gray-600 hover:text-gray-900 hover:bg-gray-100"
            )}
          >
            <tab.icon className="h-4 w-4 mb-1" />
            <span className="text-xs font-medium leading-tight">{tab.label}</span>
            <span className="text-xs text-gray-500 leading-tight mt-0.5">{tab.description}</span>
            {activeTab === tab.id && (
              <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-blue-500" />
            )}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="flex-1 overflow-hidden min-h-0">
        {activeTab === 'headings' && (
          <HeadingsSidebar
            filename={filename}
            onHeadingClick={onHeadingClick}
          />
        )}
        {activeTab === 'chatbot' && (
          <ChatbotSidebar filename={filename} />
        )}
      </div>
    </div>
  );
};

export default TabbedSidebar;
