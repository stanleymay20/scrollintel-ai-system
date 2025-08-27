import React, { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { Header } from './components/Header';
import { Dashboard } from './components/Dashboard';
import { AgentInterface } from './components/AgentInterface';
import { AnalyticsDashboard } from './components/AnalyticsDashboard';
import { VisualStudio } from './components/VisualStudio';

export default function App() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [activeSection, setActiveSection] = useState('dashboard');
  const [activeAgent, setActiveAgent] = useState<string | null>(null);
  const [isDarkMode, setIsDarkMode] = useState(false);

  // Theme management
  useEffect(() => {
    const savedTheme = localStorage.getItem('theme');
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const shouldUseDark = savedTheme === 'dark' || (!savedTheme && prefersDark);
    
    setIsDarkMode(shouldUseDark);
    if (shouldUseDark) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, []);

  const toggleTheme = () => {
    const newTheme = !isDarkMode;
    setIsDarkMode(newTheme);
    localStorage.setItem('theme', newTheme ? 'dark' : 'light');
    
    if (newTheme) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  };

  const renderMainContent = () => {
    switch (activeSection) {
      case 'dashboard':
        return <Dashboard />;
      case 'agents':
        return <AgentInterface activeAgent={activeAgent} />;
      case 'analytics':
        return <AnalyticsDashboard />;
      case 'studio':
        return <VisualStudio />;
      case 'prompts':
        return (
          <div className="p-6 text-center">
            <h2 className="text-2xl mb-4">Prompt Management</h2>
            <p className="text-muted-foreground">Optimize and manage your AI prompts for better results.</p>
          </div>
        );
      case 'users':
        return (
          <div className="p-6 text-center">
            <h2 className="text-2xl mb-4">User Management</h2>
            <p className="text-muted-foreground">Manage user accounts, roles, and permissions.</p>
          </div>
        );
      case 'security':
        return (
          <div className="p-6 text-center">
            <h2 className="text-2xl mb-4">Security Dashboard</h2>
            <p className="text-muted-foreground">Monitor security threats and system vulnerabilities.</p>
          </div>
        );
      case 'compliance':
        return (
          <div className="p-6 text-center">
            <h2 className="text-2xl mb-4">Compliance Center</h2>
            <p className="text-muted-foreground">Track compliance requirements and audit trails.</p>
          </div>
        );
      case 'settings':
        return (
          <div className="p-6 text-center">
            <h2 className="text-2xl mb-4">System Settings</h2>
            <p className="text-muted-foreground">Configure platform settings and preferences.</p>
          </div>
        );
      default:
        return <Dashboard />;
    }
  };

  return (
    <div className="h-screen flex bg-background text-foreground">
      {/* Sidebar */}
      <Sidebar
        isCollapsed={sidebarCollapsed}
        activeSection={activeSection}
        activeAgent={activeAgent}
        onSectionChange={(section) => {
          setActiveSection(section);
          if (section !== 'agents') {
            setActiveAgent(null);
          }
        }}
        onAgentSelect={(agentId) => {
          setActiveAgent(agentId);
          setActiveSection('agents');
        }}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <Header
          isCollapsed={sidebarCollapsed}
          isDarkMode={isDarkMode}
          onToggleSidebar={() => setSidebarCollapsed(!sidebarCollapsed)}
          onToggleTheme={toggleTheme}
        />

        {/* Main Content */}
        <main className="flex-1 overflow-auto bg-background">
          {renderMainContent()}
        </main>
      </div>
    </div>
  );
}