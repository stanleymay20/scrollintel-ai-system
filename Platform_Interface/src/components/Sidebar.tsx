import React from 'react';
import { Badge } from './ui/badge';
import { Button } from './ui/button';
import { ScrollArea } from './ui/scroll-area';
import { Separator } from './ui/separator';
import { Avatar, AvatarFallback, AvatarImage } from './ui/avatar';
import {
  Bot,
  Brain,
  Code,
  Database,
  BarChart3,
  Image,
  Settings,
  Users,
  Shield,
  FileText,
  Home,
  MessageSquare,
  Camera,
  Archive,
  ChevronDown,
  Circle
} from 'lucide-react';

interface Agent {
  id: string;
  name: string;
  role: string;
  status: 'online' | 'busy' | 'offline';
  avatar?: string;
  initials: string;
}

interface SidebarProps {
  isCollapsed: boolean;
  activeSection: string;
  activeAgent: string | null;
  onSectionChange: (section: string) => void;
  onAgentSelect: (agentId: string) => void;
}

const agents: Agent[] = [
  { id: 'cto', name: 'Alex Chen', role: 'CTO Agent', status: 'online', initials: 'AC' },
  { id: 'ds', name: 'Sarah Kim', role: 'Data Scientist', status: 'online', initials: 'SK' },
  { id: 'ml', name: 'Marcus Rodriguez', role: 'ML Engineer', status: 'busy', initials: 'MR' },
  { id: 'bi', name: 'Emma Thompson', role: 'BI Analyst', status: 'online', initials: 'ET' },
  { id: 'dev', name: 'James Wilson', role: 'Dev Ops', status: 'offline', initials: 'JW' },
];

const navigationItems = [
  { id: 'dashboard', label: 'Dashboard', icon: Home },
  { id: 'agents', label: 'AI Agents', icon: Bot },
  { id: 'analytics', label: 'Analytics', icon: BarChart3 },
  { id: 'studio', label: 'Visual Studio', icon: Camera },
  { id: 'prompts', label: 'Prompt Management', icon: MessageSquare },
];

const enterpriseItems = [
  { id: 'users', label: 'User Management', icon: Users },
  { id: 'security', label: 'Security', icon: Shield },
  { id: 'compliance', label: 'Compliance', icon: FileText },
  { id: 'settings', label: 'Settings', icon: Settings },
];

export function Sidebar({ isCollapsed, activeSection, activeAgent, onSectionChange, onAgentSelect }: SidebarProps) {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online': return 'bg-green-500';
      case 'busy': return 'bg-yellow-500';
      case 'offline': return 'bg-gray-400';
      default: return 'bg-gray-400';
    }
  };

  return (
    <div className={`${isCollapsed ? 'w-16' : 'w-64'} transition-all duration-300 bg-card border-r border-border flex flex-col h-full`}>
      {/* Logo Section */}
      <div className="p-4 border-b border-border">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-gradient-to-br from-blue-600 to-blue-800 rounded-lg flex items-center justify-center">
            <Brain className="w-5 h-5 text-white" />
          </div>
          {!isCollapsed && (
            <div>
              <h1 className="font-semibold text-lg">ScrollIntel</h1>
              <p className="text-xs text-muted-foreground">Enterprise AI Platform</p>
            </div>
          )}
        </div>
      </div>

      <ScrollArea className="flex-1">
        <div className="p-2">
          {/* Main Navigation */}
          <div className="space-y-1">
            {navigationItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeSection === item.id;
              return (
                <Button
                  key={item.id}
                  variant={isActive ? "secondary" : "ghost"}
                  className={`w-full justify-start gap-3 ${isCollapsed ? 'px-2' : ''}`}
                  onClick={() => onSectionChange(item.id)}
                >
                  <Icon className="w-4 h-4 shrink-0" />
                  {!isCollapsed && <span>{item.label}</span>}
                </Button>
              );
            })}
          </div>

          {/* AI Agents Section */}
          {!isCollapsed && (
            <>
              <Separator className="my-4" />
              <div className="space-y-2">
                <div className="flex items-center justify-between px-2">
                  <h3 className="text-sm text-muted-foreground uppercase tracking-wide">AI Agents</h3>
                  <Badge variant="secondary" className="text-xs">5</Badge>
                </div>
                <div className="space-y-1">
                  {agents.map((agent) => (
                    <Button
                      key={agent.id}
                      variant={activeAgent === agent.id ? "secondary" : "ghost"}
                      className="w-full justify-start gap-3 h-auto py-3"
                      onClick={() => onAgentSelect(agent.id)}
                    >
                      <div className="relative">
                        <Avatar className="w-8 h-8">
                          <AvatarImage src={agent.avatar} />
                          <AvatarFallback className="text-xs">{agent.initials}</AvatarFallback>
                        </Avatar>
                        <div className={`absolute -bottom-0.5 -right-0.5 w-3 h-3 rounded-full border-2 border-card ${getStatusColor(agent.status)}`} />
                      </div>
                      <div className="flex-1 text-left">
                        <div className="text-sm">{agent.name}</div>
                        <div className="text-xs text-muted-foreground">{agent.role}</div>
                      </div>
                    </Button>
                  ))}
                </div>
              </div>
            </>
          )}

          {/* Enterprise Section */}
          <Separator className="my-4" />
          {!isCollapsed && (
            <div className="px-2 mb-2">
              <h3 className="text-sm text-muted-foreground uppercase tracking-wide">Enterprise</h3>
            </div>
          )}
          <div className="space-y-1">
            {enterpriseItems.map((item) => {
              const Icon = item.icon;
              const isActive = activeSection === item.id;
              return (
                <Button
                  key={item.id}
                  variant={isActive ? "secondary" : "ghost"}
                  className={`w-full justify-start gap-3 ${isCollapsed ? 'px-2' : ''}`}
                  onClick={() => onSectionChange(item.id)}
                >
                  <Icon className="w-4 h-4 shrink-0" />
                  {!isCollapsed && <span>{item.label}</span>}
                </Button>
              );
            })}
          </div>
        </div>
      </ScrollArea>
    </div>
  );
}