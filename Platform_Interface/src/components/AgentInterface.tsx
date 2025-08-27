import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Textarea } from './ui/textarea';
import { Badge } from './ui/badge';
import { Avatar, AvatarFallback, AvatarImage } from './ui/avatar';
import { ScrollArea } from './ui/scroll-area';
import { Separator } from './ui/separator';
import {
  Send,
  Paperclip,
  Mic,
  MicOff,
  Download,
  MoreVertical,
  FileText,
  Image,
  Play,
  Pause,
  RefreshCw
} from 'lucide-react';

interface Message {
  id: number;
  sender: 'user' | 'agent';
  content: string;
  timestamp: string;
  type: 'text' | 'file' | 'code';
  attachments?: { name: string; size: string; type: string }[];
}

interface AgentInterfaceProps {
  activeAgent: string | null;
}

const agents = {
  cto: { name: 'Alex Chen', role: 'CTO Agent', status: 'online', avatar: '', initials: 'AC' },
  ds: { name: 'Sarah Kim', role: 'Data Scientist', status: 'online', avatar: '', initials: 'SK' },
  ml: { name: 'Marcus Rodriguez', role: 'ML Engineer', status: 'busy', avatar: '', initials: 'MR' },
  bi: { name: 'Emma Thompson', role: 'BI Analyst', status: 'online', avatar: '', initials: 'ET' },
  dev: { name: 'James Wilson', role: 'Dev Ops', status: 'offline', avatar: '', initials: 'JW' },
};

const mockMessages: Message[] = [
  {
    id: 1,
    sender: 'user',
    content: 'Hi Alex, I need help analyzing our customer churn data from the last quarter.',
    timestamp: '10:30 AM',
    type: 'text'
  },
  {
    id: 2,
    sender: 'agent',
    content: "Hello! I'd be happy to help you analyze the customer churn data. Could you please share the dataset or provide access to the database? I can perform a comprehensive analysis including churn rate trends, customer segments most at risk, and predictive modeling.",
    timestamp: '10:31 AM',
    type: 'text'
  },
  {
    id: 3,
    sender: 'user',
    content: 'Here is the data file from our CRM system.',
    timestamp: '10:32 AM',
    type: 'file',
    attachments: [{ name: 'customer_data_q4.csv', size: '2.4 MB', type: 'csv' }]
  },
  {
    id: 4,
    sender: 'agent',
    content: `Perfect! I've analyzed your customer data. Here are the key findings:

**Churn Rate Summary:**
- Overall churn rate: 12.3% (up 2.1% from Q3)
- High-risk segment: Enterprise customers with >6 months tenure
- Primary churn drivers: Pricing concerns (34%), Feature limitations (28%), Support issues (21%)

**Recommendations:**
1. Implement proactive outreach for enterprise customers showing early warning signs
2. Review pricing strategy for long-term customers
3. Enhance feature set based on feedback analysis

Would you like me to create a detailed visualization of these findings or dive deeper into any specific area?`,
    timestamp: '10:35 AM',
    type: 'text'
  }
];

export function AgentInterface({ activeAgent }: AgentInterfaceProps) {
  const [message, setMessage] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isTyping, setIsTyping] = useState(false);

  const agent = activeAgent ? agents[activeAgent as keyof typeof agents] : null;

  const handleSendMessage = () => {
    if (message.trim()) {
      // Add message logic here
      setMessage('');
      // Simulate agent typing
      setIsTyping(true);
      setTimeout(() => setIsTyping(false), 2000);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  if (!agent) {
    return (
      <div className="flex-1 flex items-center justify-center p-6">
        <div className="text-center">
          <div className="w-16 h-16 bg-primary/10 rounded-full flex items-center justify-center mx-auto mb-4">
            <Send className="w-8 h-8 text-primary" />
          </div>
          <h3 className="text-lg mb-2">Select an AI Agent</h3>
          <p className="text-muted-foreground">Choose an agent from the sidebar to start a conversation.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col h-full">
      {/* Agent Header */}
      <div className="p-4 border-b border-border bg-card">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Avatar className="w-10 h-10">
              <AvatarImage src={agent.avatar} />
              <AvatarFallback>{agent.initials}</AvatarFallback>
            </Avatar>
            <div>
              <h3 className="font-medium">{agent.name}</h3>
              <div className="flex items-center gap-2">
                <Badge variant={agent.status === 'online' ? 'default' : 'secondary'} className="text-xs">
                  {agent.status}
                </Badge>
                <span className="text-sm text-muted-foreground">{agent.role}</span>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm">
              <RefreshCw className="w-4 h-4" />
            </Button>
            <Button variant="ghost" size="sm">
              <MoreVertical className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </div>

      {/* Chat Messages */}
      <ScrollArea className="flex-1 p-4">
        <div className="space-y-4">
          {mockMessages.map((msg) => (
            <div key={msg.id} className={`flex ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div className={`max-w-[70%] ${msg.sender === 'user' ? 'order-2' : 'order-1'}`}>
                <div className={`p-3 rounded-lg ${
                  msg.sender === 'user' 
                    ? 'bg-primary text-primary-foreground' 
                    : 'bg-muted'
                }`}>
                  {msg.type === 'file' && msg.attachments && (
                    <div className="mb-2">
                      {msg.attachments.map((file, index) => (
                        <div key={index} className="flex items-center gap-2 p-2 bg-card rounded border">
                          <FileText className="w-4 h-4" />
                          <div className="flex-1">
                            <div className="text-sm font-medium">{file.name}</div>
                            <div className="text-xs text-muted-foreground">{file.size}</div>
                          </div>
                          <Button variant="ghost" size="sm">
                            <Download className="w-4 h-4" />
                          </Button>
                        </div>
                      ))}
                    </div>
                  )}
                  <p className="text-sm whitespace-pre-line">{msg.content}</p>
                </div>
                <div className={`text-xs text-muted-foreground mt-1 ${
                  msg.sender === 'user' ? 'text-right' : 'text-left'
                }`}>
                  {msg.timestamp}
                </div>
              </div>
              {msg.sender === 'agent' && (
                <Avatar className="w-8 h-8 order-1 mr-2 mt-1">
                  <AvatarImage src={agent.avatar} />
                  <AvatarFallback className="text-xs">{agent.initials}</AvatarFallback>
                </Avatar>
              )}
            </div>
          ))}
          
          {/* Typing Indicator */}
          {isTyping && (
            <div className="flex justify-start">
              <Avatar className="w-8 h-8 mr-2 mt-1">
                <AvatarImage src={agent.avatar} />
                <AvatarFallback className="text-xs">{agent.initials}</AvatarFallback>
              </Avatar>
              <div className="bg-muted p-3 rounded-lg">
                <div className="flex gap-1">
                  <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                  <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                  <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
              </div>
            </div>
          )}
        </div>
      </ScrollArea>

      {/* Message Input */}
      <div className="p-4 border-t border-border bg-card">
        <div className="flex items-end gap-3">
          <div className="flex-1">
            <Textarea
              placeholder="Type your message..."
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              className="min-h-[60px] max-h-32 resize-none"
              rows={2}
            />
          </div>
          <div className="flex flex-col gap-2">
            <Button variant="ghost" size="sm" className="p-2">
              <Paperclip className="w-4 h-4" />
            </Button>
            <Button 
              variant="ghost" 
              size="sm" 
              className="p-2"
              onClick={() => setIsRecording(!isRecording)}
            >
              {isRecording ? <MicOff className="w-4 h-4 text-red-500" /> : <Mic className="w-4 h-4" />}
            </Button>
            <Button 
              onClick={handleSendMessage}
              disabled={!message.trim()}
              className="p-2"
            >
              <Send className="w-4 h-4" />
            </Button>
          </div>
        </div>
        
        {isRecording && (
          <div className="mt-2 p-2 bg-red-50 border border-red-200 rounded-lg flex items-center gap-2">
            <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
            <span className="text-sm text-red-700">Recording... Click to stop</span>
          </div>
        )}
      </div>
    </div>
  );
}