import React, { useState, useRef, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Badge } from './ui/badge';
import { Send, Upload, Download, BarChart3, Brain } from 'lucide-react';

interface Message {
  id: string;
  type: 'user' | 'agent' | 'system';
  content: string;
  timestamp: Date;
  agentName?: string;
  metadata?: any;
}

interface Agent {
  id: string;
  name: string;
  description: string;
  status: 'online' | 'offline' | 'busy';
}

interface ChatInterfaceProps {
  selectedAgent?: Agent;
  onFileUpload?: (file: File) => void;
  onExportResults?: () => void;
}

export const ChatInterface: React.FC<ChatInterfaceProps> = ({ 
  selectedAgent, 
  onFileUpload, 
  onExportResults 
}) => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      type: 'system',
      content: 'Welcome to ScrollIntel! Select an AI agent and start asking questions about your data.',
      timestamp: new Date()
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputValue.trim() || !selectedAgent) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: inputValue,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Simulate API call to agent
      const response = await fetch('/api/agents/process', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          agent_id: selectedAgent.id,
          query: inputValue,
          context: {}
        })
      });

      const result = await response.json();

      const agentMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'agent',
        content: result.success ? result.result : result.error || 'Sorry, I encountered an error processing your request.',
        timestamp: new Date(),
        agentName: selectedAgent.name,
        metadata: result.metadata
      };

      setMessages(prev => [...prev, agentMessage]);
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'system',
        content: 'Connection error. Please try again.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && onFileUpload) {
      onFileUpload(file);
      
      const fileMessage: Message = {
        id: Date.now().toString(),
        type: 'system',
        content: `File uploaded: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`,
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, fileMessage]);
    }
  };

  const formatMessageContent = (content: string, metadata?: any) => {
    // If content is a JSON object, format it nicely
    try {
      if (typeof content === 'object') {
        return JSON.stringify(content, null, 2);
      }
      
      if (content.startsWith('{') || content.startsWith('[')) {
        const parsed = JSON.parse(content);
        return JSON.stringify(parsed, null, 2);
      }
    } catch (e) {
      // Not JSON, return as is
    }
    
    return content;
  };

  const getMessageIcon = (type: string, agentName?: string) => {
    switch (type) {
      case 'agent':
        return <Brain className="w-4 h-4 text-blue-500" />;
      case 'system':
        return <BarChart3 className="w-4 h-4 text-gray-500" />;
      default:
        return null;
    }
  };

  return (
    <Card className="h-full flex flex-col">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-xl">
            {selectedAgent ? `Chat with ${selectedAgent.name}` : 'ScrollIntel Chat'}
          </CardTitle>
          <div className="flex space-x-2">
            <Button
              size="sm"
              variant="outline"
              onClick={() => fileInputRef.current?.click()}
              disabled={!selectedAgent}
            >
              <Upload className="w-4 h-4 mr-2" />
              Upload
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={onExportResults}
              disabled={messages.length <= 1}
            >
              <Download className="w-4 h-4 mr-2" />
              Export
            </Button>
          </div>
        </div>
        
        {selectedAgent && (
          <div className="flex items-center space-x-2">
            <Badge variant="outline" className="text-xs">
              {selectedAgent.status === 'online' ? '🟢' : '🔴'} {selectedAgent.status}
            </Badge>
            <span className="text-xs text-gray-500">
              Avg response: {selectedAgent.status === 'online' ? '1.2s' : 'N/A'}
            </span>
          </div>
        )}
      </CardHeader>

      <CardContent className="flex-1 flex flex-col space-y-4">
        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto space-y-3 min-h-0">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[80%] rounded-lg p-3 ${
                  message.type === 'user'
                    ? 'bg-blue-500 text-white'
                    : message.type === 'agent'
                    ? 'bg-gray-100 text-gray-900'
                    : 'bg-yellow-50 text-yellow-800 border border-yellow-200'
                }`}
              >
                <div className="flex items-start space-x-2">
                  {getMessageIcon(message.type, message.agentName)}
                  <div className="flex-1">
                    {message.agentName && (
                      <p className="text-xs font-medium mb-1 opacity-75">
                        {message.agentName}
                      </p>
                    )}
                    <pre className="whitespace-pre-wrap text-sm font-sans">
                      {formatMessageContent(message.content, message.metadata)}
                    </pre>
                    <p className="text-xs opacity-60 mt-2">
                      {message.timestamp.toLocaleTimeString()}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          ))}
          
          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-gray-100 rounded-lg p-3 max-w-[80%]">
                <div className="flex items-center space-x-2">
                  <Brain className="w-4 h-4 text-blue-500 animate-pulse" />
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" />
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                  </div>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="border-t pt-4">
          {!selectedAgent ? (
            <div className="text-center text-gray-500 py-4">
              <Brain className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p>Select an AI agent to start chatting</p>
            </div>
          ) : (
            <div className="flex space-x-2">
              <Input
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder={`Ask ${selectedAgent.name} anything...`}
                disabled={isLoading || selectedAgent.status !== 'online'}
                className="flex-1"
              />
              <Button
                onClick={handleSendMessage}
                disabled={!inputValue.trim() || isLoading || selectedAgent.status !== 'online'}
                size="sm"
              >
                <Send className="w-4 h-4" />
              </Button>
            </div>
          )}
        </div>
      </CardContent>

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        onChange={handleFileUpload}
        accept=".csv,.xlsx,.json"
        className="hidden"
      />
    </Card>
  );
};

export default ChatInterface;