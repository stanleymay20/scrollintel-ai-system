"use client";

import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { 
  MessageCircle, 
  Send, 
  X, 
  Minimize2, 
  Maximize2,
  Phone,
  Mail,
  Clock,
  User,
  Bot,
  Paperclip,
  Smile,
  MoreVertical
} from 'lucide-react';

interface ChatMessage {
  id: string;
  content: string;
  sender: 'user' | 'agent' | 'system';
  timestamp: Date;
  senderName?: string;
  senderAvatar?: string;
  attachments?: Array<{
    name: string;
    url: string;
    type: string;
  }>;
}

interface ChatAgent {
  id: string;
  name: string;
  avatar: string;
  status: 'online' | 'away' | 'busy';
  title: string;
}

interface ChatSession {
  id: string;
  status: 'waiting' | 'connected' | 'ended';
  agent?: ChatAgent;
  queuePosition?: number;
  estimatedWaitTime?: string;
}

const LiveChat: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [newMessage, setNewMessage] = useState('');
  const [session, setSession] = useState<ChatSession | null>(null);
  const [isTyping, setIsTyping] = useState(false);
  const [userInfo, setUserInfo] = useState({
    name: '',
    email: '',
    subject: ''
  });
  const [showUserForm, setShowUserForm] = useState(true);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (isOpen && !isMinimized && chatInputRef.current) {
      chatInputRef.current.focus();
    }
  }, [isOpen, isMinimized]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const startChat = async () => {
    if (!userInfo.name || !userInfo.email) {
      return;
    }

    setShowUserForm(false);
    
    // Initialize chat session
    const newSession: ChatSession = {
      id: `chat_${Date.now()}`,
      status: 'waiting',
      queuePosition: 2,
      estimatedWaitTime: '< 2 minutes'
    };
    
    setSession(newSession);
    
    // Add welcome message
    const welcomeMessage: ChatMessage = {
      id: `msg_${Date.now()}`,
      content: `Hi ${userInfo.name}! Thanks for contacting ScrollIntel support. You're currently #${newSession.queuePosition} in queue. Estimated wait time: ${newSession.estimatedWaitTime}`,
      sender: 'system',
      timestamp: new Date()
    };
    
    setMessages([welcomeMessage]);
    
    // Simulate agent connection after delay
    setTimeout(() => {
      connectToAgent();
    }, 3000);
  };

  const connectToAgent = () => {
    const agent: ChatAgent = {
      id: 'agent_1',
      name: 'Sarah Chen',
      avatar: '/api/placeholder/32/32',
      status: 'online',
      title: 'Senior Support Specialist'
    };

    setSession(prev => prev ? {
      ...prev,
      status: 'connected',
      agent
    } : null);

    const agentMessage: ChatMessage = {
      id: `msg_${Date.now()}`,
      content: `Hi ${userInfo.name}! I'm Sarah from ScrollIntel support. I see you need help with "${userInfo.subject}". How can I assist you today?`,
      sender: 'agent',
      senderName: agent.name,
      senderAvatar: agent.avatar,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, agentMessage]);
  };

  const sendMessage = async () => {
    if (!newMessage.trim() || !session) return;

    const userMessage: ChatMessage = {
      id: `msg_${Date.now()}`,
      content: newMessage,
      sender: 'user',
      senderName: userInfo.name,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setNewMessage('');

    // Show typing indicator
    setIsTyping(true);

    // Simulate agent response
    setTimeout(() => {
      setIsTyping(false);
      
      const agentResponse: ChatMessage = {
        id: `msg_${Date.now() + 1}`,
        content: getAgentResponse(newMessage),
        sender: 'agent',
        senderName: session.agent?.name || 'Support Agent',
        senderAvatar: session.agent?.avatar,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, agentResponse]);
    }, 1500);
  };

  const getAgentResponse = (userMessage: string): string => {
    const message = userMessage.toLowerCase();
    
    if (message.includes('upload') || message.includes('file')) {
      return "I can help you with file uploads! ScrollIntel supports CSV, Excel, JSON, and SQL files up to 100MB. You can drag and drop files or use the file browser. What specific issue are you experiencing with uploads?";
    }
    
    if (message.includes('api') || message.includes('integration')) {
      return "For API integration, you'll need to get your API key from Settings > API Keys. Our API documentation is available at docs.scrollintel.com. Are you looking to integrate with a specific platform or language?";
    }
    
    if (message.includes('billing') || message.includes('subscription')) {
      return "I can help with billing questions! You can view your current plan and usage in Settings > Billing. Would you like me to check your account status or help with plan changes?";
    }
    
    if (message.includes('agent') || message.includes('cto') || message.includes('data scientist')) {
      return "Our AI agents are designed to replace human experts. The CTO Agent handles strategy, Data Scientist does analysis, ML Engineer builds models, and BI Agent creates dashboards. Which agent are you having trouble with?";
    }
    
    return "Thanks for that information! Let me look into this for you. Can you provide more details about what you're trying to accomplish? I'm here to help you get the most out of ScrollIntel.";
  };

  const endChat = () => {
    if (session) {
      setSession({ ...session, status: 'ended' });
      
      const endMessage: ChatMessage = {
        id: `msg_${Date.now()}`,
        content: "Chat session ended. Thank you for contacting ScrollIntel support! You'll receive a transcript of this conversation via email.",
        sender: 'system',
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, endMessage]);
    }
  };

  const resetChat = () => {
    setMessages([]);
    setSession(null);
    setShowUserForm(true);
    setUserInfo({ name: '', email: '', subject: '' });
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online': return 'bg-green-500';
      case 'away': return 'bg-yellow-500';
      case 'busy': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  if (!isOpen) {
    return (
      <div className="fixed bottom-6 right-6 z-50">
        <Button
          onClick={() => setIsOpen(true)}
          size="lg"
          className="rounded-full shadow-lg hover:shadow-xl transition-shadow"
        >
          <MessageCircle className="h-6 w-6 mr-2" />
          Chat Support
        </Button>
      </div>
    );
  }

  return (
    <div className="fixed bottom-6 right-6 z-50">
      <Card className={`w-96 shadow-xl transition-all duration-300 ${isMinimized ? 'h-16' : 'h-[600px]'}`}>
        {/* Chat Header */}
        <CardHeader className="p-4 border-b bg-blue-600 text-white rounded-t-lg">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="relative">
                <Avatar className="h-8 w-8">
                  <AvatarImage src={session?.agent?.avatar} />
                  <AvatarFallback>
                    {session?.agent ? session.agent.name.charAt(0) : <Bot className="h-4 w-4" />}
                  </AvatarFallback>
                </Avatar>
                {session?.agent && (
                  <div className={`absolute -bottom-1 -right-1 w-3 h-3 rounded-full border-2 border-white ${getStatusColor(session.agent.status)}`} />
                )}
              </div>
              <div>
                <h3 className="font-medium text-sm">
                  {session?.agent ? session.agent.name : 'ScrollIntel Support'}
                </h3>
                <p className="text-xs opacity-90">
                  {session?.status === 'connected' && session.agent ? 
                    session.agent.title : 
                    session?.status === 'waiting' ? 
                    'Connecting...' : 
                    'Live Chat Support'
                  }
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setIsMinimized(!isMinimized)}
                className="text-white hover:bg-blue-700 p-1"
              >
                {isMinimized ? <Maximize2 className="h-4 w-4" /> : <Minimize2 className="h-4 w-4" />}
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setIsOpen(false)}
                className="text-white hover:bg-blue-700 p-1"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </CardHeader>

        {!isMinimized && (
          <>
            <CardContent className="p-0 flex flex-col h-[calc(600px-80px)]">
              {/* User Info Form */}
              {showUserForm && (
                <div className="p-4 border-b bg-gray-50">
                  <h4 className="font-medium mb-3">Start a conversation</h4>
                  <div className="space-y-3">
                    <Input
                      placeholder="Your name"
                      value={userInfo.name}
                      onChange={(e) => setUserInfo({ ...userInfo, name: e.target.value })}
                    />
                    <Input
                      type="email"
                      placeholder="Your email"
                      value={userInfo.email}
                      onChange={(e) => setUserInfo({ ...userInfo, email: e.target.value })}
                    />
                    <Input
                      placeholder="What can we help you with?"
                      value={userInfo.subject}
                      onChange={(e) => setUserInfo({ ...userInfo, subject: e.target.value })}
                    />
                    <Button 
                      onClick={startChat} 
                      className="w-full"
                      disabled={!userInfo.name || !userInfo.email}
                    >
                      Start Chat
                    </Button>
                  </div>
                </div>
              )}

              {/* Chat Messages */}
              {!showUserForm && (
                <>
                  <div className="flex-1 overflow-y-auto p-4 space-y-4">
                    {session?.status === 'waiting' && (
                      <div className="text-center py-4">
                        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-2"></div>
                        <p className="text-sm text-gray-600">
                          Position in queue: #{session.queuePosition}
                        </p>
                        <p className="text-xs text-gray-500">
                          Estimated wait: {session.estimatedWaitTime}
                        </p>
                      </div>
                    )}

                    {messages.map((message) => (
                      <div
                        key={message.id}
                        className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
                      >
                        <div className={`flex gap-2 max-w-[80%] ${message.sender === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                          {message.sender !== 'user' && message.sender !== 'system' && (
                            <Avatar className="h-6 w-6 mt-1">
                              <AvatarImage src={message.senderAvatar} />
                              <AvatarFallback className="text-xs">
                                {message.sender === 'agent' ? <User className="h-3 w-3" /> : <Bot className="h-3 w-3" />}
                              </AvatarFallback>
                            </Avatar>
                          )}
                          <div>
                            <div
                              className={`rounded-lg px-3 py-2 ${
                                message.sender === 'user'
                                  ? 'bg-blue-600 text-white'
                                  : message.sender === 'system'
                                  ? 'bg-gray-100 text-gray-700 text-center'
                                  : 'bg-gray-100 text-gray-900'
                              }`}
                            >
                              <p className="text-sm">{message.content}</p>
                            </div>
                            <p className="text-xs text-gray-500 mt-1 px-1">
                              {formatTime(message.timestamp)}
                            </p>
                          </div>
                        </div>
                      </div>
                    ))}

                    {isTyping && (
                      <div className="flex justify-start">
                        <div className="flex gap-2 max-w-[80%]">
                          <Avatar className="h-6 w-6 mt-1">
                            <AvatarImage src={session?.agent?.avatar} />
                            <AvatarFallback className="text-xs">
                              <User className="h-3 w-3" />
                            </AvatarFallback>
                          </Avatar>
                          <div className="bg-gray-100 rounded-lg px-3 py-2">
                            <div className="flex space-x-1">
                              <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                              <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                              <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}

                    <div ref={messagesEndRef} />
                  </div>

                  {/* Chat Input */}
                  <div className="p-4 border-t">
                    {session?.status === 'connected' && (
                      <div className="flex gap-2">
                        <div className="flex-1 relative">
                          <Input
                            ref={chatInputRef}
                            placeholder="Type your message..."
                            value={newMessage}
                            onChange={(e) => setNewMessage(e.target.value)}
                            onKeyPress={handleKeyPress}
                            className="pr-20"
                          />
                          <div className="absolute right-2 top-1/2 transform -translate-y-1/2 flex gap-1">
                            <Button variant="ghost" size="sm" className="p-1">
                              <Paperclip className="h-4 w-4" />
                            </Button>
                            <Button variant="ghost" size="sm" className="p-1">
                              <Smile className="h-4 w-4" />
                            </Button>
                          </div>
                        </div>
                        <Button onClick={sendMessage} disabled={!newMessage.trim()}>
                          <Send className="h-4 w-4" />
                        </Button>
                      </div>
                    )}

                    {session?.status === 'ended' && (
                      <div className="text-center space-y-2">
                        <p className="text-sm text-gray-600">Chat session ended</p>
                        <Button onClick={resetChat} size="sm">
                          Start New Chat
                        </Button>
                      </div>
                    )}

                    {session?.status === 'connected' && (
                      <div className="flex justify-between items-center mt-2 text-xs text-gray-500">
                        <span>Connected to {session.agent?.name}</span>
                        <Button variant="ghost" size="sm" onClick={endChat} className="text-xs">
                          End Chat
                        </Button>
                      </div>
                    )}
                  </div>
                </>
              )}
            </CardContent>
          </>
        )}
      </Card>
    </div>
  );
};

export default LiveChat;