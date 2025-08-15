"use client";

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { 
  MessageCircle, 
  FileText, 
  HelpCircle, 
  Send, 
  Search,
  Star,
  ThumbsUp,
  ThumbsDown,
  Clock,
  CheckCircle,
  AlertCircle
} from 'lucide-react';

interface SupportTicket {
  id: number;
  ticket_number: string;
  subject: string;
  description: string;
  status: 'open' | 'in_progress' | 'waiting_for_customer' | 'resolved' | 'closed';
  priority: 'low' | 'medium' | 'high' | 'urgent';
  category?: string;
  created_at: string;
  updated_at: string;
  resolved_at?: string;
}

interface KnowledgeBaseArticle {
  id: number;
  title: string;
  slug: string;
  content: string;
  summary?: string;
  category?: string;
  view_count: number;
  helpful_votes: number;
  unhelpful_votes: number;
  created_at: string;
}

interface FAQ {
  id: number;
  question: string;
  answer: string;
  category?: string;
  is_featured: boolean;
  view_count: number;
  helpful_votes: number;
  unhelpful_votes: number;
}

const SupportDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState('help');
  const [searchQuery, setSearchQuery] = useState('');
  const [tickets, setTickets] = useState<SupportTicket[]>([]);
  const [articles, setArticles] = useState<KnowledgeBaseArticle[]>([]);
  const [faqs, setFaqs] = useState<FAQ[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // New ticket form state
  const [newTicket, setNewTicket] = useState({
    email: '',
    subject: '',
    description: '',
    category: '',
    priority: 'medium' as const
  });

  // Contact form state
  const [contactForm, setContactForm] = useState({
    name: '',
    email: '',
    company: '',
    subject: '',
    message: '',
    phone: '',
    inquiry_type: ''
  });

  // Feedback form state
  const [feedbackForm, setFeedbackForm] = useState({
    feedback_type: 'general_feedback',
    title: '',
    description: '',
    rating: 5,
    page_url: window.location.href
  });

  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    setLoading(true);
    try {
      // Load featured FAQs and popular articles
      const [faqsRes, articlesRes] = await Promise.all([
        fetch('/api/support/kb/faqs?featured=true'),
        fetch('/api/support/kb/articles?limit=10')
      ]);

      if (faqsRes.ok) {
        const faqsData = await faqsRes.json();
        setFaqs(faqsData);
      }

      if (articlesRes.ok) {
        const articlesData = await articlesRes.json();
        setArticles(articlesData);
      }
    } catch (err) {
      setError('Failed to load help content');
    } finally {
      setLoading(false);
    }
  };

  const searchHelpContent = async () => {
    if (!searchQuery.trim()) return;

    setLoading(true);
    try {
      const response = await fetch(`/api/support/kb/search?q=${encodeURIComponent(searchQuery)}`);
      if (response.ok) {
        const results = await response.json();
        setArticles(results.articles || []);
        setFaqs(results.faqs || []);
      }
    } catch (err) {
      setError('Search failed');
    } finally {
      setLoading(false);
    }
  };

  const submitTicket = async () => {
    if (!newTicket.email || !newTicket.subject || !newTicket.description) {
      setError('Please fill in all required fields');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('/api/support/tickets', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(newTicket)
      });

      if (response.ok) {
        const ticket = await response.json();
        setNewTicket({ email: '', subject: '', description: '', category: '', priority: 'medium' });
        setError(null);
        alert(`Ticket created successfully! Ticket number: ${ticket.ticket_number}`);
      } else {
        throw new Error('Failed to create ticket');
      }
    } catch (err) {
      setError('Failed to create ticket');
    } finally {
      setLoading(false);
    }
  };

  const submitContactForm = async () => {
    if (!contactForm.name || !contactForm.email || !contactForm.subject || !contactForm.message) {
      setError('Please fill in all required fields');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('/api/support/contact', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(contactForm)
      });

      if (response.ok) {
        const result = await response.json();
        setContactForm({ name: '', email: '', company: '', subject: '', message: '', phone: '', inquiry_type: '' });
        setError(null);
        alert(`Message sent successfully! Ticket number: ${result.ticket_number}`);
      } else {
        throw new Error('Failed to send message');
      }
    } catch (err) {
      setError('Failed to send message');
    } finally {
      setLoading(false);
    }
  };

  const submitFeedback = async () => {
    if (!feedbackForm.description) {
      setError('Please provide feedback description');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('/api/support/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(feedbackForm)
      });

      if (response.ok) {
        setFeedbackForm({ feedback_type: 'general_feedback', title: '', description: '', rating: 5, page_url: window.location.href });
        setError(null);
        alert('Feedback submitted successfully!');
      } else {
        throw new Error('Failed to submit feedback');
      }
    } catch (err) {
      setError('Failed to submit feedback');
    } finally {
      setLoading(false);
    }
  };

  const voteHelpful = async (contentType: string, contentId: number, helpful: boolean) => {
    try {
      await fetch('/api/support/kb/vote', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ content_type: contentType, content_id: contentId, helpful })
      });
    } catch (err) {
      console.error('Failed to record vote');
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'open': return <AlertCircle className="h-4 w-4 text-red-500" />;
      case 'in_progress': return <Clock className="h-4 w-4 text-yellow-500" />;
      case 'resolved': return <CheckCircle className="h-4 w-4 text-green-500" />;
      default: return <AlertCircle className="h-4 w-4 text-gray-500" />;
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'urgent': return 'bg-red-100 text-red-800';
      case 'high': return 'bg-orange-100 text-orange-800';
      case 'medium': return 'bg-yellow-100 text-yellow-800';
      case 'low': return 'bg-green-100 text-green-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="container mx-auto p-6 max-w-6xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Support Center</h1>
        <p className="text-gray-600">Get help, find answers, and contact our support team</p>
      </div>

      {error && (
        <Alert className="mb-6">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
        <TabsList className="grid w-full grid-cols-5">
          <TabsTrigger value="help" className="flex items-center gap-2">
            <HelpCircle className="h-4 w-4" />
            Help Center
          </TabsTrigger>
          <TabsTrigger value="tickets" className="flex items-center gap-2">
            <MessageCircle className="h-4 w-4" />
            Support Tickets
          </TabsTrigger>
          <TabsTrigger value="contact" className="flex items-center gap-2">
            <Send className="h-4 w-4" />
            Contact Us
          </TabsTrigger>
          <TabsTrigger value="feedback" className="flex items-center gap-2">
            <Star className="h-4 w-4" />
            Feedback
          </TabsTrigger>
          <TabsTrigger value="chat" className="flex items-center gap-2">
            <MessageCircle className="h-4 w-4" />
            Live Chat
          </TabsTrigger>
        </TabsList>

        <TabsContent value="help" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Search Help Center</CardTitle>
              <CardDescription>Find answers to common questions and browse our knowledge base</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="flex gap-2 mb-6">
                <Input
                  placeholder="Search for help articles, FAQs, and guides..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && searchHelpContent()}
                />
                <Button onClick={searchHelpContent} disabled={loading}>
                  <Search className="h-4 w-4" />
                </Button>
              </div>

              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <h3 className="text-lg font-semibold mb-4">Frequently Asked Questions</h3>
                  <div className="space-y-4">
                    {faqs.map((faq) => (
                      <Card key={faq.id} className="p-4">
                        <h4 className="font-medium mb-2">{faq.question}</h4>
                        <p className="text-sm text-gray-600 mb-3">{faq.answer}</p>
                        <div className="flex items-center justify-between text-xs text-gray-500">
                          <span>{faq.view_count} views</span>
                          <div className="flex items-center gap-2">
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => voteHelpful('faq', faq.id, true)}
                            >
                              <ThumbsUp className="h-3 w-3" />
                              {faq.helpful_votes}
                            </Button>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => voteHelpful('faq', faq.id, false)}
                            >
                              <ThumbsDown className="h-3 w-3" />
                              {faq.unhelpful_votes}
                            </Button>
                          </div>
                        </div>
                      </Card>
                    ))}
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-semibold mb-4">Help Articles</h3>
                  <div className="space-y-4">
                    {articles.map((article) => (
                      <Card key={article.id} className="p-4">
                        <h4 className="font-medium mb-2">{article.title}</h4>
                        {article.summary && (
                          <p className="text-sm text-gray-600 mb-3">{article.summary}</p>
                        )}
                        <div className="flex items-center justify-between text-xs text-gray-500">
                          <div className="flex items-center gap-2">
                            {article.category && (
                              <Badge variant="secondary">{article.category}</Badge>
                            )}
                            <span>{article.view_count} views</span>
                          </div>
                          <div className="flex items-center gap-2">
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => voteHelpful('article', article.id, true)}
                            >
                              <ThumbsUp className="h-3 w-3" />
                              {article.helpful_votes}
                            </Button>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => voteHelpful('article', article.id, false)}
                            >
                              <ThumbsDown className="h-3 w-3" />
                              {article.unhelpful_votes}
                            </Button>
                          </div>
                        </div>
                      </Card>
                    ))}
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="tickets" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Create Support Ticket</CardTitle>
              <CardDescription>Submit a detailed support request for technical assistance</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium mb-2 block">Email Address *</label>
                  <Input
                    type="email"
                    placeholder="your@email.com"
                    value={newTicket.email}
                    onChange={(e) => setNewTicket({ ...newTicket, email: e.target.value })}
                  />
                </div>
                <div>
                  <label className="text-sm font-medium mb-2 block">Category</label>
                  <Select value={newTicket.category} onValueChange={(value) => setNewTicket({ ...newTicket, category: value })}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select category" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="technical_issue">Technical Issue</SelectItem>
                      <SelectItem value="billing_question">Billing Question</SelectItem>
                      <SelectItem value="feature_request">Feature Request</SelectItem>
                      <SelectItem value="account_access">Account Access</SelectItem>
                      <SelectItem value="data_processing">Data Processing</SelectItem>
                      <SelectItem value="api_integration">API Integration</SelectItem>
                      <SelectItem value="general_inquiry">General Inquiry</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium mb-2 block">Priority</label>
                  <Select value={newTicket.priority} onValueChange={(value: any) => setNewTicket({ ...newTicket, priority: value })}>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="low">Low</SelectItem>
                      <SelectItem value="medium">Medium</SelectItem>
                      <SelectItem value="high">High</SelectItem>
                      <SelectItem value="urgent">Urgent</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div>
                <label className="text-sm font-medium mb-2 block">Subject *</label>
                <Input
                  placeholder="Brief description of your issue"
                  value={newTicket.subject}
                  onChange={(e) => setNewTicket({ ...newTicket, subject: e.target.value })}
                />
              </div>

              <div>
                <label className="text-sm font-medium mb-2 block">Description *</label>
                <Textarea
                  placeholder="Please provide detailed information about your issue, including steps to reproduce, error messages, and any relevant context..."
                  rows={6}
                  value={newTicket.description}
                  onChange={(e) => setNewTicket({ ...newTicket, description: e.target.value })}
                />
              </div>

              <Button onClick={submitTicket} disabled={loading} className="w-full">
                {loading ? 'Creating Ticket...' : 'Create Support Ticket'}
              </Button>
            </CardContent>
          </Card>

          {tickets.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Your Support Tickets</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {tickets.map((ticket) => (
                    <div key={ticket.id} className="border rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          {getStatusIcon(ticket.status)}
                          <span className="font-medium">{ticket.ticket_number}</span>
                          <Badge className={getPriorityColor(ticket.priority)}>
                            {ticket.priority}
                          </Badge>
                        </div>
                        <span className="text-sm text-gray-500">
                          {new Date(ticket.created_at).toLocaleDateString()}
                        </span>
                      </div>
                      <h4 className="font-medium mb-1">{ticket.subject}</h4>
                      <p className="text-sm text-gray-600 mb-2">{ticket.description.substring(0, 150)}...</p>
                      <div className="flex items-center justify-between">
                        <Badge variant="outline">{ticket.status.replace('_', ' ')}</Badge>
                        <Button variant="outline" size="sm">View Details</Button>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="contact" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Contact Our Team</CardTitle>
              <CardDescription>Send us a message and we'll get back to you within 24 hours</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium mb-2 block">Name *</label>
                  <Input
                    placeholder="Your full name"
                    value={contactForm.name}
                    onChange={(e) => setContactForm({ ...contactForm, name: e.target.value })}
                  />
                </div>
                <div>
                  <label className="text-sm font-medium mb-2 block">Email *</label>
                  <Input
                    type="email"
                    placeholder="your@email.com"
                    value={contactForm.email}
                    onChange={(e) => setContactForm({ ...contactForm, email: e.target.value })}
                  />
                </div>
              </div>

              <div className="grid md:grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium mb-2 block">Company</label>
                  <Input
                    placeholder="Your company name"
                    value={contactForm.company}
                    onChange={(e) => setContactForm({ ...contactForm, company: e.target.value })}
                  />
                </div>
                <div>
                  <label className="text-sm font-medium mb-2 block">Phone</label>
                  <Input
                    placeholder="Your phone number"
                    value={contactForm.phone}
                    onChange={(e) => setContactForm({ ...contactForm, phone: e.target.value })}
                  />
                </div>
              </div>

              <div>
                <label className="text-sm font-medium mb-2 block">Inquiry Type</label>
                <Select value={contactForm.inquiry_type} onValueChange={(value) => setContactForm({ ...contactForm, inquiry_type: value })}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select inquiry type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="sales">Sales Inquiry</SelectItem>
                    <SelectItem value="partnership">Partnership</SelectItem>
                    <SelectItem value="demo">Request Demo</SelectItem>
                    <SelectItem value="enterprise">Enterprise Solutions</SelectItem>
                    <SelectItem value="media">Media/Press</SelectItem>
                    <SelectItem value="general">General Question</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="text-sm font-medium mb-2 block">Subject *</label>
                <Input
                  placeholder="What can we help you with?"
                  value={contactForm.subject}
                  onChange={(e) => setContactForm({ ...contactForm, subject: e.target.value })}
                />
              </div>

              <div>
                <label className="text-sm font-medium mb-2 block">Message *</label>
                <Textarea
                  placeholder="Tell us more about your inquiry..."
                  rows={6}
                  value={contactForm.message}
                  onChange={(e) => setContactForm({ ...contactForm, message: e.target.value })}
                />
              </div>

              <Button onClick={submitContactForm} disabled={loading} className="w-full">
                {loading ? 'Sending Message...' : 'Send Message'}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="feedback" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Share Your Feedback</CardTitle>
              <CardDescription>Help us improve ScrollIntel with your suggestions and feedback</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="text-sm font-medium mb-2 block">Feedback Type</label>
                <Select value={feedbackForm.feedback_type} onValueChange={(value) => setFeedbackForm({ ...feedbackForm, feedback_type: value })}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="bug_report">Bug Report</SelectItem>
                    <SelectItem value="feature_request">Feature Request</SelectItem>
                    <SelectItem value="general_feedback">General Feedback</SelectItem>
                    <SelectItem value="user_experience">User Experience</SelectItem>
                    <SelectItem value="performance_issue">Performance Issue</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="text-sm font-medium mb-2 block">Title</label>
                <Input
                  placeholder="Brief summary of your feedback"
                  value={feedbackForm.title}
                  onChange={(e) => setFeedbackForm({ ...feedbackForm, title: e.target.value })}
                />
              </div>

              <div>
                <label className="text-sm font-medium mb-2 block">Description *</label>
                <Textarea
                  placeholder="Please provide detailed feedback..."
                  rows={6}
                  value={feedbackForm.description}
                  onChange={(e) => setFeedbackForm({ ...feedbackForm, description: e.target.value })}
                />
              </div>

              <div>
                <label className="text-sm font-medium mb-2 block">Overall Rating</label>
                <div className="flex items-center gap-2">
                  {[1, 2, 3, 4, 5].map((rating) => (
                    <Button
                      key={rating}
                      variant={feedbackForm.rating >= rating ? "default" : "outline"}
                      size="sm"
                      onClick={() => setFeedbackForm({ ...feedbackForm, rating })}
                    >
                      <Star className="h-4 w-4" />
                    </Button>
                  ))}
                  <span className="ml-2 text-sm text-gray-600">{feedbackForm.rating}/5 stars</span>
                </div>
              </div>

              <Button onClick={submitFeedback} disabled={loading} className="w-full">
                {loading ? 'Submitting Feedback...' : 'Submit Feedback'}
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="chat" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Live Chat Support</CardTitle>
              <CardDescription>Chat with our support team in real-time</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="text-center py-8">
                <MessageCircle className="h-16 w-16 mx-auto mb-4 text-gray-400" />
                <h3 className="text-lg font-medium mb-2">Chat Support Available</h3>
                <p className="text-gray-600 mb-4">
                  Our support team is available Monday - Friday, 9 AM - 6 PM EST
                </p>
                <p className="text-sm text-gray-500 mb-6">
                  Estimated wait time: &lt; 2 minutes
                </p>
                <Button size="lg">
                  Start Live Chat
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default SupportDashboard;