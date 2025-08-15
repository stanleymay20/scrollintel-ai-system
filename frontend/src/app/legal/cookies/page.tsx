"use client";

import React, { useState, useEffect } from 'react';
import { Card } from '../../../components/ui/card';
import { Badge } from '../../../components/ui/badge';
import { Button } from '../../../components/ui/button';

interface LegalDocument {
  id: number;
  document_type: string;
  version: string;
  title: string;
  content: string;
  effective_date: string;
  metadata: any;
}

export default function CookiePolicyPage() {
  const [document, setDocument] = useState<LegalDocument | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadCookiePolicy();
  }, []);

  const loadCookiePolicy = async () => {
    try {
      const response = await fetch('/api/legal/cookie-policy');
      if (response.ok) {
        const data = await response.json();
        setDocument(data);
      } else {
        throw new Error('Failed to load cookie policy');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const manageCookiePreferences = () => {
    // Clear existing consent to show banner again
    localStorage.removeItem('cookie-consent');
    window.location.reload();
  };

  if (loading) {
    return (
      <div className="max-w-4xl mx-auto p-6">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded mb-4"></div>
          <div className="h-4 bg-gray-200 rounded mb-2"></div>
          <div className="h-4 bg-gray-200 rounded mb-2"></div>
          <div className="h-4 bg-gray-200 rounded mb-2"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-4xl mx-auto p-6">
        <Card className="p-6 border-red-200 bg-red-50">
          <h1 className="text-xl font-bold text-red-800 mb-2">Error Loading Cookie Policy</h1>
          <p className="text-red-600">{error}</p>
        </Card>
      </div>
    );
  }

  if (!document) {
    return (
      <div className="max-w-4xl mx-auto p-6">
        <Card className="p-6">
          <h1 className="text-2xl font-bold mb-4">Cookie Policy</h1>
          <p className="text-gray-600">Cookie policy document not found.</p>
        </Card>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="mb-6">
        <div className="flex items-center justify-between mb-4">
          <h1 className="text-3xl font-bold">{document.title}</h1>
          <div className="flex items-center gap-2">
            <Badge variant="outline">Version {document.version}</Badge>
            <Badge variant="secondary">
              Effective: {new Date(document.effective_date).toLocaleDateString()}
            </Badge>
          </div>
        </div>
        
        <p className="text-gray-600">
          Last updated: {new Date(document.effective_date).toLocaleDateString()}
        </p>
      </div>

      <Card className="p-8">
        <div 
          className="prose prose-lg max-w-none"
          dangerouslySetInnerHTML={{ __html: document.content }}
        />
      </Card>

      {/* Cookie Management Section */}
      <Card className="mt-8 p-6 bg-blue-50 border-blue-200">
        <h2 className="text-xl font-semibold text-blue-800 mb-4">Manage Your Cookie Preferences</h2>
        
        <div className="space-y-4">
          <p className="text-blue-700">
            You can control which cookies we use by updating your preferences at any time.
          </p>
          
          <div className="grid md:grid-cols-2 gap-4">
            <div className="p-4 bg-white rounded border">
              <h3 className="font-medium mb-2">🍪 Necessary Cookies</h3>
              <p className="text-sm text-gray-600 mb-2">
                Essential for the website to function properly
              </p>
              <Badge variant="secondary">Always Active</Badge>
            </div>
            
            <div className="p-4 bg-white rounded border">
              <h3 className="font-medium mb-2">📊 Analytics Cookies</h3>
              <p className="text-sm text-gray-600 mb-2">
                Help us understand how visitors use our website
              </p>
              <Badge variant="outline">Optional</Badge>
            </div>
            
            <div className="p-4 bg-white rounded border">
              <h3 className="font-medium mb-2">🎯 Marketing Cookies</h3>
              <p className="text-sm text-gray-600 mb-2">
                Used to deliver personalized advertisements
              </p>
              <Badge variant="outline">Optional</Badge>
            </div>
            
            <div className="p-4 bg-white rounded border">
              <h3 className="font-medium mb-2">⚙️ Preference Cookies</h3>
              <p className="text-sm text-gray-600 mb-2">
                Remember your settings for a better experience
              </p>
              <Badge variant="outline">Optional</Badge>
            </div>
          </div>
          
          <div className="pt-4">
            <Button 
              onClick={manageCookiePreferences}
              className="bg-blue-600 hover:bg-blue-700"
            >
              Update Cookie Preferences
            </Button>
          </div>
        </div>
      </Card>

      <div className="mt-8 p-4 bg-gray-50 rounded">
        <p className="text-sm text-gray-600">
          If you have any questions about our use of cookies, please contact us at{' '}
          <a href="mailto:privacy@scrollintel.com" className="text-blue-600 hover:underline">
            privacy@scrollintel.com
          </a>
        </p>
      </div>
    </div>
  );
}