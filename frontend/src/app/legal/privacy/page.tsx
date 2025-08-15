"use client";

import React, { useState, useEffect } from 'react';
import { Card } from '../../../components/ui/card';
import { Badge } from '../../../components/ui/badge';

interface LegalDocument {
  id: number;
  document_type: string;
  version: string;
  title: string;
  content: string;
  effective_date: string;
  metadata: any;
}

export default function PrivacyPolicyPage() {
  const [document, setDocument] = useState<LegalDocument | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadPrivacyPolicy();
  }, []);

  const loadPrivacyPolicy = async () => {
    try {
      const response = await fetch('/api/legal/privacy-policy');
      if (response.ok) {
        const data = await response.json();
        setDocument(data);
      } else {
        throw new Error('Failed to load privacy policy');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
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
          <h1 className="text-xl font-bold text-red-800 mb-2">Error Loading Privacy Policy</h1>
          <p className="text-red-600">{error}</p>
        </Card>
      </div>
    );
  }

  if (!document) {
    return (
      <div className="max-w-4xl mx-auto p-6">
        <Card className="p-6">
          <h1 className="text-2xl font-bold mb-4">Privacy Policy</h1>
          <p className="text-gray-600">Privacy policy document not found.</p>
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

      <div className="mt-8 space-y-4">
        <Card className="p-4 bg-blue-50 border-blue-200">
          <h3 className="font-semibold text-blue-800 mb-2">Your Privacy Rights</h3>
          <p className="text-sm text-blue-700 mb-3">
            Under GDPR and other privacy laws, you have rights regarding your personal data.
          </p>
          <div className="flex gap-2">
            <a 
              href="/settings/privacy" 
              className="text-sm bg-blue-600 text-white px-3 py-1 rounded hover:bg-blue-700"
            >
              Manage Privacy Settings
            </a>
            <a 
              href="/legal/data-export" 
              className="text-sm border border-blue-600 text-blue-600 px-3 py-1 rounded hover:bg-blue-50"
            >
              Export My Data
            </a>
          </div>
        </Card>

        <div className="p-4 bg-gray-50 rounded">
          <p className="text-sm text-gray-600">
            If you have any questions about this Privacy Policy or our data practices, please contact us at{' '}
            <a href="mailto:privacy@scrollintel.com" className="text-blue-600 hover:underline">
              privacy@scrollintel.com
            </a>
          </p>
        </div>
      </div>
    </div>
  );
}