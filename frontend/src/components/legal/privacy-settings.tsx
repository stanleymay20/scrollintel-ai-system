"use client";

import React, { useState, useEffect } from 'react';
import { Button } from '../ui/button';
import { Card } from '../ui/card';
import { Alert } from '../ui/alert';
import { Badge } from '../ui/badge';

interface PrivacySettings {
  dataProcessingConsent: boolean;
  marketingEmails: boolean;
  analyticsTracking: boolean;
  thirdPartySharing: boolean;
  dataRetentionPeriod: string;
}

interface DataExportRequest {
  id: number;
  requestType: string;
  status: string;
  requestedAt: string;
  completedAt?: string;
}

export const PrivacySettingsPanel: React.FC = () => {
  const [settings, setSettings] = useState<PrivacySettings>({
    dataProcessingConsent: false,
    marketingEmails: false,
    analyticsTracking: false,
    thirdPartySharing: false,
    dataRetentionPeriod: '2_years'
  });
  
  const [exportRequests, setExportRequests] = useState<DataExportRequest[]>([]);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  useEffect(() => {
    loadPrivacySettings();
    loadExportRequests();
  }, []);

  const loadPrivacySettings = async () => {
    try {
      const response = await fetch('/api/legal/consent');
      if (response.ok) {
        const consents = await response.json();
        
        // Map consents to settings
        const newSettings = { ...settings };
        consents.forEach((consent: any) => {
          if (consent.consent_type === 'privacy_data_processing') {
            newSettings.dataProcessingConsent = consent.consent_given;
          } else if (consent.consent_type === 'privacy_marketing_emails') {
            newSettings.marketingEmails = consent.consent_given;
          } else if (consent.consent_type === 'privacy_analytics_tracking') {
            newSettings.analyticsTracking = consent.consent_given;
          } else if (consent.consent_type === 'privacy_third_party_sharing') {
            newSettings.thirdPartySharing = consent.consent_given;
          }
        });
        
        setSettings(newSettings);
      }
    } catch (error) {
      console.error('Error loading privacy settings:', error);
    }
  };

  const loadExportRequests = async () => {
    try {
      // This would be implemented when we have the export requests endpoint
      setExportRequests([]);
    } catch (error) {
      console.error('Error loading export requests:', error);
    }
  };

  const handleSettingChange = (key: keyof PrivacySettings, value: boolean | string) => {
    setSettings(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const savePrivacySettings = async () => {
    setLoading(true);
    setMessage(null);
    
    try {
      const response = await fetch('/api/legal/privacy-settings', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          data_processing_consent: settings.dataProcessingConsent,
          marketing_emails: settings.marketingEmails,
          analytics_tracking: settings.analyticsTracking,
          third_party_sharing: settings.thirdPartySharing,
          data_retention_period: settings.dataRetentionPeriod
        })
      });

      if (response.ok) {
        setMessage({ type: 'success', text: 'Privacy settings updated successfully' });
      } else {
        throw new Error('Failed to update settings');
      }
    } catch (error) {
      setMessage({ type: 'error', text: 'Failed to update privacy settings' });
    } finally {
      setLoading(false);
    }
  };

  const requestDataExport = async () => {
    setLoading(true);
    setMessage(null);
    
    try {
      const response = await fetch('/api/legal/data-export', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          request_type: 'export',
          verification_email: 'user@example.com' // This would come from user context
        })
      });

      if (response.ok) {
        setMessage({ 
          type: 'success', 
          text: 'Data export request submitted. You will receive an email when ready.' 
        });
        loadExportRequests();
      } else {
        throw new Error('Failed to request data export');
      }
    } catch (error) {
      setMessage({ type: 'error', text: 'Failed to request data export' });
    } finally {
      setLoading(false);
    }
  };

  const requestDataDeletion = async () => {
    if (!confirm('Are you sure you want to delete all your data? This action cannot be undone.')) {
      return;
    }
    
    setLoading(true);
    setMessage(null);
    
    try {
      const response = await fetch('/api/legal/data-deletion', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        }
      });

      if (response.ok) {
        setMessage({ 
          type: 'success', 
          text: 'Data deletion request submitted. Your account will be deleted within 30 days.' 
        });
      } else {
        throw new Error('Failed to request data deletion');
      }
    } catch (error) {
      setMessage({ type: 'error', text: 'Failed to request data deletion' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      <div>
        <h1 className="text-2xl font-bold mb-2">Privacy Settings</h1>
        <p className="text-gray-600">
          Manage your privacy preferences and data rights
        </p>
      </div>

      {message && (
        <Alert className={message.type === 'error' ? 'border-red-200 bg-red-50' : 'border-green-200 bg-green-50'}>
          <div className={message.type === 'error' ? 'text-red-800' : 'text-green-800'}>
            {message.text}
          </div>
        </Alert>
      )}

      {/* Privacy Preferences */}
      <Card className="p-6">
        <h2 className="text-xl font-semibold mb-4">Privacy Preferences</h2>
        
        <div className="space-y-4">
          <div className="flex items-center justify-between p-4 border rounded">
            <div className="flex-1">
              <h3 className="font-medium">Data Processing Consent</h3>
              <p className="text-sm text-gray-600 mt-1">
                Allow us to process your personal data to provide our services
              </p>
            </div>
            <input
              type="checkbox"
              checked={settings.dataProcessingConsent}
              onChange={(e) => handleSettingChange('dataProcessingConsent', e.target.checked)}
              className="w-4 h-4"
            />
          </div>

          <div className="flex items-center justify-between p-4 border rounded">
            <div className="flex-1">
              <h3 className="font-medium">Marketing Emails</h3>
              <p className="text-sm text-gray-600 mt-1">
                Receive emails about new features, updates, and promotions
              </p>
            </div>
            <input
              type="checkbox"
              checked={settings.marketingEmails}
              onChange={(e) => handleSettingChange('marketingEmails', e.target.checked)}
              className="w-4 h-4"
            />
          </div>

          <div className="flex items-center justify-between p-4 border rounded">
            <div className="flex-1">
              <h3 className="font-medium">Analytics Tracking</h3>
              <p className="text-sm text-gray-600 mt-1">
                Help us improve our service by tracking usage analytics
              </p>
            </div>
            <input
              type="checkbox"
              checked={settings.analyticsTracking}
              onChange={(e) => handleSettingChange('analyticsTracking', e.target.checked)}
              className="w-4 h-4"
            />
          </div>

          <div className="flex items-center justify-between p-4 border rounded">
            <div className="flex-1">
              <h3 className="font-medium">Third-Party Data Sharing</h3>
              <p className="text-sm text-gray-600 mt-1">
                Allow sharing anonymized data with trusted partners for research
              </p>
            </div>
            <input
              type="checkbox"
              checked={settings.thirdPartySharing}
              onChange={(e) => handleSettingChange('thirdPartySharing', e.target.checked)}
              className="w-4 h-4"
            />
          </div>

          <div className="flex items-center justify-between p-4 border rounded">
            <div className="flex-1">
              <h3 className="font-medium">Data Retention Period</h3>
              <p className="text-sm text-gray-600 mt-1">
                How long we keep your data after account deletion
              </p>
            </div>
            <select
              value={settings.dataRetentionPeriod}
              onChange={(e) => handleSettingChange('dataRetentionPeriod', e.target.value)}
              className="px-3 py-2 border rounded"
            >
              <option value="1_year">1 Year</option>
              <option value="2_years">2 Years</option>
              <option value="5_years">5 Years</option>
            </select>
          </div>
        </div>

        <div className="mt-6">
          <Button 
            onClick={savePrivacySettings} 
            disabled={loading}
            className="bg-blue-600 hover:bg-blue-700"
          >
            {loading ? 'Saving...' : 'Save Privacy Settings'}
          </Button>
        </div>
      </Card>

      {/* Data Rights */}
      <Card className="p-6">
        <h2 className="text-xl font-semibold mb-4">Your Data Rights</h2>
        
        <div className="space-y-4">
          <div className="p-4 border rounded">
            <h3 className="font-medium mb-2">Export Your Data</h3>
            <p className="text-sm text-gray-600 mb-3">
              Download a copy of all your personal data (GDPR Article 20)
            </p>
            <Button 
              onClick={requestDataExport} 
              disabled={loading}
              variant="outline"
            >
              Request Data Export
            </Button>
          </div>

          <div className="p-4 border rounded border-red-200">
            <h3 className="font-medium mb-2 text-red-800">Delete Your Data</h3>
            <p className="text-sm text-gray-600 mb-3">
              Permanently delete all your personal data (GDPR Article 17 - Right to be Forgotten)
            </p>
            <Button 
              onClick={requestDataDeletion} 
              disabled={loading}
              variant="outline"
              className="border-red-300 text-red-700 hover:bg-red-50"
            >
              Request Data Deletion
            </Button>
          </div>
        </div>
      </Card>

      {/* Export Requests History */}
      {exportRequests.length > 0 && (
        <Card className="p-6">
          <h2 className="text-xl font-semibold mb-4">Request History</h2>
          
          <div className="space-y-3">
            {exportRequests.map((request) => (
              <div key={request.id} className="flex items-center justify-between p-3 border rounded">
                <div>
                  <div className="font-medium">
                    {request.requestType === 'export' ? 'Data Export' : 'Data Deletion'}
                  </div>
                  <div className="text-sm text-gray-600">
                    Requested: {new Date(request.requestedAt).toLocaleDateString()}
                  </div>
                </div>
                <Badge 
                  variant={request.status === 'completed' ? 'default' : 'secondary'}
                >
                  {request.status}
                </Badge>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
};

export default PrivacySettingsPanel;