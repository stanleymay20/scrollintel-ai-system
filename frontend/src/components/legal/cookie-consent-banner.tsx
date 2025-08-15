"use client";

import React, { useState, useEffect } from 'react';
import { Button } from '../ui/button';
import { Card } from '../ui/card';
import { Badge } from '../ui/badge';

interface CookieSettings {
  necessary: boolean;
  analytics: boolean;
  marketing: boolean;
  preferences: boolean;
}

interface CookieConsentBannerProps {
  onAcceptAll?: () => void;
  onRejectAll?: () => void;
  onCustomize?: (settings: CookieSettings) => void;
}

export const CookieConsentBanner: React.FC<CookieConsentBannerProps> = ({
  onAcceptAll,
  onRejectAll,
  onCustomize
}) => {
  const [isVisible, setIsVisible] = useState(false);
  const [showDetails, setShowDetails] = useState(false);
  const [settings, setSettings] = useState<CookieSettings>({
    necessary: true,
    analytics: false,
    marketing: false,
    preferences: false
  });

  useEffect(() => {
    // Check if user has already made a choice
    const hasConsent = localStorage.getItem('cookie-consent');
    if (!hasConsent) {
      setIsVisible(true);
    }
  }, []);

  const handleAcceptAll = () => {
    const allAccepted = {
      necessary: true,
      analytics: true,
      marketing: true,
      preferences: true
    };
    
    localStorage.setItem('cookie-consent', JSON.stringify(allAccepted));
    setIsVisible(false);
    onAcceptAll?.();
  };

  const handleRejectAll = () => {
    const onlyNecessary = {
      necessary: true,
      analytics: false,
      marketing: false,
      preferences: false
    };
    
    localStorage.setItem('cookie-consent', JSON.stringify(onlyNecessary));
    setIsVisible(false);
    onRejectAll?.();
  };

  const handleCustomize = () => {
    localStorage.setItem('cookie-consent', JSON.stringify(settings));
    setIsVisible(false);
    onCustomize?.(settings);
  };

  const handleSettingChange = (key: keyof CookieSettings, value: boolean) => {
    if (key === 'necessary') return; // Necessary cookies cannot be disabled
    
    setSettings(prev => ({
      ...prev,
      [key]: value
    }));
  };

  if (!isVisible) return null;

  return (
    <div className="fixed bottom-0 left-0 right-0 z-50 p-4 bg-white border-t shadow-lg">
      <Card className="max-w-4xl mx-auto p-6">
        <div className="space-y-4">
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <h3 className="text-lg font-semibold mb-2">Cookie Preferences</h3>
              <p className="text-sm text-gray-600 mb-4">
                We use cookies to enhance your experience, analyze site usage, and assist in marketing efforts. 
                You can customize your preferences or accept all cookies.
              </p>
            </div>
          </div>

          {!showDetails ? (
            <div className="flex flex-wrap gap-3">
              <Button onClick={handleAcceptAll} className="bg-blue-600 hover:bg-blue-700">
                Accept All
              </Button>
              <Button onClick={handleRejectAll} variant="outline">
                Reject All
              </Button>
              <Button 
                onClick={() => setShowDetails(true)} 
                variant="outline"
              >
                Customize
              </Button>
              <Button 
                variant="ghost" 
                size="sm"
                onClick={() => window.open('/legal/cookie-policy', '_blank')}
              >
                Learn More
              </Button>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="grid gap-4">
                <div className="flex items-center justify-between p-3 border rounded">
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <h4 className="font-medium">Necessary Cookies</h4>
                      <Badge variant="secondary">Required</Badge>
                    </div>
                    <p className="text-sm text-gray-600 mt-1">
                      Essential for the website to function properly. Cannot be disabled.
                    </p>
                  </div>
                  <input
                    type="checkbox"
                    checked={settings.necessary}
                    disabled
                    className="w-4 h-4"
                  />
                </div>

                <div className="flex items-center justify-between p-3 border rounded">
                  <div className="flex-1">
                    <h4 className="font-medium">Analytics Cookies</h4>
                    <p className="text-sm text-gray-600 mt-1">
                      Help us understand how visitors interact with our website.
                    </p>
                  </div>
                  <input
                    type="checkbox"
                    checked={settings.analytics}
                    onChange={(e) => handleSettingChange('analytics', e.target.checked)}
                    className="w-4 h-4"
                  />
                </div>

                <div className="flex items-center justify-between p-3 border rounded">
                  <div className="flex-1">
                    <h4 className="font-medium">Marketing Cookies</h4>
                    <p className="text-sm text-gray-600 mt-1">
                      Used to deliver personalized advertisements and track campaign performance.
                    </p>
                  </div>
                  <input
                    type="checkbox"
                    checked={settings.marketing}
                    onChange={(e) => handleSettingChange('marketing', e.target.checked)}
                    className="w-4 h-4"
                  />
                </div>

                <div className="flex items-center justify-between p-3 border rounded">
                  <div className="flex-1">
                    <h4 className="font-medium">Preference Cookies</h4>
                    <p className="text-sm text-gray-600 mt-1">
                      Remember your preferences and settings for a better experience.
                    </p>
                  </div>
                  <input
                    type="checkbox"
                    checked={settings.preferences}
                    onChange={(e) => handleSettingChange('preferences', e.target.checked)}
                    className="w-4 h-4"
                  />
                </div>
              </div>

              <div className="flex gap-3 pt-4 border-t">
                <Button onClick={handleCustomize} className="bg-blue-600 hover:bg-blue-700">
                  Save Preferences
                </Button>
                <Button onClick={() => setShowDetails(false)} variant="outline">
                  Back
                </Button>
              </div>
            </div>
          )}
        </div>
      </Card>
    </div>
  );
};

export default CookieConsentBanner;