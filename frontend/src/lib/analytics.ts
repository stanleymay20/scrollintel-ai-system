/**
 * Analytics and Marketing Tracking Library
 * Client-side analytics tracking with Google Analytics integration
 */

interface AnalyticsConfig {
  googleAnalyticsId?: string;
  measurementId?: string;
  apiEndpoint: string;
  enableDebug?: boolean;
}

interface EventProperties {
  [key: string]: any;
}

interface UserProperties {
  [key: string]: any;
}

interface CampaignData {
  utm_source?: string;
  utm_medium?: string;
  utm_campaign?: string;
  utm_content?: string;
  utm_term?: string;
}

class AnalyticsTracker {
  private config: AnalyticsConfig;
  private userId: string | null = null;
  private sessionId: string;
  private isInitialized = false;

  constructor(config: AnalyticsConfig) {
    this.config = config;
    this.sessionId = this.generateSessionId();
    
    // Initialize Google Analytics if configured
    if (config.googleAnalyticsId) {
      this.initializeGoogleAnalytics();
    }
  }

  /**
   * Initialize the analytics tracker
   */
  async initialize(userId?: string): Promise<void> {
    try {
      this.userId = userId || this.getAnonymousUserId();
      this.isInitialized = true;

      // Track initial page view
      await this.trackPageView();

      this.log('Analytics tracker initialized', { userId: this.userId, sessionId: this.sessionId });
    } catch (error) {
      console.error('Failed to initialize analytics tracker:', error);
    }
  }

  /**
   * Track a custom event
   */
  async trackEvent(
    eventName: string,
    properties: EventProperties = {},
    campaignData?: CampaignData
  ): Promise<void> {
    if (!this.isInitialized) {
      console.warn('Analytics tracker not initialized');
      return;
    }

    try {
      const eventData = {
        user_id: this.userId,
        session_id: this.sessionId,
        event_name: eventName,
        properties: {
          ...properties,
          timestamp: new Date().toISOString(),
          page_url: window.location.href,
          page_title: document.title,
          referrer: document.referrer
        },
        page_url: window.location.href,
        user_agent: navigator.userAgent,
        ip_address: '', // Will be filled by server
        campaign_data: campaignData || this.extractUTMParameters()
      };

      // Send to our analytics API
      await this.sendToAPI('/api/analytics/track/event', eventData);

      // Send to Google Analytics if configured
      if (this.config.googleAnalyticsId && window.gtag) {
        window.gtag('event', eventName, {
          custom_parameter_1: properties,
          user_id: this.userId,
          session_id: this.sessionId
        });
      }

      this.log('Event tracked', { eventName, properties });
    } catch (error) {
      console.error('Failed to track event:', error);
    }
  }

  /**
   * Track a page view
   */
  async trackPageView(
    pageUrl?: string,
    pageTitle?: string,
    referrer?: string
  ): Promise<void> {
    if (!this.isInitialized) {
      console.warn('Analytics tracker not initialized');
      return;
    }

    try {
      const pageData = {
        user_id: this.userId,
        session_id: this.sessionId,
        page_url: pageUrl || window.location.href,
        page_title: pageTitle || document.title,
        user_agent: navigator.userAgent,
        ip_address: '', // Will be filled by server
        referrer: referrer || document.referrer
      };

      // Send to our analytics API
      await this.sendToAPI('/api/analytics/track/page-view', pageData);

      // Send to Google Analytics if configured
      if (this.config.googleAnalyticsId && window.gtag) {
        window.gtag('config', this.config.googleAnalyticsId, {
          page_title: pageData.page_title,
          page_location: pageData.page_url,
          user_id: this.userId
        });
      }

      this.log('Page view tracked', pageData);
    } catch (error) {
      console.error('Failed to track page view:', error);
    }
  }

  /**
   * Track a conversion
   */
  async trackConversion(
    conversionType: string,
    conversionValue: number = 0,
    properties: EventProperties = {}
  ): Promise<void> {
    try {
      // Track as regular event first
      await this.trackEvent('conversion', {
        conversion_type: conversionType,
        conversion_value: conversionValue,
        ...properties
      });

      // Send to conversion tracking API
      const conversionData = {
        user_id: this.userId,
        session_id: this.sessionId,
        conversion_type: conversionType,
        conversion_value: conversionValue,
        attribution_model: 'last_touch'
      };

      await this.sendToAPI('/api/analytics/conversions', conversionData);

      // Send to Google Analytics Enhanced Ecommerce if configured
      if (this.config.googleAnalyticsId && window.gtag) {
        window.gtag('event', 'purchase', {
          transaction_id: this.generateTransactionId(),
          value: conversionValue,
          currency: 'USD',
          items: [{
            item_id: conversionType,
            item_name: conversionType,
            category: 'conversion',
            quantity: 1,
            price: conversionValue
          }]
        });
      }

      this.log('Conversion tracked', { conversionType, conversionValue });
    } catch (error) {
      console.error('Failed to track conversion:', error);
    }
  }

  /**
   * Update user profile
   */
  async updateUserProfile(properties: UserProperties): Promise<void> {
    if (!this.isInitialized) {
      console.warn('Analytics tracker not initialized');
      return;
    }

    try {
      const profileData = {
        events: [{
          event_name: 'profile_update',
          timestamp: new Date().toISOString(),
          session_id: this.sessionId,
          properties
        }],
        properties
      };

      await this.sendToAPI(`/api/analytics/users/${this.userId}/profile`, profileData);

      // Update Google Analytics user properties if configured
      if (this.config.googleAnalyticsId && window.gtag) {
        window.gtag('config', this.config.googleAnalyticsId, {
          user_properties: properties
        });
      }

      this.log('User profile updated', properties);
    } catch (error) {
      console.error('Failed to update user profile:', error);
    }
  }

  /**
   * Set user ID
   */
  setUserId(userId: string): void {
    this.userId = userId;
    
    // Update Google Analytics user ID if configured
    if (this.config.googleAnalyticsId && window.gtag) {
      window.gtag('config', this.config.googleAnalyticsId, {
        user_id: userId
      });
    }

    this.log('User ID set', { userId });
  }

  /**
   * Get current user ID
   */
  getUserId(): string | null {
    return this.userId;
  }

  /**
   * Get current session ID
   */
  getSessionId(): string {
    return this.sessionId;
  }

  /**
   * Track A/B test assignment
   */
  async trackExperimentAssignment(
    experimentId: string,
    variantId: string,
    userProperties: UserProperties = {}
  ): Promise<void> {
    try {
      const assignmentData = {
        experiment_id: experimentId,
        user_id: this.userId,
        session_id: this.sessionId,
        user_properties: userProperties
      };

      const response = await this.sendToAPI(`/api/analytics/experiments/${experimentId}/assign`, assignmentData);
      
      // Track assignment as event
      await this.trackEvent('experiment_assigned', {
        experiment_id: experimentId,
        variant_id: variantId
      });

      this.log('Experiment assignment tracked', { experimentId, variantId });
      return response;
    } catch (error) {
      console.error('Failed to track experiment assignment:', error);
    }
  }

  /**
   * Track A/B test result
   */
  async trackExperimentResult(
    experimentId: string,
    metricName: string,
    metricValue: number
  ): Promise<void> {
    try {
      const resultData = {
        user_id: this.userId,
        metric_name: metricName,
        metric_value: metricValue
      };

      await this.sendToAPI(`/api/analytics/experiments/${experimentId}/results`, resultData);

      // Track result as event
      await this.trackEvent('experiment_result', {
        experiment_id: experimentId,
        metric_name: metricName,
        metric_value: metricValue
      });

      this.log('Experiment result tracked', { experimentId, metricName, metricValue });
    } catch (error) {
      console.error('Failed to track experiment result:', error);
    }
  }

  /**
   * Initialize Google Analytics
   */
  private initializeGoogleAnalytics(): void {
    if (typeof window === 'undefined') return;

    // Load Google Analytics script
    const script = document.createElement('script');
    script.async = true;
    script.src = `https://www.googletagmanager.com/gtag/js?id=${this.config.googleAnalyticsId}`;
    document.head.appendChild(script);

    // Initialize gtag
    window.dataLayer = window.dataLayer || [];
    window.gtag = function() {
      window.dataLayer.push(arguments);
    };

    window.gtag('js', new Date());
    window.gtag('config', this.config.googleAnalyticsId, {
      send_page_view: false, // We'll handle page views manually
      anonymize_ip: true,
      allow_google_signals: false,
      allow_ad_personalization_signals: false
    });

    this.log('Google Analytics initialized', { id: this.config.googleAnalyticsId });
  }

  /**
   * Extract UTM parameters from current URL
   */
  private extractUTMParameters(): CampaignData {
    if (typeof window === 'undefined') return {};

    const urlParams = new URLSearchParams(window.location.search);
    return {
      utm_source: urlParams.get('utm_source') || undefined,
      utm_medium: urlParams.get('utm_medium') || undefined,
      utm_campaign: urlParams.get('utm_campaign') || undefined,
      utm_content: urlParams.get('utm_content') || undefined,
      utm_term: urlParams.get('utm_term') || undefined
    };
  }

  /**
   * Generate session ID
   */
  private generateSessionId(): string {
    return `session_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`;
  }

  /**
   * Generate transaction ID
   */
  private generateTransactionId(): string {
    return `txn_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`;
  }

  /**
   * Get or create anonymous user ID
   */
  private getAnonymousUserId(): string {
    if (typeof window === 'undefined') return `anon_${Date.now()}`;

    let userId = localStorage.getItem('analytics_user_id');
    if (!userId) {
      userId = `anon_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`;
      localStorage.setItem('analytics_user_id', userId);
    }
    return userId;
  }

  /**
   * Send data to API
   */
  private async sendToAPI(endpoint: string, data: any): Promise<any> {
    try {
      const response = await fetch(`${this.config.apiEndpoint}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
      });

      if (!response.ok) {
        throw new Error(`API request failed: ${response.status} ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('API request failed:', error);
      throw error;
    }
  }

  /**
   * Debug logging
   */
  private log(message: string, data?: any): void {
    if (this.config.enableDebug) {
      console.log(`[Analytics] ${message}`, data);
    }
  }
}

// Global analytics instance
let analyticsInstance: AnalyticsTracker | null = null;

/**
 * Initialize analytics tracker
 */
export function initializeAnalytics(config: AnalyticsConfig): AnalyticsTracker {
  analyticsInstance = new AnalyticsTracker(config);
  return analyticsInstance;
}

/**
 * Get analytics instance
 */
export function getAnalytics(): AnalyticsTracker | null {
  return analyticsInstance;
}

/**
 * Convenience functions for common tracking
 */
export const analytics = {
  track: (eventName: string, properties?: EventProperties) => {
    return analyticsInstance?.trackEvent(eventName, properties);
  },
  
  page: (pageUrl?: string, pageTitle?: string) => {
    return analyticsInstance?.trackPageView(pageUrl, pageTitle);
  },
  
  conversion: (type: string, value?: number, properties?: EventProperties) => {
    return analyticsInstance?.trackConversion(type, value, properties);
  },
  
  identify: (userId: string, properties?: UserProperties) => {
    analyticsInstance?.setUserId(userId);
    return analyticsInstance?.updateUserProfile(properties || {});
  },
  
  experiment: {
    assign: (experimentId: string, variantId: string, properties?: UserProperties) => {
      return analyticsInstance?.trackExperimentAssignment(experimentId, variantId, properties);
    },
    
    result: (experimentId: string, metricName: string, metricValue: number) => {
      return analyticsInstance?.trackExperimentResult(experimentId, metricName, metricValue);
    }
  }
};

// Auto-track page views on route changes (for Next.js)
if (typeof window !== 'undefined') {
  let currentPath = window.location.pathname;
  
  const observer = new MutationObserver(() => {
    if (window.location.pathname !== currentPath) {
      currentPath = window.location.pathname;
      analytics.page();
    }
  });
  
  observer.observe(document.body, {
    childList: true,
    subtree: true
  });
}

// TypeScript declarations for Google Analytics
declare global {
  interface Window {
    dataLayer: any[];
    gtag: (...args: any[]) => void;
  }
}

export default AnalyticsTracker;