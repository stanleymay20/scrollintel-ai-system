/**
 * Analytics Initialization and Configuration
 * Initializes all analytics systems including Google Analytics, internal tracking,
 * and marketing attribution
 */

import { initializeAnalytics, analytics } from './analytics';

interface AnalyticsInitConfig {
  googleAnalyticsId?: string;
  measurementId?: string;
  apiEndpoint: string;
  enableDebug?: boolean;
  userId?: string;
  userProperties?: Record<string, any>;
}

class AnalyticsManager {
  private isInitialized = false;
  private config: AnalyticsInitConfig | null = null;
  private trackingQueue: Array<() => void> = [];

  /**
   * Initialize all analytics systems
   */
  async initialize(config: AnalyticsInitConfig): Promise<void> {
    try {
      this.config = config;

      // Initialize internal analytics tracker
      const tracker = initializeAnalytics({
        googleAnalyticsId: config.googleAnalyticsId,
        measurementId: config.measurementId,
        apiEndpoint: config.apiEndpoint,
        enableDebug: config.enableDebug || false
      });

      await tracker.initialize(config.userId);

      // Set user properties if provided
      if (config.userProperties) {
        await analytics.identify(config.userId || '', config.userProperties);
      }

      // Initialize Google Analytics 4 if configured
      if (config.measurementId) {
        await this.initializeGA4(config.measurementId);
      }

      // Initialize marketing attribution tracking
      this.initializeMarketingAttribution();

      // Initialize A/B testing
      this.initializeABTesting();

      // Initialize user segmentation
      this.initializeUserSegmentation();

      // Process queued tracking calls
      this.processTrackingQueue();

      this.isInitialized = true;
      this.log('Analytics systems initialized successfully');

      // Track initialization event
      await analytics.track('analytics_initialized', {
        systems: ['internal', 'google_analytics', 'attribution', 'ab_testing', 'segmentation'],
        timestamp: new Date().toISOString()
      });

    } catch (error) {
      console.error('Failed to initialize analytics:', error);
      throw error;
    }
  }

  /**
   * Initialize Google Analytics 4
   */
  private async initializeGA4(measurementId: string): Promise<void> {
    if (typeof window === 'undefined') return;

    try {
      // Load Google Analytics script if not already loaded
      if (!window.gtag) {
        const script = document.createElement('script');
        script.async = true;
        script.src = `https://www.googletagmanager.com/gtag/js?id=${measurementId}`;
        document.head.appendChild(script);

        // Initialize gtag
        window.dataLayer = window.dataLayer || [];
        window.gtag = function() {
          window.dataLayer.push(arguments);
        };

        window.gtag('js', new Date());
      }

      // Configure Google Analytics
      window.gtag('config', measurementId, {
        // Privacy settings
        anonymize_ip: true,
        allow_google_signals: false,
        allow_ad_personalization_signals: false,
        
        // Enhanced measurement
        enhanced_measurement: {
          scrolls: true,
          outbound_clicks: true,
          site_search: true,
          video_engagement: true,
          file_downloads: true
        },

        // Custom parameters
        custom_map: {
          'custom_parameter_1': 'user_type',
          'custom_parameter_2': 'subscription_tier',
          'custom_parameter_3': 'feature_usage'
        },

        // Debug mode
        debug_mode: this.config?.enableDebug || false
      });

      // Set user properties if available
      if (this.config?.userProperties) {
        window.gtag('config', measurementId, {
          user_properties: this.config.userProperties
        });
      }

      this.log('Google Analytics 4 initialized');

    } catch (error) {
      console.error('Failed to initialize Google Analytics 4:', error);
    }
  }

  /**
   * Initialize marketing attribution tracking
   */
  private initializeMarketingAttribution(): void {
    try {
      // Track UTM parameters from current URL
      const urlParams = new URLSearchParams(window.location.search);
      const utmParams = {
        utm_source: urlParams.get('utm_source'),
        utm_medium: urlParams.get('utm_medium'),
        utm_campaign: urlParams.get('utm_campaign'),
        utm_content: urlParams.get('utm_content'),
        utm_term: urlParams.get('utm_term')
      };

      // Store UTM parameters in session storage for attribution
      if (Object.values(utmParams).some(param => param !== null)) {
        sessionStorage.setItem('scrollintel_utm_params', JSON.stringify(utmParams));
        
        // Track marketing touchpoint
        analytics.track('marketing_touchpoint', {
          ...utmParams,
          referrer: document.referrer,
          landing_page: window.location.href
        });
      }

      this.log('Marketing attribution initialized');

    } catch (error) {
      console.error('Failed to initialize marketing attribution:', error);
    }
  }

  /**
   * Initialize A/B testing
   */
  private initializeABTesting(): void {
    try {
      // Check for active experiments
      this.checkActiveExperiments();

      this.log('A/B testing initialized');

    } catch (error) {
      console.error('Failed to initialize A/B testing:', error);
    }
  }

  /**
   * Initialize user segmentation
   */
  private initializeUserSegmentation(): void {
    try {
      // Track user properties for segmentation
      const userAgent = navigator.userAgent;
      const screenResolution = `${screen.width}x${screen.height}`;
      const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
      const language = navigator.language;

      analytics.track('user_segmentation_data', {
        user_agent: userAgent,
        screen_resolution: screenResolution,
        timezone: timezone,
        language: language,
        referrer: document.referrer
      });

      this.log('User segmentation initialized');

    } catch (error) {
      console.error('Failed to initialize user segmentation:', error);
    }
  }

  /**
   * Check for active A/B test experiments
   */
  private async checkActiveExperiments(): Promise<void> {
    try {
      const response = await fetch(`${this.config?.apiEndpoint}/api/analytics/experiments/dashboard`);
      if (response.ok) {
        const data = await response.json();
        const runningExperiments = data.experiments?.filter((exp: any) => exp.status === 'running') || [];

        // Assign user to running experiments
        for (const experiment of runningExperiments) {
          await this.assignToExperiment(experiment.experiment_id);
        }
      }
    } catch (error) {
      console.error('Failed to check active experiments:', error);
    }
  }

  /**
   * Assign user to A/B test experiment
   */
  private async assignToExperiment(experimentId: string): Promise<void> {
    try {
      const userId = this.config?.userId || localStorage.getItem('analytics_user_id');
      const sessionId = sessionStorage.getItem('analytics_session_id') || 'unknown';

      if (!userId) return;

      const response = await fetch(`${this.config?.apiEndpoint}/api/analytics/experiments/${experimentId}/assign`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_id: userId,
          session_id: sessionId,
          user_properties: this.config?.userProperties || {}
        })
      });

      if (response.ok) {
        const data = await response.json();
        if (data.variant_id) {
          // Store experiment assignment
          const assignments = JSON.parse(localStorage.getItem('experiment_assignments') || '{}');
          assignments[experimentId] = data.variant_id;
          localStorage.setItem('experiment_assignments', JSON.stringify(assignments));

          // Track assignment
          await analytics.experiment.assign(experimentId, data.variant_id, this.config?.userProperties);
        }
      }
    } catch (error) {
      console.error('Failed to assign to experiment:', error);
    }
  }

  /**
   * Track page view with enhanced data
   */
  async trackPageView(pageUrl?: string, pageTitle?: string): Promise<void> {
    if (!this.isInitialized) {
      this.queueTracking(() => this.trackPageView(pageUrl, pageTitle));
      return;
    }

    try {
      const url = pageUrl || window.location.href;
      const title = pageTitle || document.title;

      // Track with internal analytics
      await analytics.page(url, title);

      // Track with Google Analytics
      if (window.gtag && this.config?.measurementId) {
        window.gtag('config', this.config.measurementId, {
          page_title: title,
          page_location: url
        });
      }

      // Track for funnel analysis
      await analytics.track('page_view', {
        page_url: url,
        page_title: title,
        referrer: document.referrer,
        timestamp: new Date().toISOString()
      });

    } catch (error) {
      console.error('Failed to track page view:', error);
    }
  }

  /**
   * Track custom event with enhanced attribution
   */
  async trackEvent(eventName: string, properties: Record<string, any> = {}): Promise<void> {
    if (!this.isInitialized) {
      this.queueTracking(() => this.trackEvent(eventName, properties));
      return;
    }

    try {
      // Add UTM parameters if available
      const utmParams = sessionStorage.getItem('scrollintel_utm_params');
      if (utmParams) {
        properties.utm_data = JSON.parse(utmParams);
      }

      // Add experiment assignments
      const assignments = localStorage.getItem('experiment_assignments');
      if (assignments) {
        properties.experiment_assignments = JSON.parse(assignments);
      }

      // Track with internal analytics
      await analytics.track(eventName, properties);

      // Track with Google Analytics
      if (window.gtag) {
        window.gtag('event', eventName, {
          event_category: 'ScrollIntel',
          event_label: properties.label || '',
          value: properties.value || 0,
          custom_parameter_1: properties.user_type || '',
          custom_parameter_2: properties.subscription_tier || '',
          custom_parameter_3: properties.feature_usage || ''
        });
      }

    } catch (error) {
      console.error('Failed to track event:', error);
    }
  }

  /**
   * Track conversion with attribution
   */
  async trackConversion(conversionType: string, value: number = 0, properties: Record<string, any> = {}): Promise<void> {
    if (!this.isInitialized) {
      this.queueTracking(() => this.trackConversion(conversionType, value, properties));
      return;
    }

    try {
      // Track with internal analytics
      await analytics.conversion(conversionType, value, properties);

      // Track with Google Analytics Enhanced Ecommerce
      if (window.gtag) {
        window.gtag('event', 'purchase', {
          transaction_id: `txn_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`,
          value: value,
          currency: 'USD',
          items: [{
            item_id: conversionType,
            item_name: conversionType,
            item_category: 'conversion',
            quantity: 1,
            price: value
          }]
        });
      }

      // Track A/B test results for running experiments
      const assignments = localStorage.getItem('experiment_assignments');
      if (assignments) {
        const experimentAssignments = JSON.parse(assignments);
        for (const [experimentId, variantId] of Object.entries(experimentAssignments)) {
          await analytics.experiment.result(experimentId, 'conversion_rate', 1);
        }
      }

    } catch (error) {
      console.error('Failed to track conversion:', error);
    }
  }

  /**
   * Update user properties
   */
  async updateUserProperties(properties: Record<string, any>): Promise<void> {
    if (!this.isInitialized) {
      this.queueTracking(() => this.updateUserProperties(properties));
      return;
    }

    try {
      // Update internal analytics
      const userId = this.config?.userId || localStorage.getItem('analytics_user_id');
      if (userId) {
        await analytics.identify(userId, properties);
      }

      // Update Google Analytics user properties
      if (window.gtag && this.config?.measurementId) {
        window.gtag('config', this.config.measurementId, {
          user_properties: properties
        });
      }

      // Update stored user properties
      if (this.config) {
        this.config.userProperties = { ...this.config.userProperties, ...properties };
      }

    } catch (error) {
      console.error('Failed to update user properties:', error);
    }
  }

  /**
   * Queue tracking calls until initialization is complete
   */
  private queueTracking(trackingCall: () => void): void {
    this.trackingQueue.push(trackingCall);
  }

  /**
   * Process queued tracking calls
   */
  private processTrackingQueue(): void {
    while (this.trackingQueue.length > 0) {
      const trackingCall = this.trackingQueue.shift();
      if (trackingCall) {
        try {
          trackingCall();
        } catch (error) {
          console.error('Failed to process queued tracking call:', error);
        }
      }
    }
  }

  /**
   * Debug logging
   */
  private log(message: string, data?: any): void {
    if (this.config?.enableDebug) {
      console.log(`[AnalyticsManager] ${message}`, data);
    }
  }

  /**
   * Get initialization status
   */
  isReady(): boolean {
    return this.isInitialized;
  }

  /**
   * Get current configuration
   */
  getConfig(): AnalyticsInitConfig | null {
    return this.config;
  }
}

// Global analytics manager instance
const analyticsManager = new AnalyticsManager();

// Export convenience functions
export const initAnalytics = (config: AnalyticsInitConfig) => analyticsManager.initialize(config);
export const trackPageView = (pageUrl?: string, pageTitle?: string) => analyticsManager.trackPageView(pageUrl, pageTitle);
export const trackEvent = (eventName: string, properties?: Record<string, any>) => analyticsManager.trackEvent(eventName, properties);
export const trackConversion = (conversionType: string, value?: number, properties?: Record<string, any>) => analyticsManager.trackConversion(conversionType, value, properties);
export const updateUserProperties = (properties: Record<string, any>) => analyticsManager.updateUserProperties(properties);
export const isAnalyticsReady = () => analyticsManager.isReady();

export default analyticsManager;

// TypeScript declarations
declare global {
  interface Window {
    dataLayer: any[];
    gtag: (...args: any[]) => void;
  }
}