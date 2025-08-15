#!/usr/bin/env python3
"""
Google Analytics Setup and Integration Script
Sets up Google Analytics 4 integration with ScrollIntel platform
"""

import os
import json
import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoogleAnalyticsIntegrator:
    """Google Analytics 4 integration manager"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.measurement_id = config.get('measurement_id')
        self.api_secret = config.get('api_secret')
        self.tracking_id = config.get('tracking_id')  # Legacy UA tracking ID
        self.base_url = "https://www.google-analytics.com"
        
    def setup_gtag_config(self) -> str:
        """Generate Google Analytics gtag configuration"""
        gtag_config = f"""
<!-- Google Analytics 4 Configuration -->
<script async src="https://www.googletagmanager.com/gtag/js?id={self.measurement_id}"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){{dataLayer.push(arguments);}}
  gtag('js', new Date());

  gtag('config', '{self.measurement_id}', {{
    // Enhanced measurement settings
    enhanced_measurement: {{
      scrolls: true,
      outbound_clicks: true,
      site_search: true,
      video_engagement: true,
      file_downloads: true
    }},
    
    // Privacy settings
    anonymize_ip: true,
    allow_google_signals: false,
    allow_ad_personalization_signals: false,
    
    // Custom parameters
    custom_map: {{
      'custom_parameter_1': 'user_type',
      'custom_parameter_2': 'subscription_tier',
      'custom_parameter_3': 'feature_usage'
    }},
    
    // Debug mode (remove in production)
    debug_mode: {str(self.config.get('debug_mode', False)).lower()}
  }});

  // Custom event tracking functions
  window.trackScrollIntelEvent = function(eventName, parameters) {{
    gtag('event', eventName, {{
      event_category: 'ScrollIntel',
      event_label: parameters.label || '',
      value: parameters.value || 0,
      user_id: parameters.user_id || '',
      session_id: parameters.session_id || '',
      custom_parameter_1: parameters.user_type || '',
      custom_parameter_2: parameters.subscription_tier || '',
      custom_parameter_3: parameters.feature_usage || ''
    }});
  }};

  // Enhanced ecommerce tracking
  window.trackScrollIntelPurchase = function(transactionData) {{
    gtag('event', 'purchase', {{
      transaction_id: transactionData.transaction_id,
      value: transactionData.value,
      currency: transactionData.currency || 'USD',
      items: transactionData.items || []
    }});
  }};

  // User engagement tracking
  window.trackScrollIntelEngagement = function(engagementData) {{
    gtag('event', 'engagement', {{
      engagement_time_msec: engagementData.time_msec || 0,
      page_title: engagementData.page_title || document.title,
      page_location: engagementData.page_location || window.location.href
    }});
  }};
</script>
"""
        return gtag_config
    
    async def send_measurement_protocol_event(self, 
                                            client_id: str,
                                            event_name: str,
                                            event_parameters: Dict[str, Any]) -> bool:
        """Send event via Google Analytics Measurement Protocol"""
        try:
            url = f"{self.base_url}/mp/collect"
            
            payload = {
                "client_id": client_id,
                "events": [{
                    "name": event_name,
                    "params": event_parameters
                }]
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            params = {
                "measurement_id": self.measurement_id,
                "api_secret": self.api_secret
            }
            
            response = requests.post(
                url, 
                params=params,
                headers=headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 204:
                logger.info(f"Successfully sent event '{event_name}' to GA4")
                return True
            else:
                logger.error(f"Failed to send event to GA4: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending event to GA4: {str(e)}")
            return False
    
    async def validate_measurement_protocol(self) -> bool:
        """Validate Measurement Protocol configuration"""
        try:
            url = f"{self.base_url}/mp/collect/validate"
            
            test_payload = {
                "client_id": "test_client_123",
                "events": [{
                    "name": "test_event",
                    "params": {
                        "test_parameter": "test_value"
                    }
                }]
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            params = {
                "measurement_id": self.measurement_id,
                "api_secret": self.api_secret
            }
            
            response = requests.post(
                url,
                params=params,
                headers=headers,
                json=test_payload,
                timeout=10
            )
            
            if response.status_code == 204:
                logger.info("Measurement Protocol validation successful")
                return True
            else:
                logger.error(f"Measurement Protocol validation failed: {response.status_code}")
                if response.text:
                    logger.error(f"Response: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error validating Measurement Protocol: {str(e)}")
            return False
    
    def generate_enhanced_ecommerce_config(self) -> str:
        """Generate enhanced ecommerce tracking configuration"""
        ecommerce_config = """
// Enhanced Ecommerce Tracking for ScrollIntel
window.ScrollIntelEcommerce = {
  // Track subscription purchase
  trackSubscription: function(subscriptionData) {
    gtag('event', 'purchase', {
      transaction_id: subscriptionData.transaction_id,
      value: subscriptionData.value,
      currency: subscriptionData.currency || 'USD',
      items: [{
        item_id: subscriptionData.plan_id,
        item_name: subscriptionData.plan_name,
        item_category: 'subscription',
        item_variant: subscriptionData.billing_cycle,
        quantity: 1,
        price: subscriptionData.value
      }]
    });
  },

  // Track trial start
  trackTrialStart: function(trialData) {
    gtag('event', 'begin_checkout', {
      currency: 'USD',
      value: trialData.trial_value || 0,
      items: [{
        item_id: trialData.plan_id,
        item_name: trialData.plan_name,
        item_category: 'trial',
        quantity: 1,
        price: trialData.trial_value || 0
      }]
    });
  },

  // Track subscription upgrade
  trackUpgrade: function(upgradeData) {
    gtag('event', 'purchase', {
      transaction_id: upgradeData.transaction_id,
      value: upgradeData.upgrade_value,
      currency: 'USD',
      items: [{
        item_id: upgradeData.new_plan_id,
        item_name: upgradeData.new_plan_name,
        item_category: 'upgrade',
        item_variant: upgradeData.billing_cycle,
        quantity: 1,
        price: upgradeData.upgrade_value
      }]
    });
  },

  // Track feature usage
  trackFeatureUsage: function(featureData) {
    gtag('event', 'select_content', {
      content_type: 'feature',
      content_id: featureData.feature_id,
      custom_parameter_3: featureData.feature_name
    });
  }
};
"""
        return ecommerce_config
    
    def generate_custom_dimensions_config(self) -> Dict[str, Any]:
        """Generate custom dimensions configuration"""
        return {
            "custom_dimensions": {
                "user_type": {
                    "parameter_name": "user_type",
                    "display_name": "User Type",
                    "description": "Type of user (free, trial, paid)",
                    "scope": "USER"
                },
                "subscription_tier": {
                    "parameter_name": "subscription_tier", 
                    "display_name": "Subscription Tier",
                    "description": "User's subscription tier (basic, premium, enterprise)",
                    "scope": "USER"
                },
                "feature_usage": {
                    "parameter_name": "feature_usage",
                    "display_name": "Feature Usage",
                    "description": "Features used by the user",
                    "scope": "EVENT"
                },
                "agent_interaction": {
                    "parameter_name": "agent_interaction",
                    "display_name": "Agent Interaction",
                    "description": "AI agent interactions",
                    "scope": "EVENT"
                },
                "data_source": {
                    "parameter_name": "data_source",
                    "display_name": "Data Source",
                    "description": "Source of uploaded data",
                    "scope": "EVENT"
                }
            },
            "custom_metrics": {
                "session_duration": {
                    "parameter_name": "session_duration_minutes",
                    "display_name": "Session Duration (Minutes)",
                    "description": "Duration of user session in minutes",
                    "measurement_unit": "MINUTES"
                },
                "api_calls": {
                    "parameter_name": "api_calls_count",
                    "display_name": "API Calls Count",
                    "description": "Number of API calls made",
                    "measurement_unit": "COUNT"
                },
                "data_processed": {
                    "parameter_name": "data_processed_mb",
                    "display_name": "Data Processed (MB)",
                    "description": "Amount of data processed in MB",
                    "measurement_unit": "MEGABYTES"
                }
            }
        }
    
    def create_next_js_integration(self) -> str:
        """Create Next.js specific integration code"""
        nextjs_integration = f"""
// pages/_app.tsx or app/layout.tsx
import {{ useEffect }} from 'react';
import {{ useRouter }} from 'next/router';

// Google Analytics tracking ID
const GA_TRACKING_ID = '{self.measurement_id}';

// Initialize Google Analytics
export const initGA = () => {{
  if (typeof window !== 'undefined') {{
    window.gtag('config', GA_TRACKING_ID, {{
      page_title: document.title,
      page_location: window.location.href,
    }});
  }}
}};

// Track page views
export const trackPageView = (url: string) => {{
  if (typeof window !== 'undefined') {{
    window.gtag('config', GA_TRACKING_ID, {{
      page_path: url,
    }});
  }}
}};

// Track custom events
export const trackEvent = (action: string, parameters: any = {{}}) => {{
  if (typeof window !== 'undefined') {{
    window.gtag('event', action, {{
      event_category: 'ScrollIntel',
      event_label: parameters.label || '',
      value: parameters.value || 0,
      ...parameters
    }});
  }}
}};

// Hook for tracking route changes
export const useGoogleAnalytics = () => {{
  const router = useRouter();

  useEffect(() => {{
    const handleRouteChange = (url: string) => {{
      trackPageView(url);
    }};

    router.events.on('routeChangeComplete', handleRouteChange);
    return () => {{
      router.events.off('routeChangeComplete', handleRouteChange);
    }};
  }}, [router.events]);
}};

// Component for Google Analytics
export const GoogleAnalytics = () => {{
  useEffect(() => {{
    initGA();
  }}, []);

  return null;
}};
"""
        return nextjs_integration
    
    def generate_privacy_compliant_config(self) -> str:
        """Generate privacy-compliant configuration"""
        privacy_config = f"""
// Privacy-compliant Google Analytics configuration
window.ScrollIntelPrivacy = {{
  // Cookie consent management
  cookieConsent: false,
  
  // Initialize with privacy settings
  init: function() {{
    gtag('config', '{self.measurement_id}', {{
      // Privacy settings
      anonymize_ip: true,
      allow_google_signals: false,
      allow_ad_personalization_signals: false,
      
      // Cookie settings
      cookie_expires: 63072000, // 2 years
      cookie_update: true,
      cookie_flags: 'SameSite=Strict;Secure',
      
      // Data retention
      storage: 'none', // Disable all storage until consent
    }});
  }},
  
  // Enable tracking after consent
  enableTracking: function() {{
    this.cookieConsent = true;
    gtag('config', '{self.measurement_id}', {{
      storage: 'granted',
      analytics_storage: 'granted',
      functionality_storage: 'granted'
    }});
  }},
  
  // Disable tracking
  disableTracking: function() {{
    this.cookieConsent = false;
    gtag('config', '{self.measurement_id}', {{
      storage: 'denied',
      analytics_storage: 'denied',
      functionality_storage: 'denied'
    }});
  }},
  
  // Check consent status
  hasConsent: function() {{
    return this.cookieConsent;
  }}
}};

// GDPR compliance helper
window.ScrollIntelGDPR = {{
  // Request data deletion
  requestDataDeletion: function(userId) {{
    // This would integrate with GA4 Data Deletion API
    console.log('Data deletion requested for user:', userId);
    // Implementation would call your backend to handle GA4 data deletion
  }},
  
  // Export user data
  exportUserData: function(userId) {{
    // This would integrate with GA4 Reporting API
    console.log('Data export requested for user:', userId);
    // Implementation would call your backend to export user data
  }}
}};
"""
        return privacy_config
    
    async def setup_conversion_tracking(self) -> Dict[str, Any]:
        """Setup conversion tracking configuration"""
        conversions = {
            "subscription_created": {
                "event_name": "purchase",
                "parameters": {
                    "currency": "USD",
                    "transaction_id": "required",
                    "value": "required",
                    "items": "required"
                }
            },
            "trial_started": {
                "event_name": "begin_checkout",
                "parameters": {
                    "currency": "USD",
                    "value": 0,
                    "items": "required"
                }
            },
            "user_signup": {
                "event_name": "sign_up",
                "parameters": {
                    "method": "email"
                }
            },
            "feature_activated": {
                "event_name": "select_content",
                "parameters": {
                    "content_type": "feature",
                    "content_id": "required"
                }
            },
            "data_uploaded": {
                "event_name": "file_download",
                "parameters": {
                    "file_extension": "csv",
                    "file_name": "required",
                    "link_url": "required"
                }
            }
        }
        
        return {
            "conversion_events": conversions,
            "setup_instructions": [
                "1. Configure conversion events in GA4 interface",
                "2. Set up Enhanced Conversions if collecting email/phone",
                "3. Link to Google Ads for conversion import",
                "4. Set up conversion goals and attribution models",
                "5. Configure audience definitions for remarketing"
            ]
        }
    
    def create_dashboard_config(self) -> Dict[str, Any]:
        """Create GA4 dashboard configuration"""
        return {
            "dashboard_config": {
                "key_metrics": [
                    "active_users",
                    "new_users", 
                    "sessions",
                    "bounce_rate",
                    "average_session_duration",
                    "pages_per_session",
                    "conversions",
                    "conversion_rate"
                ],
                "custom_reports": [
                    {
                        "name": "ScrollIntel User Engagement",
                        "metrics": ["active_users", "engagement_rate", "engaged_sessions"],
                        "dimensions": ["user_type", "subscription_tier", "feature_usage"]
                    },
                    {
                        "name": "Feature Usage Analysis", 
                        "metrics": ["event_count", "unique_events"],
                        "dimensions": ["event_name", "feature_usage", "user_type"]
                    },
                    {
                        "name": "Conversion Funnel",
                        "metrics": ["conversions", "conversion_rate"],
                        "dimensions": ["conversion_event", "traffic_source", "medium"]
                    }
                ],
                "audiences": [
                    {
                        "name": "High Value Users",
                        "conditions": [
                            "subscription_tier = premium OR enterprise",
                            "session_duration > 300 seconds"
                        ]
                    },
                    {
                        "name": "Trial Users",
                        "conditions": [
                            "user_type = trial",
                            "days_since_first_visit <= 14"
                        ]
                    },
                    {
                        "name": "Feature Power Users",
                        "conditions": [
                            "feature_usage_count >= 10",
                            "active_users_28_day >= 1"
                        ]
                    }
                ]
            }
        }


async def main():
    """Main setup function"""
    # Configuration
    config = {
        "measurement_id": os.getenv("GA4_MEASUREMENT_ID", "G-XXXXXXXXXX"),
        "api_secret": os.getenv("GA4_API_SECRET", "your_api_secret"),
        "tracking_id": os.getenv("GA_TRACKING_ID", "UA-XXXXXXXXX-X"),  # Legacy
        "debug_mode": os.getenv("GA_DEBUG_MODE", "false").lower() == "true"
    }
    
    if not config["measurement_id"] or config["measurement_id"] == "G-XXXXXXXXXX":
        logger.error("Please set GA4_MEASUREMENT_ID environment variable")
        return
    
    if not config["api_secret"] or config["api_secret"] == "your_api_secret":
        logger.error("Please set GA4_API_SECRET environment variable")
        return
    
    # Initialize integrator
    integrator = GoogleAnalyticsIntegrator(config)
    
    # Validate configuration
    logger.info("Validating Google Analytics configuration...")
    is_valid = await integrator.validate_measurement_protocol()
    
    if not is_valid:
        logger.error("Google Analytics configuration validation failed")
        return
    
    # Generate configuration files
    logger.info("Generating Google Analytics configuration files...")
    
    # Create output directory
    output_dir = "generated_analytics_config"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate gtag configuration
    gtag_config = integrator.setup_gtag_config()
    with open(f"{output_dir}/gtag_config.html", "w") as f:
        f.write(gtag_config)
    
    # Generate ecommerce configuration
    ecommerce_config = integrator.generate_enhanced_ecommerce_config()
    with open(f"{output_dir}/ecommerce_config.js", "w") as f:
        f.write(ecommerce_config)
    
    # Generate Next.js integration
    nextjs_integration = integrator.create_next_js_integration()
    with open(f"{output_dir}/nextjs_integration.tsx", "w") as f:
        f.write(nextjs_integration)
    
    # Generate privacy configuration
    privacy_config = integrator.generate_privacy_compliant_config()
    with open(f"{output_dir}/privacy_config.js", "w") as f:
        f.write(privacy_config)
    
    # Generate custom dimensions config
    custom_config = integrator.generate_custom_dimensions_config()
    with open(f"{output_dir}/custom_dimensions.json", "w") as f:
        json.dump(custom_config, f, indent=2)
    
    # Generate conversion tracking config
    conversion_config = await integrator.setup_conversion_tracking()
    with open(f"{output_dir}/conversion_tracking.json", "w") as f:
        json.dump(conversion_config, f, indent=2)
    
    # Generate dashboard config
    dashboard_config = integrator.create_dashboard_config()
    with open(f"{output_dir}/dashboard_config.json", "w") as f:
        json.dump(dashboard_config, f, indent=2)
    
    # Test event sending
    logger.info("Testing event sending...")
    test_success = await integrator.send_measurement_protocol_event(
        client_id="test_client_123",
        event_name="scrollintel_setup_complete",
        event_parameters={
            "setup_timestamp": datetime.utcnow().isoformat(),
            "configuration_version": "1.0.0"
        }
    )
    
    if test_success:
        logger.info("✅ Google Analytics setup completed successfully!")
        logger.info(f"📁 Configuration files generated in: {output_dir}/")
        logger.info("📋 Next steps:")
        logger.info("1. Copy gtag_config.html content to your HTML head section")
        logger.info("2. Add ecommerce_config.js to your application")
        logger.info("3. Integrate nextjs_integration.tsx if using Next.js")
        logger.info("4. Configure custom dimensions in GA4 interface")
        logger.info("5. Set up conversion events and goals")
        logger.info("6. Test tracking in GA4 Real-time reports")
    else:
        logger.error("❌ Google Analytics setup failed during testing")


if __name__ == "__main__":
    asyncio.run(main())