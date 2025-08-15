"""
Progressive Web App (PWA) capabilities for ScrollIntel.
Provides full offline support, service worker management, and native app-like experience.
"""

import asyncio
import logging
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import hashlib
import gzip
import base64

from .offline_first_architecture import offline_first_architecture

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache strategies for PWA resources."""
    CACHE_FIRST = "cache_first"
    NETWORK_FIRST = "network_first"
    CACHE_ONLY = "cache_only"
    NETWORK_ONLY = "network_only"
    STALE_WHILE_REVALIDATE = "stale_while_revalidate"


class ResourceType(Enum):
    """Types of resources for caching."""
    STATIC = "static"
    DYNAMIC = "dynamic"
    API = "api"
    MEDIA = "media"
    DOCUMENT = "document"


@dataclass
class CacheRule:
    """Caching rule for PWA resources."""
    pattern: str
    strategy: CacheStrategy
    resource_type: ResourceType
    max_age_hours: int = 24
    max_entries: int = 100
    network_timeout_ms: int = 3000
    cache_name: str = "default"


@dataclass
class PWAManifest:
    """PWA manifest configuration."""
    name: str = "ScrollIntel"
    short_name: str = "ScrollIntel"
    description: str = "AI-Powered Data Intelligence Platform"
    start_url: str = "/"
    display: str = "standalone"
    theme_color: str = "#1f2937"
    background_color: str = "#ffffff"
    orientation: str = "portrait-primary"
    icons: List[Dict[str, Any]] = field(default_factory=list)
    categories: List[str] = field(default_factory=lambda: ["productivity", "business", "analytics"])
    lang: str = "en"
    dir: str = "ltr"


@dataclass
class InstallPrompt:
    """Install prompt configuration."""
    enabled: bool = True
    delay_days: int = 3
    min_visits: int = 5
    show_after_engagement: bool = True
    custom_message: str = "Install ScrollIntel for the best experience!"


class ProgressiveWebApp:
    """Progressive Web App manager for ScrollIntel."""
    
    def __init__(self, static_path: str = "frontend/public"):
        self.static_path = Path(static_path)
        
        # PWA configuration
        self.manifest = PWAManifest()
        self.install_prompt = InstallPrompt()
        
        # Cache management
        self.cache_rules: List[CacheRule] = []
        self.cache_storage: Dict[str, Dict[str, Any]] = {}
        self.cache_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Service worker management
        self.service_worker_version = "1.0.0"
        self.service_worker_cache_name = f"scrollintel-v{self.service_worker_version}"
        
        # Background sync
        self.background_sync_tags: List[str] = []
        self.sync_handlers: Dict[str, Callable] = {}
        
        # Push notifications
        self.push_subscription: Optional[Dict[str, Any]] = None
        self.notification_handlers: Dict[str, Callable] = {}
        
        # Installation tracking
        self.install_events: List[Dict[str, Any]] = []
        self.user_engagement: Dict[str, Any] = {
            'visits': 0,
            'time_spent': 0,
            'last_visit': None,
            'first_visit': None
        }
        
        # Performance metrics
        self.performance_metrics = {
            'cache_hits': 0,
            'cache_misses': 0,
            'network_requests': 0,
            'offline_requests': 0,
            'background_syncs': 0,
            'push_notifications_sent': 0,
            'install_prompts_shown': 0,
            'successful_installs': 0
        }
        
        # Initialize PWA
        self._setup_default_cache_rules()
        self._setup_default_manifest()
        self._setup_default_icons()
    
    def _setup_default_cache_rules(self):
        """Setup default caching rules for PWA."""
        
        # Static assets - cache first with long expiry
        self.add_cache_rule(CacheRule(
            pattern=r"\.(js|css|woff2?|png|jpg|jpeg|gif|svg|ico)$",
            strategy=CacheStrategy.CACHE_FIRST,
            resource_type=ResourceType.STATIC,
            max_age_hours=168,  # 1 week
            max_entries=200,
            cache_name="static-assets"
        ))
        
        # HTML documents - network first with fallback
        self.add_cache_rule(CacheRule(
            pattern=r"\.html?$",
            strategy=CacheStrategy.NETWORK_FIRST,
            resource_type=ResourceType.DOCUMENT,
            max_age_hours=1,
            max_entries=50,
            network_timeout_ms=2000,
            cache_name="documents"
        ))
        
        # API calls - stale while revalidate
        self.add_cache_rule(CacheRule(
            pattern=r"/api/",
            strategy=CacheStrategy.STALE_WHILE_REVALIDATE,
            resource_type=ResourceType.API,
            max_age_hours=1,
            max_entries=100,
            network_timeout_ms=5000,
            cache_name="api-cache"
        ))
        
        # Media files - cache first with medium expiry
        self.add_cache_rule(CacheRule(
            pattern=r"\.(mp4|webm|mp3|wav|pdf)$",
            strategy=CacheStrategy.CACHE_FIRST,
            resource_type=ResourceType.MEDIA,
            max_age_hours=72,  # 3 days
            max_entries=50,
            cache_name="media"
        ))
        
        # Dynamic content - network first
        self.add_cache_rule(CacheRule(
            pattern=r"/dashboard|/analytics|/reports",
            strategy=CacheStrategy.NETWORK_FIRST,
            resource_type=ResourceType.DYNAMIC,
            max_age_hours=0.5,  # 30 minutes
            max_entries=30,
            network_timeout_ms=3000,
            cache_name="dynamic"
        ))
    
    def _setup_default_manifest(self):
        """Setup default PWA manifest."""
        self.manifest = PWAManifest(
            name="ScrollIntel - AI Data Intelligence",
            short_name="ScrollIntel",
            description="Transform your data into actionable insights with AI-powered analytics",
            start_url="/",
            display="standalone",
            theme_color="#1f2937",
            background_color="#ffffff",
            orientation="any",
            categories=["productivity", "business", "analytics", "artificial-intelligence"],
            lang="en",
            dir="ltr"
        )
    
    def _setup_default_icons(self):
        """Setup default PWA icons."""
        icon_sizes = [72, 96, 128, 144, 152, 192, 384, 512]
        
        self.manifest.icons = [
            {
                "src": f"/icons/icon-{size}x{size}.png",
                "sizes": f"{size}x{size}",
                "type": "image/png",
                "purpose": "any maskable" if size >= 192 else "any"
            }
            for size in icon_sizes
        ]
        
        # Add vector icon
        self.manifest.icons.append({
            "src": "/icons/icon.svg",
            "sizes": "any",
            "type": "image/svg+xml",
            "purpose": "any"
        })
    
    def add_cache_rule(self, rule: CacheRule):
        """Add a caching rule."""
        self.cache_rules.append(rule)
        logger.info(f"Added cache rule: {rule.pattern} -> {rule.strategy.value}")
    
    def generate_manifest(self) -> Dict[str, Any]:
        """Generate PWA manifest JSON."""
        manifest_dict = {
            "name": self.manifest.name,
            "short_name": self.manifest.short_name,
            "description": self.manifest.description,
            "start_url": self.manifest.start_url,
            "display": self.manifest.display,
            "theme_color": self.manifest.theme_color,
            "background_color": self.manifest.background_color,
            "orientation": self.manifest.orientation,
            "icons": self.manifest.icons,
            "categories": self.manifest.categories,
            "lang": self.manifest.lang,
            "dir": self.manifest.dir,
            "scope": "/",
            "prefer_related_applications": False,
            "shortcuts": [
                {
                    "name": "Dashboard",
                    "short_name": "Dashboard",
                    "description": "View your analytics dashboard",
                    "url": "/dashboard",
                    "icons": [{"src": "/icons/dashboard-96x96.png", "sizes": "96x96"}]
                },
                {
                    "name": "Upload Data",
                    "short_name": "Upload",
                    "description": "Upload new data files",
                    "url": "/upload",
                    "icons": [{"src": "/icons/upload-96x96.png", "sizes": "96x96"}]
                },
                {
                    "name": "AI Chat",
                    "short_name": "Chat",
                    "description": "Chat with AI assistant",
                    "url": "/chat",
                    "icons": [{"src": "/icons/chat-96x96.png", "sizes": "96x96"}]
                }
            ]
        }
        
        return manifest_dict
    
    def generate_service_worker(self) -> str:
        """Generate service worker JavaScript code."""
        sw_code = f"""
// ScrollIntel Service Worker v{self.service_worker_version}
const CACHE_NAME = '{self.service_worker_cache_name}';
const OFFLINE_URL = '/offline.html';

// Cache rules
const CACHE_RULES = {json.dumps([
    {
        'pattern': rule.pattern,
        'strategy': rule.strategy.value,
        'resource_type': rule.resource_type.value,
        'max_age_hours': rule.max_age_hours,
        'max_entries': rule.max_entries,
        'network_timeout_ms': rule.network_timeout_ms,
        'cache_name': rule.cache_name
    }
    for rule in self.cache_rules
], indent=2)};

// Static assets to precache
const PRECACHE_ASSETS = [
    '/',
    '/offline.html',
    '/manifest.json',
    '/icons/icon-192x192.png',
    '/static/css/main.css',
    '/static/js/main.js'
];

// Install event - precache static assets
self.addEventListener('install', event => {{
    console.log('Service Worker installing...');
    
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => {{
                console.log('Precaching static assets');
                return cache.addAll(PRECACHE_ASSETS);
            }})
            .then(() => {{
                console.log('Service Worker installed successfully');
                return self.skipWaiting();
            }})
            .catch(error => {{
                console.error('Service Worker installation failed:', error);
            }})
    );
}});

// Activate event - clean up old caches
self.addEventListener('activate', event => {{
    console.log('Service Worker activating...');
    
    event.waitUntil(
        caches.keys()
            .then(cacheNames => {{
                return Promise.all(
                    cacheNames.map(cacheName => {{
                        if (cacheName !== CACHE_NAME && cacheName.startsWith('scrollintel-')) {{
                            console.log('Deleting old cache:', cacheName);
                            return caches.delete(cacheName);
                        }
                    }})
                );
            }})
            .then(() => {{
                console.log('Service Worker activated');
                return self.clients.claim();
            }})
    );
}});

// Fetch event - handle requests with caching strategies
self.addEventListener('fetch', event => {{
    const request = event.request;
    const url = new URL(request.url);
    
    // Skip non-GET requests
    if (request.method !== 'GET') {{
        return;
    }}
    
    // Skip chrome-extension and other non-http requests
    if (!url.protocol.startsWith('http')) {{
        return;
    }}
    
    // Find matching cache rule
    const rule = findMatchingRule(request.url);
    
    if (rule) {{
        event.respondWith(handleRequest(request, rule));
    }} else {{
        // Default strategy for unmatched requests
        event.respondWith(
            fetch(request).catch(() => {{
                // Return offline page for navigation requests
                if (request.destination === 'document') {{
                    return caches.match(OFFLINE_URL);
                }}
                return new Response('Offline', {{ status: 503 }});
            }})
        );
    }}
}});

// Background sync event
self.addEventListener('sync', event => {{
    console.log('Background sync triggered:', event.tag);
    
    if (event.tag === 'background-sync-data') {{
        event.waitUntil(syncOfflineData());
    }} else if (event.tag === 'background-sync-analytics') {{
        event.waitUntil(syncAnalyticsData());
    }}
}});

// Push notification event
self.addEventListener('push', event => {{
    console.log('Push notification received');
    
    const options = {{
        body: 'You have new insights available!',
        icon: '/icons/icon-192x192.png',
        badge: '/icons/badge-72x72.png',
        vibrate: [100, 50, 100],
        data: {{
            dateOfArrival: Date.now(),
            primaryKey: 1
        }},
        actions: [
            {{
                action: 'explore',
                title: 'View Dashboard',
                icon: '/icons/dashboard-96x96.png'
            }},
            {{
                action: 'close',
                title: 'Close',
                icon: '/icons/close-96x96.png'
            }}
        ]
    }};
    
    if (event.data) {{
        const payload = event.data.json();
        options.body = payload.body || options.body;
        options.data = {{ ...options.data, ...payload.data }};
    }}
    
    event.waitUntil(
        self.registration.showNotification('ScrollIntel', options)
    );
}});

// Notification click event
self.addEventListener('notificationclick', event => {{
    console.log('Notification clicked:', event.action);
    
    event.notification.close();
    
    if (event.action === 'explore') {{
        event.waitUntil(
            clients.openWindow('/dashboard')
        );
    }} else if (event.action === 'close') {{
        // Just close the notification
    }} else {{
        // Default action - open the app
        event.waitUntil(
            clients.openWindow('/')
        );
    }}
}});

// Helper functions
function findMatchingRule(url) {{
    for (const rule of CACHE_RULES) {{
        const regex = new RegExp(rule.pattern);
        if (regex.test(url)) {{
            return rule;
        }}
    }}
    return null;
}}

async function handleRequest(request, rule) {{
    const cacheName = rule.cache_name || CACHE_NAME;
    
    switch (rule.strategy) {{
        case 'cache_first':
            return cacheFirst(request, cacheName, rule);
        case 'network_first':
            return networkFirst(request, cacheName, rule);
        case 'cache_only':
            return cacheOnly(request, cacheName);
        case 'network_only':
            return networkOnly(request);
        case 'stale_while_revalidate':
            return staleWhileRevalidate(request, cacheName, rule);
        default:
            return fetch(request);
    }}
}}

async function cacheFirst(request, cacheName, rule) {{
    const cache = await caches.open(cacheName);
    const cachedResponse = await cache.match(request);
    
    if (cachedResponse) {{
        // Check if cache is still valid
        const cacheTime = new Date(cachedResponse.headers.get('sw-cache-time') || 0);
        const maxAge = rule.max_age_hours * 60 * 60 * 1000;
        
        if (Date.now() - cacheTime.getTime() < maxAge) {{
            return cachedResponse;
        }}
    }}
    
    try {{
        const networkResponse = await fetch(request);
        if (networkResponse.ok) {{
            const responseToCache = networkResponse.clone();
            responseToCache.headers.set('sw-cache-time', new Date().toISOString());
            await cache.put(request, responseToCache);
        }}
        return networkResponse;
    }} catch (error) {{
        return cachedResponse || new Response('Offline', {{ status: 503 }});
    }}
}}

async function networkFirst(request, cacheName, rule) {{
    const cache = await caches.open(cacheName);
    
    try {{
        const networkResponse = await Promise.race([
            fetch(request),
            new Promise((_, reject) => 
                setTimeout(() => reject(new Error('timeout')), rule.network_timeout_ms)
            )
        ]);
        
        if (networkResponse.ok) {{
            const responseToCache = networkResponse.clone();
            responseToCache.headers.set('sw-cache-time', new Date().toISOString());
            await cache.put(request, responseToCache);
        }}
        return networkResponse;
    }} catch (error) {{
        const cachedResponse = await cache.match(request);
        return cachedResponse || new Response('Offline', {{ status: 503 }});
    }}
}}

async function cacheOnly(request, cacheName) {{
    const cache = await caches.open(cacheName);
    return await cache.match(request) || new Response('Not in cache', {{ status: 404 }});
}}

async function networkOnly(request) {{
    return await fetch(request);
}}

async function staleWhileRevalidate(request, cacheName, rule) {{
    const cache = await caches.open(cacheName);
    const cachedResponse = await cache.match(request);
    
    // Start network request in background
    const networkResponsePromise = fetch(request).then(response => {{
        if (response.ok) {{
            const responseToCache = response.clone();
            responseToCache.headers.set('sw-cache-time', new Date().toISOString());
            cache.put(request, responseToCache);
        }}
        return response;
    }}).catch(() => null);
    
    // Return cached response immediately if available
    if (cachedResponse) {{
        return cachedResponse;
    }}
    
    // Wait for network response if no cache
    return await networkResponsePromise || new Response('Offline', {{ status: 503 }});
}}

async function syncOfflineData() {{
    console.log('Syncing offline data...');
    
    try {{
        // This would integrate with the offline data manager
        const response = await fetch('/api/sync/offline-data', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify({{ action: 'sync_all' }})
        }});
        
        if (response.ok) {{
            console.log('Offline data sync completed');
        }} else {{
            console.error('Offline data sync failed:', response.status);
        }}
    }} catch (error) {{
        console.error('Offline data sync error:', error);
    }}
}}

async function syncAnalyticsData() {{
    console.log('Syncing analytics data...');
    
    try {{
        const response = await fetch('/api/sync/analytics', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify({{ action: 'sync_analytics' }})
        }});
        
        if (response.ok) {{
            console.log('Analytics sync completed');
        }} else {{
            console.error('Analytics sync failed:', response.status);
        }}
    }} catch (error) {{
        console.error('Analytics sync error:', error);
    }}
}}

console.log('ScrollIntel Service Worker loaded');
"""
        return sw_code
    
    def generate_offline_page(self) -> str:
        """Generate offline fallback page HTML."""
        offline_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ScrollIntel - Offline</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
        }
        .offline-container {
            text-align: center;
            padding: 2rem;
            max-width: 500px;
        }
        .offline-icon {
            font-size: 4rem;
            margin-bottom: 1rem;
        }
        .offline-title {
            font-size: 2rem;
            margin-bottom: 1rem;
            font-weight: 300;
        }
        .offline-message {
            font-size: 1.1rem;
            margin-bottom: 2rem;
            opacity: 0.9;
            line-height: 1.6;
        }
        .offline-actions {
            display: flex;
            gap: 1rem;
            justify-content: center;
            flex-wrap: wrap;
        }
        .btn {
            padding: 0.75rem 1.5rem;
            border: 2px solid white;
            background: transparent;
            color: white;
            text-decoration: none;
            border-radius: 25px;
            font-weight: 500;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        .btn:hover {
            background: white;
            color: #667eea;
        }
        .btn-primary {
            background: white;
            color: #667eea;
        }
        .btn-primary:hover {
            background: transparent;
            color: white;
        }
        .offline-features {
            margin-top: 3rem;
            text-align: left;
        }
        .feature-list {
            list-style: none;
            padding: 0;
        }
        .feature-list li {
            padding: 0.5rem 0;
            opacity: 0.8;
        }
        .feature-list li:before {
            content: "✓ ";
            color: #4ade80;
            font-weight: bold;
        }
        @media (max-width: 600px) {
            .offline-container {
                padding: 1rem;
            }
            .offline-title {
                font-size: 1.5rem;
            }
            .offline-actions {
                flex-direction: column;
                align-items: center;
            }
        }
    </style>
</head>
<body>
    <div class="offline-container">
        <div class="offline-icon">📱</div>
        <h1 class="offline-title">You're Offline</h1>
        <p class="offline-message">
            Don't worry! ScrollIntel works offline too. You can still access your cached data and continue working.
        </p>
        
        <div class="offline-actions">
            <button class="btn btn-primary" onclick="window.location.reload()">
                Try Again
            </button>
            <a href="/dashboard" class="btn">
                View Dashboard
            </a>
        </div>
        
        <div class="offline-features">
            <h3>Available Offline:</h3>
            <ul class="feature-list">
                <li>View cached dashboards and reports</li>
                <li>Access previously loaded data</li>
                <li>Create and edit visualizations</li>
                <li>Save work locally (syncs when online)</li>
                <li>Use AI chat with cached responses</li>
            </ul>
        </div>
    </div>
    
    <script>
        // Check for connection and reload when back online
        window.addEventListener('online', () => {
            window.location.reload();
        });
        
        // Show connection status
        if (navigator.onLine) {
            document.querySelector('.offline-message').innerHTML = 
                'Connection restored! <a href="/" style="color: #4ade80;">Return to ScrollIntel</a>';
        }
    </script>
</body>
</html>
"""
        return offline_html
    
    async def handle_install_prompt(self, user_id: str) -> Dict[str, Any]:
        """Handle PWA install prompt logic."""
        if not self.install_prompt.enabled:
            return {'show_prompt': False, 'reason': 'disabled'}
        
        # Check user engagement
        engagement = self.user_engagement
        
        # Check minimum visits
        if engagement['visits'] < self.install_prompt.min_visits:
            return {
                'show_prompt': False, 
                'reason': 'insufficient_visits',
                'visits': engagement['visits'],
                'required': self.install_prompt.min_visits
            }
        
        # Check delay period
        if engagement['first_visit']:
            first_visit = datetime.fromisoformat(engagement['first_visit'])
            days_since_first = (datetime.now() - first_visit).days
            
            if days_since_first < self.install_prompt.delay_days:
                return {
                    'show_prompt': False,
                    'reason': 'delay_period',
                    'days_since_first': days_since_first,
                    'required_days': self.install_prompt.delay_days
                }
        
        # Check if already prompted recently
        recent_prompts = [
            event for event in self.install_events
            if event.get('type') == 'prompt_shown' and
            event.get('user_id') == user_id and
            (datetime.now() - datetime.fromisoformat(event['timestamp'])).days < 7
        ]
        
        if recent_prompts:
            return {
                'show_prompt': False,
                'reason': 'recently_prompted',
                'last_prompt': recent_prompts[-1]['timestamp']
            }
        
        # Show prompt
        self.performance_metrics['install_prompts_shown'] += 1
        
        install_event = {
            'type': 'prompt_shown',
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'engagement': engagement.copy()
        }
        self.install_events.append(install_event)
        
        return {
            'show_prompt': True,
            'message': self.install_prompt.custom_message,
            'event_id': len(self.install_events) - 1
        }
    
    async def handle_install_result(self, user_id: str, event_id: int, 
                                  result: str, user_choice: str = None) -> Dict[str, Any]:
        """Handle PWA install result."""
        if event_id >= len(self.install_events):
            return {'error': 'Invalid event ID'}
        
        install_event = self.install_events[event_id]
        install_event['result'] = result
        install_event['user_choice'] = user_choice
        install_event['completed_at'] = datetime.now().isoformat()
        
        if result == 'accepted':
            self.performance_metrics['successful_installs'] += 1
            logger.info(f"PWA installed by user {user_id}")
        
        return {
            'recorded': True,
            'result': result,
            'total_installs': self.performance_metrics['successful_installs']
        }
    
    def track_user_engagement(self, user_id: str, action: str, duration: float = 0):
        """Track user engagement for install prompt logic."""
        now = datetime.now().isoformat()
        
        if action == 'visit':
            self.user_engagement['visits'] += 1
            self.user_engagement['last_visit'] = now
            
            if not self.user_engagement['first_visit']:
                self.user_engagement['first_visit'] = now
        
        elif action == 'time_spent':
            self.user_engagement['time_spent'] += duration
        
        logger.debug(f"User engagement tracked: {action} for user {user_id}")
    
    def register_background_sync(self, tag: str, handler: Callable):
        """Register background sync handler."""
        self.background_sync_tags.append(tag)
        self.sync_handlers[tag] = handler
        logger.info(f"Registered background sync handler: {tag}")
    
    def register_notification_handler(self, event_type: str, handler: Callable):
        """Register push notification handler."""
        self.notification_handlers[event_type] = handler
        logger.info(f"Registered notification handler: {event_type}")
    
    async def send_push_notification(self, user_id: str, title: str, body: str,
                                   data: Optional[Dict[str, Any]] = None,
                                   actions: Optional[List[Dict[str, str]]] = None) -> bool:
        """Send push notification to user."""
        if not self.push_subscription:
            logger.warning("No push subscription available")
            return False
        
        try:
            # This would integrate with a push service like FCM or Web Push
            # For now, simulate sending
            notification_data = {
                'title': title,
                'body': body,
                'data': data or {},
                'actions': actions or [],
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id
            }
            
            logger.info(f"Push notification sent to {user_id}: {title}")
            self.performance_metrics['push_notifications_sent'] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Push notification failed: {e}")
            return False
    
    async def trigger_background_sync(self, tag: str, data: Optional[Dict[str, Any]] = None) -> bool:
        """Trigger background sync."""
        if tag not in self.background_sync_tags:
            logger.warning(f"Unknown background sync tag: {tag}")
            return False
        
        handler = self.sync_handlers.get(tag)
        if not handler:
            logger.warning(f"No handler for background sync tag: {tag}")
            return False
        
        try:
            await handler(data)
            self.performance_metrics['background_syncs'] += 1
            logger.info(f"Background sync completed: {tag}")
            return True
            
        except Exception as e:
            logger.error(f"Background sync failed for {tag}: {e}")
            return False
    
    def get_pwa_status(self) -> Dict[str, Any]:
        """Get PWA status and metrics."""
        return {
            'service_worker_version': self.service_worker_version,
            'cache_name': self.service_worker_cache_name,
            'cache_rules_count': len(self.cache_rules),
            'background_sync_tags': self.background_sync_tags,
            'install_prompt_enabled': self.install_prompt.enabled,
            'push_subscription_active': self.push_subscription is not None,
            'user_engagement': self.user_engagement.copy(),
            'performance_metrics': self.performance_metrics.copy(),
            'recent_install_events': [
                event for event in self.install_events[-10:]  # Last 10 events
            ]
        }
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information."""
        cache_info = {}
        
        for rule in self.cache_rules:
            cache_info[rule.cache_name] = {
                'pattern': rule.pattern,
                'strategy': rule.strategy.value,
                'resource_type': rule.resource_type.value,
                'max_age_hours': rule.max_age_hours,
                'max_entries': rule.max_entries,
                'network_timeout_ms': rule.network_timeout_ms
            }
        
        return {
            'cache_rules': cache_info,
            'total_rules': len(self.cache_rules),
            'cache_storage_size': len(self.cache_storage),
            'performance': {
                'cache_hits': self.performance_metrics['cache_hits'],
                'cache_misses': self.performance_metrics['cache_misses'],
                'hit_rate': (
                    self.performance_metrics['cache_hits'] / 
                    max(self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses'], 1)
                )
            }
        }
    
    async def update_cache_strategy(self, pattern: str, strategy: CacheStrategy) -> bool:
        """Update cache strategy for a pattern."""
        for rule in self.cache_rules:
            if rule.pattern == pattern:
                rule.strategy = strategy
                logger.info(f"Updated cache strategy for {pattern}: {strategy.value}")
                return True
        
        logger.warning(f"Cache rule not found for pattern: {pattern}")
        return False
    
    def set_push_subscription(self, subscription: Dict[str, Any]):
        """Set push notification subscription."""
        self.push_subscription = subscription
        logger.info("Push subscription updated")
    
    def update_manifest(self, updates: Dict[str, Any]):
        """Update PWA manifest."""
        for key, value in updates.items():
            if hasattr(self.manifest, key):
                setattr(self.manifest, key, value)
                logger.info(f"Updated manifest {key}: {value}")
    
    def save_to_files(self):
        """Save PWA files to disk."""
        try:
            # Create directories
            self.static_path.mkdir(parents=True, exist_ok=True)
            
            # Save manifest
            manifest_path = self.static_path / "manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(self.generate_manifest(), f, indent=2)
            
            # Save service worker
            sw_path = self.static_path / "sw.js"
            with open(sw_path, 'w') as f:
                f.write(self.generate_service_worker())
            
            # Save offline page
            offline_path = self.static_path / "offline.html"
            with open(offline_path, 'w') as f:
                f.write(self.generate_offline_page())
            
            logger.info(f"PWA files saved to {self.static_path}")
            
        except Exception as e:
            logger.error(f"Failed to save PWA files: {e}")


# Global instance
progressive_web_app = ProgressiveWebApp()


# Integration with offline architecture
async def setup_pwa_integration():
    """Setup PWA integration with offline architecture."""
    
    # Register background sync handlers
    progressive_web_app.register_background_sync(
        'background-sync-data',
        lambda data: offline_first_architecture.force_sync_all()
    )
    
    progressive_web_app.register_background_sync(
        'background-sync-analytics',
        lambda data: sync_analytics_data()
    )
    
    # Register notification handlers
    progressive_web_app.register_notification_handler(
        'data_sync_complete',
        lambda data: send_sync_notification(data)
    )
    
    progressive_web_app.register_notification_handler(
        'new_insights',
        lambda data: send_insights_notification(data)
    )
    
    logger.info("PWA integration setup completed")


async def sync_analytics_data():
    """Sync analytics data in background."""
    try:
        # This would sync analytics data
        logger.info("Analytics data sync completed")
    except Exception as e:
        logger.error(f"Analytics sync failed: {e}")


async def send_sync_notification(data: Dict[str, Any]):
    """Send notification when sync completes."""
    await progressive_web_app.send_push_notification(
        user_id=data.get('user_id', 'unknown'),
        title="Data Sync Complete",
        body="Your offline changes have been synchronized.",
        data={'type': 'sync_complete'}
    )


async def send_insights_notification(data: Dict[str, Any]):
    """Send notification for new insights."""
    await progressive_web_app.send_push_notification(
        user_id=data.get('user_id', 'unknown'),
        title="New Insights Available",
        body="ScrollIntel has discovered new patterns in your data.",
        data={'type': 'new_insights', 'insights_count': data.get('count', 0)},
        actions=[
            {'action': 'view', 'title': 'View Insights'},
            {'action': 'dismiss', 'title': 'Dismiss'}
        ]
    )


# Initialize PWA integration
asyncio.create_task(setup_pwa_integration())