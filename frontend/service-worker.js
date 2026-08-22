/**
 * AgroAI Progressive Web App (PWA) Service Worker
 * Enables 100% offline functionality, caching the application shell,
 * in-browser ONNX vision model, and agronomist databases.
 */

const CACHE_NAME = 'agroai-cache-v1.4.0';
const OFFLINE_URLS = [
    '/',
    '/index.html',
    '/styles.css',
    '/app.js',
    '/manifest.json',
    '/icon-192.png',
    '/icon-512.png',
    '/crop_disease_model.onnx',
    'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/ort.min.js',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css',
    'https://cdn.tailwindcss.com'
];

// Install Event: Precache all essential assets & ONNX model
self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME).then((cache) => {
            console.log('[ServiceWorker] Precaching offline application shell & ONNX model...');
            return cache.addAll(OFFLINE_URLS).catch((err) => {
                console.warn('[ServiceWorker] Some non-critical assets failed to precache:', err);
            });
        }).then(() => self.skipWaiting())
    );
});

// Activate Event: Cleanup stale caches
self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((keyList) => {
            return Promise.all(
                keyList.map((key) => {
                    if (key !== CACHE_NAME) {
                        console.log('[ServiceWorker] Removing old cache version:', key);
                        return caches.delete(key);
                    }
                })
            );
        }).then(() => self.clients.claim())
    );
});

// Fetch Event: Cache-first for model & assets, network-first for live dynamic APIs
self.addEventListener('fetch', (event) => {
    const url = new URL(event.request.url);

    // Dynamic weather / cloud API calls -> Network with graceful offline fallback
    if (url.pathname.startsWith('/api/weather-risk') || url.pathname.startsWith('/api/chat')) {
        event.respondWith(
            fetch(event.request).catch(() => {
                return new Response(
                    JSON.stringify({ offline: true, message: "Operating in Field Offline Mode" }),
                    { headers: { 'Content-Type': 'application/json' } }
                );
            })
        );
        return;
    }

    // Static assets, ONNX model & App Shell -> Cache-First Strategy
    event.respondWith(
        caches.match(event.request).then((cachedResponse) => {
            if (cachedResponse) {
                return cachedResponse;
            }
            return fetch(event.request).then((networkResponse) => {
                if (networkResponse && networkResponse.status === 200 && event.request.method === 'GET') {
                    const responseToCache = networkResponse.clone();
                    caches.open(CACHE_NAME).then((cache) => {
                        cache.put(event.request, responseToCache);
                    });
                }
                return networkResponse;
            }).catch(() => {
                // If offline and requesting root navigation, return cached index.html
                if (event.request.mode === 'navigate') {
                    return caches.match('/index.html');
                }
            });
        })
    );
});
