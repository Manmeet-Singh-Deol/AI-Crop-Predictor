/**
 * AgroAI Progressive Web App (PWA) Service Worker
 * Enables 100% offline functionality, caching the application shell,
 * in-browser ONNX vision model, and agronomist databases.
 */

const CACHE_NAME = 'agroai-cache-v1.5.0';
const OFFLINE_URLS = [
    '/',
    '/index.html',
    '/styles.css',
    '/app.js',
    '/offline_db.js',
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
                console.warn('[ServiceWorker] Non-critical asset precache note:', err);
            });
        }).then(() => self.skipWaiting())
    );
});

// Activate Event: Immediately delete all stale caches and claim clients
self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys().then((keyList) => {
            return Promise.all(
                keyList.map((key) => {
                    if (key !== CACHE_NAME) {
                        console.log('[ServiceWorker] Purging stale cache:', key);
                        return caches.delete(key);
                    }
                })
            );
        }).then(() => self.clients.claim())
    );
});

// Fetch Event Strategy
self.addEventListener('fetch', (event) => {
    const url = new URL(event.request.url);

    // 1. Dynamic weather & chat APIs -> Network with offline fallback
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

    // 2. Large ONNX model, Icons, and CDN libraries -> Cache-First for speed & zero network usage
    if (url.pathname.endsWith('.onnx') || url.pathname.endsWith('.png') || url.hostname.includes('cdn') || url.hostname.includes('cdnjs')) {
        event.respondWith(
            caches.match(event.request).then((cached) => {
                if (cached) return cached;
                return fetch(event.request).then((netRes) => {
                    if (netRes && netRes.status === 200) {
                        const clone = netRes.clone();
                        caches.open(CACHE_NAME).then((c) => c.put(event.request, clone));
                    }
                    return netRes;
                });
            })
        );
        return;
    }

    // 3. App Shell (HTML, JS, CSS, /api/samples) -> Network-First (Fresh on reload, Cache fallback when offline)
    event.respondWith(
        fetch(event.request).then((netRes) => {
            if (netRes && netRes.status === 200 && event.request.method === 'GET') {
                const clone = netRes.clone();
                caches.open(CACHE_NAME).then((c) => c.put(event.request, clone));
            }
            return netRes;
        }).catch(() => {
            return caches.match(event.request).then((cached) => {
                if (cached) return cached;
                if (event.request.mode === 'navigate') {
                    return caches.match('/index.html');
                }
            });
        })
    );
});
