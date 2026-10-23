'use client';

import { useEffect } from 'react';

// Keep the validated video/canvas controllers while the App Router renders pages.
// Plain anchor navigation deliberately gives each controller a fresh document.
export default function PageScripts({ scripts, bodyClass }) {
  useEffect(() => {
    if (bodyClass) document.body.classList.add(...bodyClass.split(' '));
    window.sportScansScripts ??= new Map();
    async function load() {
      for (const src of scripts) {
        if (!window.sportScansScripts.has(src)) {
          window.sportScansScripts.set(src, new Promise((resolve, reject) => {
            const script = document.createElement('script');
            script.src = `/${src}`;
            script.onload = resolve;
            script.onerror = () => reject(new Error(`Could not load ${src}`));
            document.body.appendChild(script);
          }));
        }
        await window.sportScansScripts.get(src);
      }
    }
    load().catch(error => {
      console.error(error);
      const message = document.getElementById('app-error');
      if (message) { message.hidden = false; message.textContent = error.message; }
    });
  }, [scripts, bodyClass]);
  return null;
}
