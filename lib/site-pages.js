import fs from 'node:fs';
import path from 'node:path';

// Trusted, version-controlled research page templates. No user HTML is accepted.
export const pages = ['index', 'rink', 'tracking', 'tracking-replay', 'vlm', 'sam3', 'sam3-improvements', 'roboflow', 'new-games', 'overhead', 'shots'];
export function readPage(name) {
  if (!pages.includes(name)) throw new Error('Unknown page');
  const html = fs.readFileSync(path.join(process.cwd(), 'web', `${name}.html`), 'utf8');
  const body = html.match(/<body([^>]*)>([\s\S]*?)<\/body>/i);
  return {
    title: html.match(/<title>(.*?)<\/title>/i)[1],
    bodyClass: body[1].match(/class="([^"]+)"/)?.[1] || '',
    markup: body[2].replace(/<script\b[^>]*>[\s\S]*?<\/script>/gi, ''),
    styles: [...html.matchAll(/<link[^>]+href="([^"]+\.css)"[^>]*>/g)].map(m => m[1]),
    scripts: [...html.matchAll(/<script[^>]+src="([^"]+)"[^>]*>/g)].map(m => m[1]),
  };
}
