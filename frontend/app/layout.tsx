import type { Metadata, Viewport } from 'next';
import { Vazirmatn } from 'next/font/google';

import { AppShell } from '@/components/layout/app-shell';

import './globals.css';
import { Providers } from './providers';

const vazir = Vazirmatn({
  subsets: ['arabic', 'latin'],
  variable: '--font-vazir',
  display: 'swap',
  fallback: ['system-ui', 'sans-serif'],
});

export const metadata: Metadata = {
  title: 'نوو پالس | هوش پیش‌بینی تقاضا',
  description: 'پیش‌بینی، تحلیل و شبیه‌سازی تقاضای بازار اقامت',
  applicationName: 'Novo Pulse',
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  themeColor: '#3563e9',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="fa" dir="rtl" className={vazir.variable} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: `(function(){try{var t=localStorage.getItem('novo-pulse-theme');if(t!=='dark')t='light';var e=document.documentElement;e.dataset.theme=t;e.classList.toggle('dark',t==='dark');e.style.colorScheme=t}catch(e){}})()` }} />
      </head>
      <body className="min-h-screen">
        <Providers>
          <AppShell>{children}</AppShell>
        </Providers>
      </body>
    </html>
  );
}
