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
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="fa" dir="rtl" className={vazir.variable}>
      <body className="min-h-screen">
        <Providers>
          <AppShell>{children}</AppShell>
        </Providers>
      </body>
    </html>
  );
}
