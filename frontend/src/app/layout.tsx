import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import CookieConsentBanner from '../components/legal/cookie-consent-banner'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'ScrollIntel v4.0+ - AI-CTO Platform',
  description: 'The world\'s most advanced sovereign AI-CTO replacement platform',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        {children}
        <CookieConsentBanner />
      </body>
    </html>
  )
}