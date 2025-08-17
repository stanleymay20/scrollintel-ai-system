'use client'

import React, { useState, useEffect } from 'react'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { oneDark, oneLight } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { Copy, Check, Play, Download } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { useTheme } from 'next-themes'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import rehypeRaw from 'rehype-raw'
import 'katex/dist/katex.min.css'

interface MessageProcessorProps {
  content: string
  contentType: 'text' | 'markdown' | 'code' | 'mixed'
  isStreaming?: boolean
}

interface CodeBlockProps {
  children?: React.ReactNode
  className?: string
  inline?: boolean
}

export function MessageProcessor({ content, contentType, isStreaming }: MessageProcessorProps) {
  const { theme } = useTheme()
  const [copiedBlocks, setCopiedBlocks] = useState<Set<string>>(new Set())

  const handleCopyCode = async (code: string, blockId: string) => {
    try {
      await navigator.clipboard.writeText(code)
      setCopiedBlocks(prev => new Set(prev).add(blockId))
      setTimeout(() => {
        setCopiedBlocks(prev => {
          const newSet = new Set(prev)
          newSet.delete(blockId)
          return newSet
        })
      }, 2000)
    } catch (error) {
      console.error('Failed to copy code:', error)
    }
  }

  const CodeBlock = ({ children, className, inline }: CodeBlockProps) => {
    const match = /language-(\w+)/.exec(className || '')
    const language = match ? match[1] : 'text'
    const codeString = String(children)
    const blockId = `${language}-${codeString.slice(0, 20)}`
    const isCopied = copiedBlocks.has(blockId)

    if (inline) {
      return (
        <code className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded text-sm font-mono">
          {children}
        </code>
      )
    }

    return (
      <div className="relative group my-4">
        <div className="flex items-center justify-between bg-gray-100 dark:bg-gray-800 px-4 py-2 rounded-t-lg border-b">
          <div className="flex items-center space-x-2">
            <Badge variant="secondary" className="text-xs">
              {language}
            </Badge>
            <span className="text-xs text-gray-600 dark:text-gray-400">
              {codeString.split('\n').length} lines
            </span>
          </div>
          
          <div className="flex items-center space-x-1 opacity-0 group-hover:opacity-100 transition-opacity">
            <Button
              variant="ghost"
              size="sm"
              className="h-7 px-2"
              onClick={() => handleCopyCode(codeString, blockId)}
            >
              {isCopied ? (
                <Check className="h-3 w-3 text-green-500" />
              ) : (
                <Copy className="h-3 w-3" />
              )}
            </Button>
            
            {(language === 'javascript' || language === 'python' || language === 'bash') && (
              <Button
                variant="ghost"
                size="sm"
                className="h-7 px-2"
                onClick={() => {
                  // This would integrate with a code execution service
                  console.log('Execute code:', codeString)
                }}
              >
                <Play className="h-3 w-3" />
              </Button>
            )}
            
            <Button
              variant="ghost"
              size="sm"
              className="h-7 px-2"
              onClick={() => {
                const blob = new Blob([codeString], { type: 'text/plain' })
                const url = URL.createObjectURL(blob)
                const a = document.createElement('a')
                a.href = url
                a.download = `code.${language}`
                a.click()
                URL.revokeObjectURL(url)
              }}
            >
              <Download className="h-3 w-3" />
            </Button>
          </div>
        </div>
        
        <SyntaxHighlighter
          style={theme === 'dark' ? oneDark : oneLight}
          language={language}
          customStyle={{
            margin: 0,
            borderTopLeftRadius: 0,
            borderTopRightRadius: 0,
            borderBottomLeftRadius: '0.5rem',
            borderBottomRightRadius: '0.5rem',
          }}
          showLineNumbers={codeString.split('\n').length > 5}
          wrapLines={true}
          wrapLongLines={true}
        >
          {codeString}
        </SyntaxHighlighter>
      </div>
    )
  }

  const TableComponent = ({ children }: { children: React.ReactNode }) => (
    <div className="overflow-x-auto my-4">
      <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
        {children}
      </table>
    </div>
  )

  const components = {
    code: CodeBlock,
    table: TableComponent,
    thead: ({ children }: { children: React.ReactNode }) => (
      <thead className="bg-gray-50 dark:bg-gray-800">{children}</thead>
    ),
    tbody: ({ children }: { children: React.ReactNode }) => (
      <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
        {children}
      </tbody>
    ),
    th: ({ children }: { children: React.ReactNode }) => (
      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
        {children}
      </th>
    ),
    td: ({ children }: { children: React.ReactNode }) => (
      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 dark:text-gray-100">
        {children}
      </td>
    ),
    blockquote: ({ children }: { children: React.ReactNode }) => (
      <blockquote className="border-l-4 border-blue-500 pl-4 py-2 my-4 bg-blue-50 dark:bg-blue-900/20 italic">
        {children}
      </blockquote>
    ),
    h1: ({ children }: { children: React.ReactNode }) => (
      <h1 className="text-2xl font-bold mt-6 mb-4 text-gray-900 dark:text-gray-100">
        {children}
      </h1>
    ),
    h2: ({ children }: { children: React.ReactNode }) => (
      <h2 className="text-xl font-semibold mt-5 mb-3 text-gray-900 dark:text-gray-100">
        {children}
      </h2>
    ),
    h3: ({ children }: { children: React.ReactNode }) => (
      <h3 className="text-lg font-medium mt-4 mb-2 text-gray-900 dark:text-gray-100">
        {children}
      </h3>
    ),
    ul: ({ children }: { children: React.ReactNode }) => (
      <ul className="list-disc list-inside my-3 space-y-1 text-gray-700 dark:text-gray-300">
        {children}
      </ul>
    ),
    ol: ({ children }: { children: React.ReactNode }) => (
      <ol className="list-decimal list-inside my-3 space-y-1 text-gray-700 dark:text-gray-300">
        {children}
      </ol>
    ),
    li: ({ children }: { children: React.ReactNode }) => (
      <li className="ml-4">{children}</li>
    ),
    p: ({ children }: { children: React.ReactNode }) => (
      <p className="my-2 text-gray-700 dark:text-gray-300 leading-relaxed">
        {children}
      </p>
    ),
    a: ({ href, children }: { href?: string; children: React.ReactNode }) => (
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className="text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 underline"
      >
        {children}
      </a>
    ),
    strong: ({ children }: { children: React.ReactNode }) => (
      <strong className="font-semibold text-gray-900 dark:text-gray-100">
        {children}
      </strong>
    ),
    em: ({ children }: { children: React.ReactNode }) => (
      <em className="italic text-gray-700 dark:text-gray-300">{children}</em>
    ),
    hr: () => (
      <hr className="my-6 border-gray-200 dark:border-gray-700" />
    ),
  }

  const renderContent = () => {
    if (contentType === 'text') {
      return (
        <div className="whitespace-pre-wrap text-gray-700 dark:text-gray-300 leading-relaxed">
          {content}
          {isStreaming && (
            <span className="inline-block w-2 h-5 bg-blue-500 animate-pulse ml-1" />
          )}
        </div>
      )
    }

    if (contentType === 'code') {
      return (
        <CodeBlock className="language-text">
          {content}
        </CodeBlock>
      )
    }

    // For markdown and mixed content
    return (
      <div className="prose prose-sm max-w-none dark:prose-invert">
        <ReactMarkdown
          components={components}
          remarkPlugins={[remarkGfm, remarkMath]}
          rehypePlugins={[rehypeKatex, rehypeRaw]}
        >
          {content}
        </ReactMarkdown>
        {isStreaming && (
          <span className="inline-block w-2 h-5 bg-blue-500 animate-pulse ml-1" />
        )}
      </div>
    )
  }

  return (
    <div className="message-content">
      {renderContent()}
    </div>
  )
}

// Streaming text effect component
export function StreamingText({ text, speed = 50 }: { text: string; speed?: number }) {
  const [displayedText, setDisplayedText] = useState('')
  const [currentIndex, setCurrentIndex] = useState(0)

  useEffect(() => {
    if (currentIndex < text.length) {
      const timer = setTimeout(() => {
        setDisplayedText(prev => prev + text[currentIndex])
        setCurrentIndex(prev => prev + 1)
      }, speed)

      return () => clearTimeout(timer)
    }
  }, [currentIndex, text, speed])

  return (
    <span>
      {displayedText}
      {currentIndex < text.length && (
        <span className="inline-block w-2 h-5 bg-blue-500 animate-pulse ml-1" />
      )}
    </span>
  )
}