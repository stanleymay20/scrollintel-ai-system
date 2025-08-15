'use client'

import { ChatInterface } from '@/components/chat/chat-interface'

export default function ChatPage() {
  return (
    <div className="flex-1 flex flex-col">
      <div className="border-b p-4">
        <h1 className="text-2xl font-bold">AI Chat Interface</h1>
        <p className="text-muted-foreground">Interact with ScrollIntel AI agents</p>
      </div>
      <div className="flex-1">
        <ChatInterface />
      </div>
    </div>
  )
}