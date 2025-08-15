# ScrollIntel Frontend v4.0+ ScrollSanctified HyperSovereign Edition™

This is the Next.js frontend for ScrollIntel, the world's most advanced AI-CTO replacement platform.

## Features

- **Modern Tech Stack**: Next.js 14, React 18, TypeScript, Tailwind CSS
- **ScrollPulse UI**: Custom design system with ShadCN UI components
- **Agent Dashboard**: Real-time agent status monitoring and interaction
- **Chat Interface**: Natural language interaction with AI agents
- **File Upload**: Drag-and-drop file upload with progress tracking
- **System Metrics**: Live system performance monitoring
- **Responsive Design**: Mobile-first responsive layout
- **Comprehensive Testing**: Jest and React Testing Library

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Installation

```bash
npm install
```

### Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the application.

### Testing

```bash
# Run tests
npm test

# Run tests in watch mode
npm run test:watch
```

### Building

```bash
npm run build
npm start
```

## Project Structure

```
frontend/
├── src/
│   ├── app/                 # Next.js app directory
│   │   ├── globals.css      # Global styles
│   │   ├── layout.tsx       # Root layout
│   │   └── page.tsx         # Main dashboard page
│   ├── components/          # React components
│   │   ├── ui/              # Base UI components
│   │   ├── dashboard/       # Dashboard-specific components
│   │   ├── chat/            # Chat interface components
│   │   ├── upload/          # File upload components
│   │   └── layout/          # Layout components
│   ├── lib/                 # Utility functions
│   ├── types/               # TypeScript type definitions
│   └── __tests__/           # Test files
├── public/                  # Static assets
└── package.json
```

## Key Components

### AgentStatusCard
Displays individual agent information including:
- Agent name and status
- Capabilities and metrics
- Interactive buttons for agent communication

### ChatInterface
Provides natural language interaction with agents:
- Message history display
- Real-time typing indicators
- Agent selection and context

### FileUploadComponent
Handles file uploads with:
- Drag-and-drop interface
- Progress tracking
- File type validation
- Error handling

### SystemMetricsCard
Shows real-time system metrics:
- CPU and memory usage
- Active agent count
- Request statistics
- Performance indicators

## Styling

The application uses:
- **Tailwind CSS** for utility-first styling
- **Custom ScrollIntel theme** with brand colors
- **ShadCN UI components** for consistent design
- **Responsive design** for all screen sizes

## API Integration

The frontend communicates with the ScrollIntel backend through:
- RESTful API endpoints
- Real-time WebSocket connections
- File upload handling
- Authentication and session management

## Testing

Comprehensive test coverage includes:
- Component unit tests
- Integration tests
- Utility function tests
- User interaction tests

## Deployment

The application is configured for deployment on:
- **Vercel** (recommended for Next.js)
- **Netlify**
- **Docker containers**
- **Static hosting**

## Environment Variables

Create a `.env.local` file:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Contributing

1. Follow the existing code style
2. Write tests for new components
3. Update documentation as needed
4. Ensure all tests pass before submitting

## License

ScrollSanctified™ - All rights reserved