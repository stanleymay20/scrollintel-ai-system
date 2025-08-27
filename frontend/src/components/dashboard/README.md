# Dashboard Customization Interface

A comprehensive React-based dashboard customization system that provides drag-and-drop functionality, real-time preview, and responsive design capabilities for creating advanced analytics dashboards.

## Components Overview

### 1. DashboardCustomizationInterface
The main orchestrating component that provides the complete dashboard building experience.

**Features:**
- Drag-and-drop widget placement
- Real-time preview with live data
- Responsive design for mobile, tablet, and desktop
- Template management (save, load, export, import)
- History management with undo/redo
- WebSocket integration for live updates

### 2. DashboardBuilder
Core building blocks for dashboard customization including:
- **WidgetPalette**: Categorized widget library with search
- **DropZone**: Grid-based drag-and-drop canvas
- **DraggableWidget**: Interactive widget components
- **WidgetPropertiesPanel**: Configuration interface
- **TemplateSelector**: Template browsing and selection

### 3. TemplatePreview
Real-time dashboard preview component with:
- Live data visualization using Recharts
- Multiple chart types (line, bar, pie, area, gauge)
- Responsive breakpoint testing
- Fullscreen preview mode
- WebSocket-powered live updates

### 4. MobileDashboardBuilder
Specialized mobile-first dashboard builder featuring:
- Mobile device frame preview
- Widget priority management
- Touch-optimized layouts
- Responsive configuration per breakpoint
- Mobile-specific optimizations

## Widget Types

### Metrics & KPIs
- **Metric Card**: Single value display with trend indicators
- **KPI Grid**: Multiple metrics in organized grid
- **Progress Bar**: Goal progress visualization
- **Gauge Chart**: Circular progress indicators

### Charts & Visualizations
- **Line Chart**: Time series data visualization
- **Bar Chart**: Categorical data comparison
- **Pie Chart**: Proportional data display
- **Area Chart**: Filled line charts for trends
- **Scatter Plot**: Correlation analysis
- **Heatmap**: Matrix data visualization

### Tables & Lists
- **Data Table**: Sortable, filterable tabular data
- **Summary Table**: Aggregated data display
- **Ranking List**: Ordered data presentation

### Advanced Widgets
- **ROI Calculator**: Interactive financial analysis
- **Forecast Chart**: Predictive analytics display
- **Alert Panel**: System notifications
- **Custom HTML**: Flexible content embedding

## Data Sources

The system supports multiple data sources:
- **ROI Calculator**: Financial metrics and calculations
- **Performance Monitor**: System and operational metrics
- **Cost Tracker**: Expense and budget data
- **Deployment Tracker**: Release and deployment status
- **User Analytics**: Behavioral and usage data
- **System Metrics**: Technical performance data
- **Business Intelligence**: Strategic insights

## Responsive Design

### Breakpoints
- **Mobile**: < 768px (4-column grid)
- **Tablet**: 768px - 1024px (8-column grid)
- **Desktop**: > 1024px (12-column grid)

### Mobile Optimizations
- Touch-friendly interactions
- Stacked widget layouts
- Priority-based widget ordering
- Swipe navigation support
- Battery-efficient updates

## Usage Examples

### Basic Dashboard Creation
```tsx
import DashboardCustomizationInterface from './components/dashboard/dashboard-customization-interface';

function App() {
  const handleSave = (template) => {
    // Save template to backend
    console.log('Saving template:', template);
  };

  return (
    <DashboardCustomizationInterface
      userId="user-123"
      userRole="admin"
      onSave={handleSave}
      onPublish={(template) => console.log('Publishing:', template)}
      onShare={(template) => console.log('Sharing:', template)}
    />
  );
}
```

### Template Preview Only
```tsx
import { TemplatePreview } from './components/dashboard/template-preview';

function PreviewPage({ template }) {
  return (
    <TemplatePreview
      template={template}
      isLivePreview={true}
      currentBreakpoint="desktop"
    />
  );
}
```

### Mobile-Specific Builder
```tsx
import { MobileDashboardBuilder } from './components/dashboard/mobile-dashboard-builder';

function MobileBuilder({ template, onUpdate }) {
  return (
    <MobileDashboardBuilder
      template={template}
      onTemplateUpdate={onUpdate}
      onSave={() => console.log('Saving mobile config')}
      onPreview={() => console.log('Previewing mobile layout')}
    />
  );
}
```

## API Integration

### WebSocket Events
The system listens for real-time updates via WebSocket:

```javascript
// Dashboard data updates
socket.on('dashboard_message', (message) => {
  switch (message.type) {
    case 'dashboard_update':
      // Update dashboard data
      break;
    case 'metrics_update':
      // Update specific metrics
      break;
    case 'alert':
      // Show system alerts
      break;
  }
});

// Request updates
socket.emit('request_dashboard_update', { dashboardId, userId });
```

### REST API Endpoints
```javascript
// Save template
POST /api/dashboard/templates
{
  "template": { /* template object */ }
}

// Load template
GET /api/dashboard/templates/:id

// Get dashboard data
GET /api/dashboard/:id/data

// Get widget data
GET /api/dashboard/widgets/:id/data
```

## Customization

### Adding New Widget Types
1. Define widget configuration in `WIDGET_CATEGORIES`
2. Add rendering logic in `TemplatePreview`
3. Create widget-specific property panels
4. Add mobile optimization rules

### Custom Themes
```javascript
const customTheme = {
  primary_color: '#your-primary',
  secondary_color: '#your-secondary',
  background_color: '#your-background',
  text_color: '#your-text'
};
```

### Data Source Integration
1. Add data source to `DATA_SOURCES` array
2. Implement data fetching logic
3. Add WebSocket event handlers
4. Create mock data generators for preview

## Testing

The components include comprehensive test coverage:
- Unit tests for individual components
- Integration tests for complete workflows
- Responsive design testing
- Error handling verification
- WebSocket integration testing

Run tests with:
```bash
npm test -- --testPathPattern=dashboard
```

## Performance Considerations

- **Lazy Loading**: Widgets load data on demand
- **Virtualization**: Large datasets use virtual scrolling
- **Debounced Updates**: Real-time updates are throttled
- **Memoization**: React.memo used for expensive renders
- **WebSocket Optimization**: Selective event subscriptions

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Dependencies

- React 18+
- react-dnd (drag and drop)
- recharts (data visualization)
- lucide-react (icons)
- tailwindcss (styling)

## Contributing

1. Follow the existing component structure
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Ensure responsive design compatibility
5. Test across all supported breakpoints