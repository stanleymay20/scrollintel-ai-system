import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { DndProvider } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';
import '@testing-library/jest-dom';
import { DashboardCustomizer } from '@/components/dashboard/dashboard-customizer';

// Mock the drag and drop backend for testing
const TestDndProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <DndProvider backend={HTML5Backend}>
    {children}
  </DndProvider>
);

const mockTemplate = {
  id: 'test-template',
  name: 'Test Dashboard',
  description: 'A test dashboard template',
  widgets: [
    {
      id: 'widget-1',
      type: 'metrics_grid',
      title: 'Test Metrics',
      position: { x: 0, y: 0, width: 4, height: 3 },
      config: { metrics: ['cpu', 'memory'] },
      dataSource: 'system_metrics'
    },
    {
      id: 'widget-2',
      type: 'line_chart',
      title: 'Test Chart',
      position: { x: 4, y: 0, width: 8, height: 4 },
      config: { timeRange: '24h' },
      dataSource: 'chart_data'
    }
  ],
  layoutConfig: {
    gridSize: 12,
    rowHeight: 60
  }
};

const mockProps = {
  template: mockTemplate,
  onSave: jest.fn(),
  onPreview: jest.fn(),
  onShare: jest.fn()
};

describe('DashboardCustomizer', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders the dashboard customizer with template name', () => {
    render(
      <TestDndProvider>
        <DashboardCustomizer {...mockProps} />
      </TestDndProvider>
    );

    expect(screen.getByText('Dashboard Customizer')).toBeInTheDocument();
    expect(screen.getByText('Test Dashboard')).toBeInTheDocument();
  });

  it('displays widget palette with available widget types', () => {
    render(
      <TestDndProvider>
        <DashboardCustomizer {...mockProps} />
      </TestDndProvider>
    );

    expect(screen.getByText('Widget Palette')).toBeInTheDocument();
    expect(screen.getByText('Metrics Grid')).toBeInTheDocument();
    expect(screen.getByText('Line Chart')).toBeInTheDocument();
    expect(screen.getByText('Bar Chart')).toBeInTheDocument();
    expect(screen.getByText('Pie Chart')).toBeInTheDocument();
  });

  it('renders existing widgets on the canvas', () => {
    render(
      <TestDndProvider>
        <DashboardCustomizer {...mockProps} />
      </TestDndProvider>
    );

    expect(screen.getByText('Test Metrics')).toBeInTheDocument();
    expect(screen.getByText('Test Chart')).toBeInTheDocument();
    expect(screen.getByText('metrics_grid')).toBeInTheDocument();
    expect(screen.getByText('line_chart')).toBeInTheDocument();
  });

  it('shows properties panel when no widget is selected', () => {
    render(
      <TestDndProvider>
        <DashboardCustomizer {...mockProps} />
      </TestDndProvider>
    );

    expect(screen.getByText('Properties')).toBeInTheDocument();
    expect(screen.getByText('Select a widget to edit its properties')).toBeInTheDocument();
  });

  it('shows widget properties when a widget is selected', () => {
    render(
      <TestDndProvider>
        <DashboardCustomizer {...mockProps} />
      </TestDndProvider>
    );

    // Click on a widget to select it
    const widget = screen.getByText('Test Metrics').closest('div');
    fireEvent.click(widget!);

    // Properties panel should show widget details
    expect(screen.getByDisplayValue('Test Metrics')).toBeInTheDocument();
    expect(screen.getByDisplayValue('metrics_grid')).toBeInTheDocument();
    expect(screen.getByDisplayValue('system_metrics')).toBeInTheDocument();
  });

  it('allows editing widget properties', async () => {
    render(
      <TestDndProvider>
        <DashboardCustomizer {...mockProps} />
      </TestDndProvider>
    );

    // Select a widget
    const widget = screen.getByText('Test Metrics').closest('div');
    fireEvent.click(widget!);

    // Edit the title
    const titleInput = screen.getByDisplayValue('Test Metrics');
    fireEvent.change(titleInput, { target: { value: 'Updated Metrics' } });

    await waitFor(() => {
      expect(screen.getByText('Updated Metrics')).toBeInTheDocument();
    });
  });

  it('allows editing widget dimensions', async () => {
    render(
      <TestDndProvider>
        <DashboardCustomizer {...mockProps} />
      </TestDndProvider>
    );

    // Select a widget
    const widget = screen.getByText('Test Metrics').closest('div');
    fireEvent.click(widget!);

    // Edit width
    const widthInput = screen.getByDisplayValue('4');
    fireEvent.change(widthInput, { target: { value: '6' } });

    await waitFor(() => {
      expect(screen.getByDisplayValue('6')).toBeInTheDocument();
    });
  });

  it('allows changing widget type', async () => {
    render(
      <TestDndProvider>
        <DashboardCustomizer {...mockProps} />
      </TestDndProvider>
    );

    // Select a widget
    const widget = screen.getByText('Test Metrics').closest('div');
    fireEvent.click(widget!);

    // Change widget type
    const typeSelect = screen.getByDisplayValue('metrics_grid');
    fireEvent.change(typeSelect, { target: { value: 'bar_chart' } });

    await waitFor(() => {
      expect(screen.getByDisplayValue('bar_chart')).toBeInTheDocument();
    });
  });

  it('allows editing widget configuration JSON', async () => {
    render(
      <TestDndProvider>
        <DashboardCustomizer {...mockProps} />
      </TestDndProvider>
    );

    // Select a widget
    const widget = screen.getByText('Test Metrics').closest('div');
    fireEvent.click(widget!);

    // Edit configuration
    const configTextarea = screen.getByDisplayValue(/"metrics":\s*\[\s*"cpu",\s*"memory"\s*\]/);
    const newConfig = JSON.stringify({ metrics: ['cpu', 'memory', 'disk'] }, null, 2);
    fireEvent.change(configTextarea, { target: { value: newConfig } });

    // Configuration should be updated (though we can't easily test the internal state)
    expect(configTextarea).toHaveValue(newConfig);
  });

  it('has undo functionality', () => {
    render(
      <TestDndProvider>
        <DashboardCustomizer {...mockProps} />
      </TestDndProvider>
    );

    const undoButton = screen.getByRole('button', { name: /undo/i });
    expect(undoButton).toBeInTheDocument();
    expect(undoButton).toBeDisabled(); // Should be disabled initially
  });

  it('has redo functionality', () => {
    render(
      <TestDndProvider>
        <DashboardCustomizer {...mockProps} />
      </TestDndProvider>
    );

    const redoButton = screen.getByRole('button', { name: /redo/i });
    expect(redoButton).toBeInTheDocument();
    expect(redoButton).toBeDisabled(); // Should be disabled initially
  });

  it('has preview functionality', () => {
    render(
      <TestDndProvider>
        <DashboardCustomizer {...mockProps} />
      </TestDndProvider>
    );

    const previewButton = screen.getByRole('button', { name: /preview/i });
    fireEvent.click(previewButton);

    expect(mockProps.onPreview).toHaveBeenCalledTimes(1);
  });

  it('has share functionality', () => {
    render(
      <TestDndProvider>
        <DashboardCustomizer {...mockProps} />
      </TestDndProvider>
    );

    const shareButton = screen.getByRole('button', { name: /share/i });
    fireEvent.click(shareButton);

    expect(mockProps.onShare).toHaveBeenCalledTimes(1);
  });

  it('has save functionality', () => {
    render(
      <TestDndProvider>
        <DashboardCustomizer {...mockProps} />
      </TestDndProvider>
    );

    const saveButton = screen.getByRole('button', { name: /save/i });
    fireEvent.click(saveButton);

    expect(mockProps.onSave).toHaveBeenCalledTimes(1);
    expect(mockProps.onSave).toHaveBeenCalledWith(
      expect.objectContaining({
        id: 'test-template',
        name: 'Test Dashboard',
        widgets: expect.any(Array)
      })
    );
  });

  it('allows deleting widgets', async () => {
    render(
      <TestDndProvider>
        <DashboardCustomizer {...mockProps} />
      </TestDndProvider>
    );

    // Select a widget
    const widget = screen.getByText('Test Metrics').closest('div');
    fireEvent.click(widget!);

    // Click delete button
    const deleteButton = screen.getByRole('button', { name: '' }); // Trash icon button
    fireEvent.click(deleteButton);

    await waitFor(() => {
      expect(screen.queryByText('Test Metrics')).not.toBeInTheDocument();
    });
  });

  it('shows grid background on canvas', () => {
    render(
      <TestDndProvider>
        <DashboardCustomizer {...mockProps} />
      </TestDndProvider>
    );

    // Check for canvas with grid styling
    const canvas = screen.getByRole('generic', { hidden: true });
    expect(canvas).toHaveStyle({
      backgroundImage: expect.stringContaining('linear-gradient')
    });
  });

  it('displays widget dimensions in grid units', () => {
    render(
      <TestDndProvider>
        <DashboardCustomizer {...mockProps} />
      </TestDndProvider>
    );

    expect(screen.getByText('4x3 grid units')).toBeInTheDocument();
    expect(screen.getByText('8x4 grid units')).toBeInTheDocument();
  });

  it('enables undo after making changes', async () => {
    render(
      <TestDndProvider>
        <DashboardCustomizer {...mockProps} />
      </TestDndProvider>
    );

    // Select a widget and make a change
    const widget = screen.getByText('Test Metrics').closest('div');
    fireEvent.click(widget!);

    const titleInput = screen.getByDisplayValue('Test Metrics');
    fireEvent.change(titleInput, { target: { value: 'Changed Title' } });

    await waitFor(() => {
      const undoButton = screen.getByRole('button', { name: /undo/i });
      expect(undoButton).not.toBeDisabled();
    });
  });

  it('handles invalid JSON in configuration gracefully', () => {
    render(
      <TestDndProvider>
        <DashboardCustomizer {...mockProps} />
      </TestDndProvider>
    );

    // Select a widget
    const widget = screen.getByText('Test Metrics').closest('div');
    fireEvent.click(widget!);

    // Enter invalid JSON
    const configTextarea = screen.getByDisplayValue(/"metrics":\s*\[\s*"cpu",\s*"memory"\s*\]/);
    fireEvent.change(configTextarea, { target: { value: 'invalid json' } });

    // Should not crash the application
    expect(configTextarea).toHaveValue('invalid json');
  });

  it('shows drop zone when dragging widgets', () => {
    // This test would require more complex drag and drop simulation
    // For now, we'll just check that the canvas exists and can receive drops
    render(
      <TestDndProvider>
        <DashboardCustomizer {...mockProps} />
      </TestDndProvider>
    );

    // Canvas should be present and ready for drops
    const canvas = document.querySelector('[style*="background-image"]');
    expect(canvas).toBeInTheDocument();
  });
});

describe('DashboardCustomizer Widget Palette', () => {
  it('displays all widget types', () => {
    render(
      <TestDndProvider>
        <DashboardCustomizer {...mockProps} />
      </TestDndProvider>
    );

    const expectedWidgetTypes = [
      'Metrics Grid',
      'Line Chart',
      'Bar Chart',
      'Pie Chart',
      'Gauge Chart',
      'Status Grid',
      'Alert List',
      'Kanban Board',
      'Area Chart',
      'Data Table'
    ];

    expectedWidgetTypes.forEach(widgetType => {
      expect(screen.getByText(widgetType)).toBeInTheDocument();
    });
  });

  it('shows widget icons', () => {
    render(
      <TestDndProvider>
        <DashboardCustomizer {...mockProps} />
      </TestDndProvider>
    );

    // Check for emoji icons (simplified test)
    expect(screen.getByText('📊')).toBeInTheDocument();
    expect(screen.getByText('📈')).toBeInTheDocument();
    expect(screen.getByText('🥧')).toBeInTheDocument();
  });
});

describe('DashboardCustomizer Properties Panel', () => {
  it('shows all editable properties for selected widget', () => {
    render(
      <TestDndProvider>
        <DashboardCustomizer {...mockProps} />
      </TestDndProvider>
    );

    // Select a widget
    const widget = screen.getByText('Test Metrics').closest('div');
    fireEvent.click(widget!);

    // Check for all property fields
    expect(screen.getByLabelText('Title')).toBeInTheDocument();
    expect(screen.getByLabelText('Type')).toBeInTheDocument();
    expect(screen.getByLabelText('Width')).toBeInTheDocument();
    expect(screen.getByLabelText('Height')).toBeInTheDocument();
    expect(screen.getByLabelText('Data Source')).toBeInTheDocument();
    expect(screen.getByLabelText('Configuration')).toBeInTheDocument();
  });

  it('validates width and height inputs', () => {
    render(
      <TestDndProvider>
        <DashboardCustomizer {...mockProps} />
      </TestDndProvider>
    );

    // Select a widget
    const widget = screen.getByText('Test Metrics').closest('div');
    fireEvent.click(widget!);

    // Width input should have min/max constraints
    const widthInput = screen.getByLabelText('Width') as HTMLInputElement;
    expect(widthInput.min).toBe('1');
    expect(widthInput.max).toBe('12');

    // Height input should have min constraint
    const heightInput = screen.getByLabelText('Height') as HTMLInputElement;
    expect(heightInput.min).toBe('1');
  });
});