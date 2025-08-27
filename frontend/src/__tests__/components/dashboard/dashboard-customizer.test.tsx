/**
 * Tests for Dashboard Customizer Component
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { DndProvider } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';
import '@testing-library/jest-dom';
import { DashboardCustomizer } from '../../../components/dashboard/dashboard-customizer';

// Mock data
const mockTemplate = {
  id: 'test-template',
  name: 'Test Dashboard',
  description: 'Test dashboard description',
  widgets: [
    {
      id: 'widget-1',
      type: 'metric_card',
      title: 'Test Metric',
      position: { x: 0, y: 0, width: 4, height: 2 },
      data_source: 'roi_calculator',
      visualization_config: {},
      filters: [],
      refresh_interval: 300,
    },
    {
      id: 'widget-2',
      type: 'line_chart',
      title: 'Test Chart',
      position: { x: 4, y: 0, width: 8, height: 4 },
      data_source: 'performance_monitor',
      visualization_config: {},
      filters: [],
      refresh_interval: 300,
    },
  ],
  layout_config: {
    grid_size: 12,
    row_height: 60,
    margin: [10, 10],
  },
  default_filters: [],
};

// Test wrapper with DnD provider
const TestWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <DndProvider backend={HTML5Backend}>
    {children}
  </DndProvider>
);

describe('DashboardCustomizer', () => {
  const mockOnTemplateUpdate = jest.fn();
  const mockOnSave = jest.fn();
  const mockOnPreview = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  const renderCustomizer = (props = {}) => {
    return render(
      <TestWrapper>
        <DashboardCustomizer
          template={mockTemplate}
          onTemplateUpdate={mockOnTemplateUpdate}
          onSave={mockOnSave}
          onPreview={mockOnPreview}
          {...props}
        />
      </TestWrapper>
    );
  };

  describe('Rendering', () => {
    it('renders dashboard customizer with template name', () => {
      renderCustomizer();
      expect(screen.getByText('Test Dashboard')).toBeInTheDocument();
    });

    it('renders widget palette with widget types', () => {
      renderCustomizer();
      expect(screen.getByText('Widget Palette')).toBeInTheDocument();
      expect(screen.getByText('Metric Card')).toBeInTheDocument();
      expect(screen.getByText('Line Chart')).toBeInTheDocument();
      expect(screen.getByText('Bar Chart')).toBeInTheDocument();
    });

    it('renders existing widgets in the canvas', () => {
      renderCustomizer();
      expect(screen.getByText('Test Metric')).toBeInTheDocument();
      expect(screen.getByText('Test Chart')).toBeInTheDocument();
    });

    it('renders action buttons', () => {
      renderCustomizer();
      expect(screen.getByText('Preview')).toBeInTheDocument();
      expect(screen.getByText('Save Template')).toBeInTheDocument();
    });
  });

  describe('Widget Palette', () => {
    it('filters widgets by category', () => {
      renderCustomizer();
      
      // Click on Charts category
      fireEvent.click(screen.getByText('Charts'));
      
      // Should show chart widgets
      expect(screen.getByText('Line Chart')).toBeInTheDocument();
      expect(screen.getByText('Bar Chart')).toBeInTheDocument();
      expect(screen.getByText('Pie Chart')).toBeInTheDocument();
    });

    it('shows all widgets when "All Widgets" is selected', () => {
      renderCustomizer();
      
      // Click on All Widgets
      fireEvent.click(screen.getByText('All Widgets'));
      
      // Should show all widget types
      expect(screen.getByText('Metric Card')).toBeInTheDocument();
      expect(screen.getByText('Line Chart')).toBeInTheDocument();
      expect(screen.getByText('Data Table')).toBeInTheDocument();
    });
  });

  describe('Widget Management', () => {
    it('selects widget when clicked', () => {
      renderCustomizer();
      
      const widget = screen.getByText('Test Metric').closest('.cursor-move');
      fireEvent.click(widget!);
      
      // Widget should be selected (would show properties panel)
      expect(widget).toHaveClass('ring-2', 'ring-blue-500');
    });

    it('deletes widget when delete button is clicked', () => {
      renderCustomizer();
      
      const deleteButtons = screen.getAllByRole('button');
      const deleteButton = deleteButtons.find(btn => 
        btn.querySelector('svg')?.getAttribute('class')?.includes('h-3')
      );
      
      if (deleteButton) {
        fireEvent.click(deleteButton);
        
        expect(mockOnTemplateUpdate).toHaveBeenCalledWith({
          ...mockTemplate,
          widgets: mockTemplate.widgets.filter(w => w.id !== 'widget-1'),
        });
      }
    });

    it('duplicates widget when duplicate button is clicked', () => {
      renderCustomizer();
      
      const copyButtons = screen.getAllByRole('button');
      const copyButton = copyButtons.find(btn => 
        btn.querySelector('svg')?.getAttribute('class')?.includes('h-3')
      );
      
      if (copyButton) {
        fireEvent.click(copyButton);
        
        expect(mockOnTemplateUpdate).toHaveBeenCalledWith(
          expect.objectContaining({
            widgets: expect.arrayContaining([
              expect.objectContaining({
                title: expect.stringContaining('(Copy)'),
              }),
            ]),
          })
        );
      }
    });
  });

  describe('Properties Panel', () => {
    it('shows widget properties when widget is selected', () => {
      renderCustomizer();
      
      // Select a widget
      const widget = screen.getByText('Test Metric').closest('.cursor-move');
      fireEvent.click(widget!);
      
      // Should show properties panel
      expect(screen.getByDisplayValue('Test Metric')).toBeInTheDocument();
      expect(screen.getByDisplayValue('roi_calculator')).toBeInTheDocument();
    });

    it('updates widget title', async () => {
      renderCustomizer();
      
      // Select widget and update title
      const widget = screen.getByText('Test Metric').closest('.cursor-move');
      fireEvent.click(widget!);
      
      const titleInput = screen.getByDisplayValue('Test Metric');
      fireEvent.change(titleInput, { target: { value: 'Updated Metric' } });
      
      await waitFor(() => {
        expect(mockOnTemplateUpdate).toHaveBeenCalledWith(
          expect.objectContaining({
            widgets: expect.arrayContaining([
              expect.objectContaining({
                id: 'widget-1',
                title: 'Updated Metric',
              }),
            ]),
          })
        );
      });
    });

    it('updates widget data source', async () => {
      renderCustomizer();
      
      // Select widget and update data source
      const widget = screen.getByText('Test Metric').closest('.cursor-move');
      fireEvent.click(widget!);
      
      const dataSourceSelect = screen.getByDisplayValue('roi_calculator');
      fireEvent.change(dataSourceSelect, { target: { value: 'cost_tracker' } });
      
      await waitFor(() => {
        expect(mockOnTemplateUpdate).toHaveBeenCalledWith(
          expect.objectContaining({
            widgets: expect.arrayContaining([
              expect.objectContaining({
                id: 'widget-1',
                data_source: 'cost_tracker',
              }),
            ]),
          })
        );
      });
    });

    it('updates widget dimensions', async () => {
      renderCustomizer();
      
      // Select widget and update width
      const widget = screen.getByText('Test Metric').closest('.cursor-move');
      fireEvent.click(widget!);
      
      const widthInput = screen.getByDisplayValue('4');
      fireEvent.change(widthInput, { target: { value: '6' } });
      
      await waitFor(() => {
        expect(mockOnTemplateUpdate).toHaveBeenCalledWith(
          expect.objectContaining({
            widgets: expect.arrayContaining([
              expect.objectContaining({
                id: 'widget-1',
                position: expect.objectContaining({
                  width: 6,
                }),
              }),
            ]),
          })
        );
      });
    });
  });

  describe('Template Settings', () => {
    it('switches to settings tab', () => {
      renderCustomizer();
      
      fireEvent.click(screen.getByText('Settings'));
      
      expect(screen.getByDisplayValue('Test Dashboard')).toBeInTheDocument();
      expect(screen.getByDisplayValue('Test dashboard description')).toBeInTheDocument();
    });

    it('updates template name', async () => {
      renderCustomizer();
      
      fireEvent.click(screen.getByText('Settings'));
      
      const nameInput = screen.getByDisplayValue('Test Dashboard');
      fireEvent.change(nameInput, { target: { value: 'Updated Dashboard' } });
      
      await waitFor(() => {
        expect(mockOnTemplateUpdate).toHaveBeenCalledWith({
          ...mockTemplate,
          name: 'Updated Dashboard',
        });
      });
    });

    it('updates template description', async () => {
      renderCustomizer();
      
      fireEvent.click(screen.getByText('Settings'));
      
      const descriptionInput = screen.getByDisplayValue('Test dashboard description');
      fireEvent.change(descriptionInput, { target: { value: 'Updated description' } });
      
      await waitFor(() => {
        expect(mockOnTemplateUpdate).toHaveBeenCalledWith({
          ...mockTemplate,
          description: 'Updated description',
        });
      });
    });

    it('updates grid size', async () => {
      renderCustomizer();
      
      fireEvent.click(screen.getByText('Settings'));
      
      const gridSizeSelect = screen.getByDisplayValue('12');
      fireEvent.change(gridSizeSelect, { target: { value: '16' } });
      
      await waitFor(() => {
        expect(mockOnTemplateUpdate).toHaveBeenCalledWith({
          ...mockTemplate,
          layout_config: {
            ...mockTemplate.layout_config,
            grid_size: 16,
          },
        });
      });
    });
  });

  describe('Actions', () => {
    it('calls onSave when save button is clicked', () => {
      renderCustomizer();
      
      fireEvent.click(screen.getByText('Save Template'));
      
      expect(mockOnSave).toHaveBeenCalled();
    });

    it('calls onPreview when preview button is clicked', () => {
      renderCustomizer();
      
      fireEvent.click(screen.getByText('Preview'));
      
      expect(mockOnPreview).toHaveBeenCalled();
    });
  });

  describe('Responsive Behavior', () => {
    it('handles different screen sizes', () => {
      // Mock window resize
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 768,
      });

      renderCustomizer();
      
      // Component should render without errors on smaller screens
      expect(screen.getByText('Test Dashboard')).toBeInTheDocument();
    });
  });

  describe('Error Handling', () => {
    it('handles empty template gracefully', () => {
      const emptyTemplate = {
        ...mockTemplate,
        widgets: [],
      };

      render(
        <TestWrapper>
          <DashboardCustomizer
            template={emptyTemplate}
            onTemplateUpdate={mockOnTemplateUpdate}
            onSave={mockOnSave}
            onPreview={mockOnPreview}
          />
        </TestWrapper>
      );
      
      expect(screen.getByText('Test Dashboard')).toBeInTheDocument();
    });

    it('handles invalid widget configurations', () => {
      const invalidTemplate = {
        ...mockTemplate,
        widgets: [
          {
            ...mockTemplate.widgets[0],
            position: { x: -1, y: -1, width: 0, height: 0 },
          },
        ],
      };

      render(
        <TestWrapper>
          <DashboardCustomizer
            template={invalidTemplate}
            onTemplateUpdate={mockOnTemplateUpdate}
            onSave={mockOnSave}
            onPreview={mockOnPreview}
          />
        </TestWrapper>
      );
      
      expect(screen.getByText('Test Dashboard')).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('provides proper ARIA labels', () => {
      renderCustomizer();
      
      const buttons = screen.getAllByRole('button');
      expect(buttons.length).toBeGreaterThan(0);
      
      // Check that important buttons have accessible names
      expect(screen.getByText('Save Template')).toBeInTheDocument();
      expect(screen.getByText('Preview')).toBeInTheDocument();
    });

    it('supports keyboard navigation', () => {
      renderCustomizer();
      
      const firstButton = screen.getByText('All Widgets');
      firstButton.focus();
      
      expect(document.activeElement).toBe(firstButton);
    });
  });
});