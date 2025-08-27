/**
 * Integration Tests for Dashboard Customization Workflow
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { DndProvider } from 'react-dnd';
import { HTML5Backend } from 'react-dnd-html5-backend';
import '@testing-library/jest-dom';

import DashboardCustomizationInterface from '../../../components/dashboard/dashboard-customization-interface';

// Mock WebSocket hook
jest.mock('../../../hooks/useWebSocket', () => ({
  useWebSocket: () => ({
    socket: null,
    isConnected: false,
    connectionStatus: { reconnecting: false, error: null }
  })
}));

const TestWrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <DndProvider backend={HTML5Backend}>
    {children}
  </DndProvider>
);

const mockProps = {
  userId: 'test-user-123',
  userRole: 'admin',
  onSave: jest.fn(),
  onPublish: jest.fn(),
  onShare: jest.fn(),
};

describe('Dashboard Customization Workflow Integration', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('completes full dashboard creation workflow', async () => {
    render(
      <TestWrapper>
        <DashboardCustomizationInterface {...mockProps} />
      </TestWrapper>
    );

    // 1. Start with empty dashboard
    expect(screen.getByText('New Dashboard')).toBeInTheDocument();

    // 2. Add template name and description
    fireEvent.click(screen.getByText('Settings'));
    
    const nameInput = screen.getByDisplayValue('New Dashboard');
    fireEvent.change(nameInput, { target: { value: 'Executive Analytics Dashboard' } });

    const descriptionInput = screen.getByDisplayValue('Custom dashboard template');
    fireEvent.change(descriptionInput, { target: { value: 'Comprehensive executive analytics with ROI tracking' } });

    // 3. Switch back to design tab
    fireEvent.click(screen.getByText('Design'));

    // 4. Verify widget palette is available
    expect(screen.getByText('Metrics & KPIs')).toBeInTheDocument();
    expect(screen.getByText('Charts & Visualizations')).toBeInTheDocument();

    // 5. Save the dashboard
    const saveButton = screen.getByRole('button', { name: /save/i });
    fireEvent.click(saveButton);

    await waitFor(() => {
      expect(mockProps.onSave).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'Executive Analytics Dashboard',
          description: 'Comprehensive executive analytics with ROI tracking'
        })
      );
    });
  });

  it('handles responsive design workflow', async () => {
    render(
      <TestWrapper>
        <DashboardCustomizationInterface {...mockProps} />
      </TestWrapper>
    );

    // Test breakpoint switching
    const mobileButton = screen.getByRole('button', { name: /mobile/i });
    const tabletButton = screen.getByRole('button', { name: /tablet/i });
    const desktopButton = screen.getByRole('button', { name: /desktop/i });

    // Switch to mobile view
    fireEvent.click(mobileButton);
    
    // Switch to tablet view
    fireEvent.click(tabletButton);
    
    // Switch back to desktop view
    fireEvent.click(desktopButton);

    // All breakpoint switches should work without errors
    expect(screen.getByText('New Dashboard')).toBeInTheDocument();
  });

  it('handles template export and import workflow', async () => {
    // Mock file operations
    const mockCreateObjectURL = jest.fn(() => 'mock-url');
    const mockClick = jest.fn();
    const mockRevokeObjectURL = jest.fn();

    global.URL.createObjectURL = mockCreateObjectURL;
    global.URL.revokeObjectURL = mockRevokeObjectURL;

    const mockAnchor = {
      href: '',
      download: '',
      click: mockClick,
    };
    jest.spyOn(document, 'createElement').mockReturnValue(mockAnchor as any);

    render(
      <TestWrapper>
        <DashboardCustomizationInterface {...mockProps} />
      </TestWrapper>
    );

    // Export template
    fireEvent.click(screen.getByText('Export'));
    const exportButton = screen.getByRole('button', { name: /export template/i });
    fireEvent.click(exportButton);

    expect(mockCreateObjectURL).toHaveBeenCalled();
    expect(mockClick).toHaveBeenCalled();
  });
});