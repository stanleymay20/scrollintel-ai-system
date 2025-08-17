"use client";

import React, { useState, useCallback, useRef } from 'react';
import ReactFlow, {
  Node,
  Edge,
  addEdge,
  Connection,
  useNodesState,
  useEdgesState,
  Controls,
  MiniMap,
  Background,
  BackgroundVariant,
  ReactFlowProvider,
  ReactFlowInstance,
} from 'reactflow';
import 'reactflow/dist/style.css';

import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ComponentLibrary } from './component-library';
import { PipelineValidation } from './pipeline-validation';

interface PipelineBuilderProps {
  pipelineId?: string;
  onSave?: (pipeline: any) => void;
  onValidate?: (pipeline: any) => void;
}

const initialNodes: Node[] = [];
const initialEdges: Edge[] = [];

export function PipelineBuilder({ pipelineId, onSave, onValidate }: PipelineBuilderProps) {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);
  const [reactFlowInstance, setReactFlowInstance] = useState<ReactFlowInstance | null>(null);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [validationResult, setValidationResult] = useState<any>(null);
  const [isValidating, setIsValidating] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  
  const reactFlowWrapper = useRef<HTMLDivElement>(null);

  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

  const onInit = (rfi: ReactFlowInstance) => setReactFlowInstance(rfi);

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();

      const reactFlowBounds = reactFlowWrapper.current?.getBoundingClientRect();
      const type = event.dataTransfer.getData('application/reactflow');
      const label = event.dataTransfer.getData('application/reactflow-label');

      if (typeof type === 'undefined' || !type || !reactFlowInstance || !reactFlowBounds) {
        return;
      }

      const position = reactFlowInstance.project({
        x: event.clientX - reactFlowBounds.left,
        y: event.clientY - reactFlowBounds.top,
      });

      const newNode: Node = {
        id: `${type}-${Date.now()}`,
        type: 'default',
        position,
        data: { 
          label,
          nodeType: type,
          config: getDefaultConfig(type)
        },
        style: getNodeStyle(type),
      };

      setNodes((nds) => nds.concat(newNode));
    },
    [reactFlowInstance, setNodes]
  );

  const getDefaultConfig = (nodeType: string) => {
    const configs = {
      'data-source': {
        sourceType: 'database',
        connectionString: '',
        query: '',
        schema: {}
      },
      'transformation': {
        transformationType: 'filter',
        expression: '',
        parameters: {}
      },
      'data-target': {
        targetType: 'database',
        connectionString: '',
        table: '',
        writeMode: 'append'
      },
      'validation': {
        validationType: 'schema',
        rules: [],
        onFailure: 'stop'
      }
    };
    return configs[nodeType as keyof typeof configs] || {};
  };

  const getNodeStyle = (nodeType: string) => {
    const styles = {
      'data-source': { 
        background: '#e3f2fd', 
        border: '2px solid #2196f3',
        borderRadius: '8px',
        padding: '10px'
      },
      'transformation': { 
        background: '#f3e5f5', 
        border: '2px solid #9c27b0',
        borderRadius: '8px',
        padding: '10px'
      },
      'data-target': { 
        background: '#e8f5e8', 
        border: '2px solid #4caf50',
        borderRadius: '8px',
        padding: '10px'
      },
      'validation': { 
        background: '#fff3e0', 
        border: '2px solid #ff9800',
        borderRadius: '8px',
        padding: '10px'
      }
    };
    return styles[nodeType as keyof typeof styles] || {};
  };

  const onNodeClick = useCallback((event: React.MouseEvent, node: Node) => {
    setSelectedNode(node);
  }, []);

  const validatePipeline = async () => {
    setIsValidating(true);
    try {
      const pipelineData = {
        nodes: nodes.map(node => ({
          id: node.id,
          type: node.data.nodeType,
          config: node.data.config,
          position: node.position
        })),
        edges: edges.map(edge => ({
          id: edge.id,
          source: edge.source,
          target: edge.target
        }))
      };

      if (onValidate) {
        const result = await onValidate(pipelineData);
        setValidationResult(result);
      } else {
        // Mock validation for demo
        const mockResult = {
          isValid: nodes.length > 0 && edges.length > 0,
          errors: nodes.length === 0 ? ['Pipeline must have at least one node'] : [],
          warnings: edges.length === 0 ? ['Pipeline should have connections between nodes'] : []
        };
        setValidationResult(mockResult);
      }
    } catch (error) {
      console.error('Validation error:', error);
      setValidationResult({
        isValid: false,
        errors: ['Validation failed: ' + (error as Error).message],
        warnings: []
      });
    } finally {
      setIsValidating(false);
    }
  };

  const savePipeline = async () => {
    setIsSaving(true);
    try {
      const pipelineData = {
        id: pipelineId,
        nodes: nodes.map(node => ({
          id: node.id,
          type: node.data.nodeType,
          name: node.data.label,
          config: node.data.config,
          position: node.position
        })),
        connections: edges.map(edge => ({
          id: edge.id,
          sourceNodeId: edge.source,
          targetNodeId: edge.target
        }))
      };

      if (onSave) {
        await onSave(pipelineData);
      }
    } catch (error) {
      console.error('Save error:', error);
    } finally {
      setIsSaving(false);
    }
  };

  const clearPipeline = () => {
    setNodes([]);
    setEdges([]);
    setSelectedNode(null);
    setValidationResult(null);
  };

  return (
    <div className="h-screen flex">
      {/* Component Library Sidebar */}
      <div className="w-80 border-r bg-gray-50 p-4">
        <ComponentLibrary />
        
        {/* Pipeline Actions */}
        <Card className="mt-4">
          <CardHeader>
            <CardTitle className="text-sm">Pipeline Actions</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <Button 
              onClick={validatePipeline} 
              disabled={isValidating}
              className="w-full"
              variant="outline"
            >
              {isValidating ? 'Validating...' : 'Validate Pipeline'}
            </Button>
            <Button 
              onClick={savePipeline} 
              disabled={isSaving}
              className="w-full"
            >
              {isSaving ? 'Saving...' : 'Save Pipeline'}
            </Button>
            <Button 
              onClick={clearPipeline} 
              variant="destructive"
              className="w-full"
            >
              Clear Pipeline
            </Button>
          </CardContent>
        </Card>

        {/* Pipeline Stats */}
        <Card className="mt-4">
          <CardHeader>
            <CardTitle className="text-sm">Pipeline Stats</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span>Nodes:</span>
                <Badge variant="secondary">{nodes.length}</Badge>
              </div>
              <div className="flex justify-between">
                <span>Connections:</span>
                <Badge variant="secondary">{edges.length}</Badge>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Validation Results */}
        {validationResult && (
          <PipelineValidation result={validationResult} />
        )}
      </div>

      {/* Main Pipeline Canvas */}
      <div className="flex-1" ref={reactFlowWrapper}>
        <ReactFlowProvider>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onInit={onInit}
            onDrop={onDrop}
            onDragOver={onDragOver}
            onNodeClick={onNodeClick}
            fitView
            attributionPosition="top-right"
          >
            <Controls />
            <MiniMap />
            <Background variant={BackgroundVariant.Dots} gap={12} size={1} />
          </ReactFlow>
        </ReactFlowProvider>
      </div>

      {/* Node Configuration Panel */}
      {selectedNode && (
        <div className="w-80 border-l bg-gray-50 p-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Node Configuration</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <label className="text-sm font-medium">Node Type</label>
                  <Badge className="ml-2">{selectedNode.data.nodeType}</Badge>
                </div>
                <div>
                  <label className="text-sm font-medium">Label</label>
                  <input
                    type="text"
                    value={selectedNode.data.label}
                    onChange={(e) => {
                      const updatedNode = {
                        ...selectedNode,
                        data: { ...selectedNode.data, label: e.target.value }
                      };
                      setSelectedNode(updatedNode);
                      setNodes(nds => nds.map(n => n.id === selectedNode.id ? updatedNode : n));
                    }}
                    className="w-full mt-1 px-3 py-2 border border-gray-300 rounded-md text-sm"
                  />
                </div>
                <div>
                  <label className="text-sm font-medium">Configuration</label>
                  <textarea
                    value={JSON.stringify(selectedNode.data.config, null, 2)}
                    onChange={(e) => {
                      try {
                        const config = JSON.parse(e.target.value);
                        const updatedNode = {
                          ...selectedNode,
                          data: { ...selectedNode.data, config }
                        };
                        setSelectedNode(updatedNode);
                        setNodes(nds => nds.map(n => n.id === selectedNode.id ? updatedNode : n));
                      } catch (error) {
                        // Invalid JSON, don't update
                      }
                    }}
                    rows={8}
                    className="w-full mt-1 px-3 py-2 border border-gray-300 rounded-md text-sm font-mono"
                  />
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}