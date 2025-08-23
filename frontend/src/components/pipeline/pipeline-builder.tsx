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
  Panel,
  ReactFlowProvider,
  ReactFlowInstance,
} from 'reactflow';
import 'reactflow/dist/style.css';

import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ComponentLibrary } from './component-library';
import { PipelineValidation } from './pipeline-validation';

// Custom node types
import DataSourceNode from './nodes/DataSourceNode';
import TransformationNode from './nodes/TransformationNode';
import DataSinkNode from './nodes/DataSinkNode';

const nodeTypes = {
  dataSource: DataSourceNode,
  transformation: TransformationNode,
  dataSink: DataSinkNode,
};

interface Pipeline {
  id: string;
  name: string;
  description: string;
  status: string;
  validation_status: string;
}

interface PipelineBuilderProps {
  pipeline?: Pipeline;
  onSave?: (pipeline: Pipeline) => void;
  onValidate?: (pipelineId: string) => void;
}

export function PipelineBuilder({ pipeline, onSave, onValidate }: PipelineBuilderProps) {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [reactFlowInstance, setReactFlowInstance] = useState<ReactFlowInstance | null>(null);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [showComponentLibrary, setShowComponentLibrary] = useState(true);
  const [showValidation, setShowValidation] = useState(false);
  const [pipelineName, setPipelineName] = useState(pipeline?.name || 'New Pipeline');
  const [pipelineDescription, setPipelineDescription] = useState(pipeline?.description || '');
  
  const reactFlowWrapper = useRef<HTMLDivElement>(null);

  const onConnect = useCallback(
    (params: Connection) => setEdges((eds) => addEdge(params, eds)),
    [setEdges]
  );

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
      const nodeType = event.dataTransfer.getData('application/reactflow-nodetype');

      if (typeof type === 'undefined' || !type || !reactFlowBounds) {
        return;
      }

      const position = reactFlowInstance?.project({
        x: event.clientX - reactFlowBounds.left,
        y: event.clientY - reactFlowBounds.top,
      });

      if (!position) return;

      const newNode: Node = {
        id: `${type}-${Date.now()}`,
        type: nodeType,
        position,
        data: {
          label,
          componentType: type,
          config: {},
          onEdit: handleNodeEdit,
          onDelete: handleNodeDelete,
        },
      };

      setNodes((nds) => nds.concat(newNode));
    },
    [reactFlowInstance, setNodes]
  );

  const handleNodeEdit = useCallback((nodeId: string) => {
    const node = nodes.find(n => n.id === nodeId);
    if (node) {
      setSelectedNode(node);
    }
  }, [nodes]);

  const handleNodeDelete = useCallback((nodeId: string) => {
    setNodes((nds) => nds.filter(n => n.id !== nodeId));
    setEdges((eds) => eds.filter(e => e.source !== nodeId && e.target !== nodeId));
  }, [setNodes, setEdges]);

  const handleSave = async () => {
    if (!pipeline?.id) return;

    const pipelineData = {
      ...pipeline,
      name: pipelineName,
      description: pipelineDescription,
      nodes: nodes.map(node => ({
        id: node.id,
        name: node.data.label,
        node_type: node.type,
        component_type: node.data.componentType,
        position_x: node.position.x,
        position_y: node.position.y,
        config: node.data.config || {},
      })),
      connections: edges.map(edge => ({
        source_node_id: edge.source,
        target_node_id: edge.target,
        source_port: edge.sourceHandle || 'output',
        target_port: edge.targetHandle || 'input',
      })),
    };

    try {
      const response = await fetch(`/api/pipelines/${pipeline.id}`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(pipelineData),
      });

      if (response.ok) {
        const updatedPipeline = await response.json();
        onSave?.(updatedPipeline);
      }
    } catch (error) {
      console.error('Failed to save pipeline:', error);
    }
  };

  const handleValidate = async () => {
    if (!pipeline?.id) return;
    
    setShowValidation(true);
    onValidate?.(pipeline.id);
  };

  const handleRunPipeline = async () => {
    if (!pipeline?.id) return;

    try {
      const response = await fetch(`/api/pipelines/${pipeline.id}/execute`, {
        method: 'POST',
      });

      if (response.ok) {
        console.log('Pipeline execution started');
      }
    } catch (error) {
      console.error('Failed to run pipeline:', error);
    }
  };

  return (
    <div className="h-screen flex flex-col">
      {/* Header */}
      <div className="border-b bg-white p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div>
              <input
                type="text"
                value={pipelineName}
                onChange={(e) => setPipelineName(e.target.value)}
                className="text-xl font-semibold bg-transparent border-none outline-none"
              />
              <input
                type="text"
                value={pipelineDescription}
                onChange={(e) => setPipelineDescription(e.target.value)}
                placeholder="Pipeline description..."
                className="text-sm text-gray-600 bg-transparent border-none outline-none block mt-1"
              />
            </div>
            {pipeline?.status && (
              <Badge variant={pipeline.status === 'active' ? 'default' : 'secondary'}>
                {pipeline.status}
              </Badge>
            )}
            {pipeline?.validation_status && (
              <Badge 
                variant={
                  pipeline.validation_status === 'valid' ? 'default' :
                  pipeline.validation_status === 'invalid' ? 'destructive' : 'secondary'
                }
              >
                {pipeline.validation_status}
              </Badge>
            )}
          </div>
          
          <div className="flex items-center space-x-2">
            <Button
              variant="outline"
              onClick={() => setShowComponentLibrary(!showComponentLibrary)}
            >
              Components
            </Button>
            <Button
              variant="outline"
              onClick={handleValidate}
            >
              Validate
            </Button>
            <Button
              variant="outline"
              onClick={handleSave}
            >
              Save
            </Button>
            <Button
              onClick={handleRunPipeline}
              disabled={pipeline?.validation_status !== 'valid'}
            >
              Run Pipeline
            </Button>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="flex-1 flex">
        {/* Component Library Sidebar */}
        {showComponentLibrary && (
          <div className="w-80 border-r bg-gray-50">
            <ComponentLibrary />
          </div>
        )}

        {/* Pipeline Canvas */}
        <div className="flex-1 relative" ref={reactFlowWrapper}>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onInit={setReactFlowInstance}
            onDrop={onDrop}
            onDragOver={onDragOver}
            nodeTypes={nodeTypes}
            fitView
          >
            <Controls />
            <MiniMap />
            <Background variant={BackgroundVariant.Dots} gap={12} size={1} />
            
            <Panel position="top-left">
              <Card className="w-64">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm">Pipeline Stats</CardTitle>
                </CardHeader>
                <CardContent className="text-xs space-y-1">
                  <div className="flex justify-between">
                    <span>Nodes:</span>
                    <span>{nodes.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Connections:</span>
                    <span>{edges.length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Data Sources:</span>
                    <span>{nodes.filter(n => n.type === 'dataSource').length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Transformations:</span>
                    <span>{nodes.filter(n => n.type === 'transformation').length}</span>
                  </div>
                </CardContent>
              </Card>
            </Panel>
          </ReactFlow>
        </div>

        {/* Validation Panel */}
        {showValidation && (
          <div className="w-80 border-l bg-white">
            <PipelineValidation
              pipelineId={pipeline?.id}
              onClose={() => setShowValidation(false)}
            />
          </div>
        )}
      </div>
    </div>
  );
}

export default function PipelineBuilderWrapper(props: PipelineBuilderProps) {
  return (
    <ReactFlowProvider>
      <PipelineBuilder {...props} />
    </ReactFlowProvider>
  );
}