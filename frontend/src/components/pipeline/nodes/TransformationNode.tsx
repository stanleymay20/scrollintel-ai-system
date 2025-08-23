"use client";

import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { 
  Filter, 
  Shuffle, 
  BarChart3, 
  GitMerge, 
  Settings, 
  Trash2,
  AlertCircle,
  ArrowUpDown
} from 'lucide-react';

interface TransformationNodeData {
  label: string;
  componentType: string;
  config?: Record<string, any>;
  isValid?: boolean;
  validationErrors?: string[];
  onEdit?: (nodeId: string) => void;
  onDelete?: (nodeId: string) => void;
}

const getIcon = (componentType: string) => {
  switch (componentType) {
    case 'filter':
      return <Filter className="w-4 h-4" />;
    case 'map':
      return <Shuffle className="w-4 h-4" />;
    case 'aggregate':
      return <BarChart3 className="w-4 h-4" />;
    case 'join':
      return <GitMerge className="w-4 h-4" />;
    case 'sort':
      return <ArrowUpDown className="w-4 h-4" />;
    default:
      return <Settings className="w-4 h-4" />;
  }
};

const getColor = (componentType: string) => {
  switch (componentType) {
    case 'filter':
      return 'bg-indigo-500';
    case 'map':
      return 'bg-teal-500';
    case 'aggregate':
      return 'bg-yellow-500';
    case 'join':
      return 'bg-pink-500';
    case 'sort':
      return 'bg-gray-500';
    default:
      return 'bg-blue-500';
  }
};

const getConfigSummary = (componentType: string, config: Record<string, any>) => {
  switch (componentType) {
    case 'filter':
      return config.condition ? `Where: ${config.condition}` : 'No condition set';
    case 'map':
      return config.mappings ? `${Object.keys(config.mappings).length} mappings` : 'No mappings';
    case 'aggregate':
      return config.groupBy ? `Group by: ${config.groupBy.join(', ')}` : 'No grouping';
    case 'join':
      return config.joinType ? `${config.joinType} join` : 'Inner join';
    case 'sort':
      return config.columns ? `Sort by: ${config.columns.join(', ')}` : 'No sort columns';
    default:
      return 'Not configured';
  }
};

export default function TransformationNode({ id, data, selected }: NodeProps<TransformationNodeData>) {
  const { label, componentType, config = {}, isValid = true, validationErrors = [], onEdit, onDelete } = data;

  const handleEdit = () => {
    onEdit?.(id);
  };

  const handleDelete = () => {
    onDelete?.(id);
  };

  return (
    <>
      {/* Input Handle */}
      <Handle
        type="target"
        position={Position.Left}
        id="input"
        className="w-3 h-3 bg-green-500 border-2 border-white"
      />

      <Card className={`min-w-[200px] ${selected ? 'ring-2 ring-blue-500' : ''} ${!isValid ? 'border-red-300' : ''}`}>
        <CardContent className="p-3">
          {/* Header */}
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center space-x-2">
              <div className={`p-1.5 rounded ${getColor(componentType)} text-white`}>
                {getIcon(componentType)}
              </div>
              <div className="flex-1">
                <h4 className="font-medium text-sm truncate">{label}</h4>
                <Badge variant="outline" className="text-xs mt-1">
                  {componentType}
                </Badge>
              </div>
            </div>
            
            {!isValid && (
              <AlertCircle className="w-4 h-4 text-red-500 flex-shrink-0" />
            )}
          </div>

          {/* Configuration Summary */}
          <div className="text-xs text-gray-600 mb-2">
            {getConfigSummary(componentType, config)}
          </div>

          {/* Validation Errors */}
          {validationErrors.length > 0 && (
            <div className="text-xs text-red-600 mb-2">
              {validationErrors.slice(0, 2).map((error, index) => (
                <div key={index} className="truncate">• {error}</div>
              ))}
              {validationErrors.length > 2 && (
                <div>... and {validationErrors.length - 2} more</div>
              )}
            </div>
          )}

          {/* Actions */}
          <div className="flex items-center justify-between">
            <Badge 
              variant={isValid ? 'default' : 'destructive'}
              className="text-xs"
            >
              {isValid ? 'Ready' : 'Error'}
            </Badge>
            
            <div className="flex space-x-1">
              <Button
                variant="ghost"
                size="sm"
                onClick={handleEdit}
                className="h-6 w-6 p-0"
              >
                <Settings className="w-3 h-3" />
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={handleDelete}
                className="h-6 w-6 p-0 text-red-500 hover:text-red-700"
              >
                <Trash2 className="w-3 h-3" />
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Output Handle */}
      <Handle
        type="source"
        position={Position.Right}
        id="output"
        className="w-3 h-3 bg-blue-500 border-2 border-white"
      />
    </>
  );
}