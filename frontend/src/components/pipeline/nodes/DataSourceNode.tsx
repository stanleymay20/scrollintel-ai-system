"use client";

import React from 'react';
import { Handle, Position, NodeProps } from 'reactflow';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { 
  Database, 
  FileText, 
  Cloud, 
  Zap, 
  Settings, 
  Trash2,
  AlertCircle
} from 'lucide-react';

interface DataSourceNodeData {
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
    case 'postgresql':
    case 'mysql':
    case 'sqlserver':
    case 'oracle':
      return <Database className="w-4 h-4" />;
    case 'csv':
    case 'json':
    case 'parquet':
    case 'excel':
      return <FileText className="w-4 h-4" />;
    case 'rest_api':
    case 'graphql':
      return <Cloud className="w-4 h-4" />;
    case 'kafka':
    case 'kinesis':
    case 'pubsub':
      return <Zap className="w-4 h-4" />;
    default:
      return <Database className="w-4 h-4" />;
  }
};

const getColor = (componentType: string) => {
  switch (componentType) {
    case 'postgresql':
      return 'bg-blue-500';
    case 'mysql':
      return 'bg-orange-500';
    case 'csv':
    case 'json':
      return 'bg-green-500';
    case 'rest_api':
    case 'graphql':
      return 'bg-purple-500';
    case 'kafka':
    case 'kinesis':
      return 'bg-red-500';
    default:
      return 'bg-gray-500';
  }
};

export default function DataSourceNode({ id, data, selected }: NodeProps<DataSourceNodeData>) {
  const { label, componentType, config, isValid = true, validationErrors = [], onEdit, onDelete } = data;

  const handleEdit = () => {
    onEdit?.(id);
  };

  const handleDelete = () => {
    onDelete?.(id);
  };

  return (
    <>
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
          {config && Object.keys(config).length > 0 && (
            <div className="text-xs text-gray-600 mb-2">
              {config.host && (
                <div>Host: {config.host}</div>
              )}
              {config.database && (
                <div>DB: {config.database}</div>
              )}
              {config.table && (
                <div>Table: {config.table}</div>
              )}
              {config.file_path && (
                <div>File: {config.file_path.split('/').pop()}</div>
              )}
              {config.url && (
                <div>URL: {new URL(config.url).hostname}</div>
              )}
            </div>
          )}

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