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
  Settings, 
  Trash2,
  AlertCircle,
  Download
} from 'lucide-react';

interface DataSinkNodeData {
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
    case 'postgresql_sink':
    case 'mysql_sink':
    case 'sqlserver_sink':
      return <Database className="w-4 h-4" />;
    case 'csv_export':
    case 'json_export':
    case 'parquet_export':
      return <FileText className="w-4 h-4" />;
    case 'api_sink':
    case 'webhook_sink':
      return <Cloud className="w-4 h-4" />;
    default:
      return <Download className="w-4 h-4" />;
  }
};

const getColor = (componentType: string) => {
  switch (componentType) {
    case 'postgresql_sink':
      return 'bg-blue-600';
    case 'mysql_sink':
      return 'bg-orange-600';
    case 'csv_export':
    case 'json_export':
      return 'bg-green-600';
    case 'api_sink':
    case 'webhook_sink':
      return 'bg-purple-600';
    default:
      return 'bg-gray-600';
  }
};

export default function DataSinkNode({ id, data, selected }: NodeProps<DataSinkNodeData>) {
  const { label, componentType, config, isValid = true, validationErrors = [], onEdit, onDelete } = data;

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
              {config.format && (
                <div>Format: {config.format}</div>
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
    </>
  );
}