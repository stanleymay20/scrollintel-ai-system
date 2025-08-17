"use client";

import React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

interface ComponentItem {
  id: string;
  name: string;
  type: string;
  description: string;
  icon: string;
  category: string;
}

const componentLibrary: ComponentItem[] = [
  // Data Sources
  {
    id: 'db-source',
    name: 'Database Source',
    type: 'data-source',
    description: 'Connect to SQL databases (PostgreSQL, MySQL, SQL Server)',
    icon: '🗄️',
    category: 'Data Sources'
  },
  {
    id: 'api-source',
    name: 'REST API Source',
    type: 'data-source',
    description: 'Fetch data from REST APIs',
    icon: '🌐',
    category: 'Data Sources'
  },
  {
    id: 'file-source',
    name: 'File Source',
    type: 'data-source',
    description: 'Read from CSV, JSON, Parquet, Excel files',
    icon: '📁',
    category: 'Data Sources'
  },
  {
    id: 'stream-source',
    name: 'Stream Source',
    type: 'data-source',
    description: 'Connect to Kafka, Kinesis, Pub/Sub streams',
    icon: '🌊',
    category: 'Data Sources'
  },
  
  // Transformations
  {
    id: 'filter-transform',
    name: 'Filter',
    type: 'transformation',
    description: 'Filter rows based on conditions',
    icon: '🔍',
    category: 'Transformations'
  },
  {
    id: 'map-transform',
    name: 'Map/Transform',
    type: 'transformation',
    description: 'Transform column values',
    icon: '🔄',
    category: 'Transformations'
  },
  {
    id: 'aggregate-transform',
    name: 'Aggregate',
    type: 'transformation',
    description: 'Group and aggregate data',
    icon: '📊',
    category: 'Transformations'
  },
  {
    id: 'join-transform',
    name: 'Join',
    type: 'transformation',
    description: 'Join multiple datasets',
    icon: '🔗',
    category: 'Transformations'
  },
  {
    id: 'sort-transform',
    name: 'Sort',
    type: 'transformation',
    description: 'Sort data by columns',
    icon: '📈',
    category: 'Transformations'
  },
  {
    id: 'dedupe-transform',
    name: 'Deduplicate',
    type: 'transformation',
    description: 'Remove duplicate records',
    icon: '🧹',
    category: 'Transformations'
  },
  
  // Data Targets
  {
    id: 'db-target',
    name: 'Database Target',
    type: 'data-target',
    description: 'Write to SQL databases',
    icon: '💾',
    category: 'Data Targets'
  },
  {
    id: 'warehouse-target',
    name: 'Data Warehouse',
    type: 'data-target',
    description: 'Write to data warehouses (Snowflake, BigQuery)',
    icon: '🏢',
    category: 'Data Targets'
  },
  {
    id: 'lake-target',
    name: 'Data Lake',
    type: 'data-target',
    description: 'Write to data lakes (S3, ADLS, GCS)',
    icon: '🏞️',
    category: 'Data Targets'
  },
  {
    id: 'api-target',
    name: 'API Target',
    type: 'data-target',
    description: 'Send data to REST APIs',
    icon: '📤',
    category: 'Data Targets'
  },
  
  // Validation
  {
    id: 'schema-validation',
    name: 'Schema Validation',
    type: 'validation',
    description: 'Validate data schema and types',
    icon: '✅',
    category: 'Validation'
  },
  {
    id: 'quality-validation',
    name: 'Quality Check',
    type: 'validation',
    description: 'Check data quality rules',
    icon: '🎯',
    category: 'Validation'
  },
  {
    id: 'anomaly-detection',
    name: 'Anomaly Detection',
    type: 'validation',
    description: 'Detect data anomalies',
    icon: '🚨',
    category: 'Validation'
  }
];

const categories = Array.from(new Set(componentLibrary.map(item => item.category)));

export function ComponentLibrary() {
  const onDragStart = (event: React.DragEvent, nodeType: string, label: string) => {
    event.dataTransfer.setData('application/reactflow', nodeType);
    event.dataTransfer.setData('application/reactflow-label', label);
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm">Component Library</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {categories.map(category => (
          <div key={category}>
            <h4 className="text-xs font-semibold text-gray-600 mb-2">{category}</h4>
            <div className="space-y-2">
              {componentLibrary
                .filter(item => item.category === category)
                .map(item => (
                  <div
                    key={item.id}
                    className="p-3 border rounded-lg cursor-move hover:bg-gray-50 transition-colors"
                    draggable
                    onDragStart={(event) => onDragStart(event, item.type, item.name)}
                  >
                    <div className="flex items-start space-x-2">
                      <span className="text-lg">{item.icon}</span>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center space-x-2">
                          <h5 className="text-sm font-medium truncate">{item.name}</h5>
                          <Badge 
                            variant="secondary" 
                            className="text-xs"
                          >
                            {item.type.replace('-', ' ')}
                          </Badge>
                        </div>
                        <p className="text-xs text-gray-500 mt-1 line-clamp-2">
                          {item.description}
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
            </div>
          </div>
        ))}
        
        <div className="mt-4 p-3 bg-blue-50 rounded-lg">
          <p className="text-xs text-blue-700">
            💡 <strong>Tip:</strong> Drag components from the library onto the canvas to build your pipeline.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}