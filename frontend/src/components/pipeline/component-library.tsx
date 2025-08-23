"use client";

import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { 
  Database, 
  Filter, 
  Shuffle, 
  BarChart3, 
  GitMerge, 
  FileText, 
  Cloud, 
  Zap,
  Search,
  Settings
} from 'lucide-react';

interface ComponentTemplate {
  id: string;
  name: string;
  description: string;
  category: string;
  nodeType: string;
  componentType: string;
  icon: React.ReactNode;
  color: string;
}

const componentTemplates: ComponentTemplate[] = [
  // Data Sources
  {
    id: 'postgres-source',
    name: 'PostgreSQL',
    description: 'Connect to PostgreSQL database',
    category: 'Data Sources',
    nodeType: 'dataSource',
    componentType: 'postgresql',
    icon: <Database className="w-4 h-4" />,
    color: 'bg-blue-500'
  },
  {
    id: 'mysql-source',
    name: 'MySQL',
    description: 'Connect to MySQL database',
    category: 'Data Sources',
    nodeType: 'dataSource',
    componentType: 'mysql',
    icon: <Database className="w-4 h-4" />,
    color: 'bg-orange-500'
  },
  {
    id: 'csv-source',
    name: 'CSV File',
    description: 'Read data from CSV files',
    category: 'Data Sources',
    nodeType: 'dataSource',
    componentType: 'csv',
    icon: <FileText className="w-4 h-4" />,
    color: 'bg-green-500'
  },
  {
    id: 'rest-api-source',
    name: 'REST API',
    description: 'Fetch data from REST endpoints',
    category: 'Data Sources',
    nodeType: 'dataSource',
    componentType: 'rest_api',
    icon: <Cloud className="w-4 h-4" />,
    color: 'bg-purple-500'
  },
  {
    id: 'kafka-source',
    name: 'Kafka Stream',
    description: 'Stream data from Kafka topics',
    category: 'Data Sources',
    nodeType: 'dataSource',
    componentType: 'kafka',
    icon: <Zap className="w-4 h-4" />,
    color: 'bg-red-500'
  },

  // Transformations
  {
    id: 'filter-transform',
    name: 'Filter',
    description: 'Filter rows based on conditions',
    category: 'Transformations',
    nodeType: 'transformation',
    componentType: 'filter',
    icon: <Filter className="w-4 h-4" />,
    color: 'bg-indigo-500'
  },
  {
    id: 'map-transform',
    name: 'Map',
    description: 'Transform column values',
    category: 'Transformations',
    nodeType: 'transformation',
    componentType: 'map',
    icon: <Shuffle className="w-4 h-4" />,
    color: 'bg-teal-500'
  },
  {
    id: 'aggregate-transform',
    name: 'Aggregate',
    description: 'Group and aggregate data',
    category: 'Transformations',
    nodeType: 'transformation',
    componentType: 'aggregate',
    icon: <BarChart3 className="w-4 h-4" />,
    color: 'bg-yellow-500'
  },
  {
    id: 'join-transform',
    name: 'Join',
    description: 'Join multiple data streams',
    category: 'Transformations',
    nodeType: 'transformation',
    componentType: 'join',
    icon: <GitMerge className="w-4 h-4" />,
    color: 'bg-pink-500'
  },
  {
    id: 'sort-transform',
    name: 'Sort',
    description: 'Sort data by columns',
    category: 'Transformations',
    nodeType: 'transformation',
    componentType: 'sort',
    icon: <Settings className="w-4 h-4" />,
    color: 'bg-gray-500'
  },

  // Data Sinks
  {
    id: 'postgres-sink',
    name: 'PostgreSQL Sink',
    description: 'Write data to PostgreSQL',
    category: 'Data Sinks',
    nodeType: 'dataSink',
    componentType: 'postgresql_sink',
    icon: <Database className="w-4 h-4" />,
    color: 'bg-blue-600'
  },
  {
    id: 'csv-sink',
    name: 'CSV Export',
    description: 'Export data to CSV files',
    category: 'Data Sinks',
    nodeType: 'dataSink',
    componentType: 'csv_export',
    icon: <FileText className="w-4 h-4" />,
    color: 'bg-green-600'
  },
  {
    id: 'api-sink',
    name: 'API Endpoint',
    description: 'Send data to API endpoints',
    category: 'Data Sinks',
    nodeType: 'dataSink',
    componentType: 'api_sink',
    icon: <Cloud className="w-4 h-4" />,
    color: 'bg-purple-600'
  }
];

export function ComponentLibrary() {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('All');

  const categories = ['All', ...Array.from(new Set(componentTemplates.map(t => t.category)))];

  const filteredTemplates = componentTemplates.filter(template => {
    const matchesSearch = template.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         template.description.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCategory = selectedCategory === 'All' || template.category === selectedCategory;
    return matchesSearch && matchesCategory;
  });

  const onDragStart = (event: React.DragEvent, template: ComponentTemplate) => {
    event.dataTransfer.setData('application/reactflow', template.componentType);
    event.dataTransfer.setData('application/reactflow-label', template.name);
    event.dataTransfer.setData('application/reactflow-nodetype', template.nodeType);
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    <div className="h-full flex flex-col">
      <div className="p-4 border-b">
        <h3 className="text-lg font-semibold mb-3">Component Library</h3>
        
        {/* Search */}
        <div className="relative mb-3">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
          <Input
            placeholder="Search components..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10"
          />
        </div>

        {/* Category Filter */}
        <div className="flex flex-wrap gap-1">
          {categories.map(category => (
            <Badge
              key={category}
              variant={selectedCategory === category ? 'default' : 'outline'}
              className="cursor-pointer text-xs"
              onClick={() => setSelectedCategory(category)}
            >
              {category}
            </Badge>
          ))}
        </div>
      </div>

      <ScrollArea className="flex-1">
        <div className="p-4 space-y-3">
          {filteredTemplates.map(template => (
            <Card
              key={template.id}
              className="cursor-grab active:cursor-grabbing hover:shadow-md transition-shadow"
              draggable
              onDragStart={(event) => onDragStart(event, template)}
            >
              <CardContent className="p-3">
                <div className="flex items-start space-x-3">
                  <div className={`p-2 rounded-md ${template.color} text-white flex-shrink-0`}>
                    {template.icon}
                  </div>
                  <div className="flex-1 min-w-0">
                    <h4 className="font-medium text-sm truncate">{template.name}</h4>
                    <p className="text-xs text-gray-600 mt-1 line-clamp-2">
                      {template.description}
                    </p>
                    <Badge variant="outline" className="mt-2 text-xs">
                      {template.category}
                    </Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}

          {filteredTemplates.length === 0 && (
            <div className="text-center py-8 text-gray-500">
              <Search className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p className="text-sm">No components found</p>
              <p className="text-xs">Try adjusting your search or filter</p>
            </div>
          )}
        </div>
      </ScrollArea>

      {/* Usage Instructions */}
      <div className="p-4 border-t bg-gray-50">
        <p className="text-xs text-gray-600">
          <strong>Tip:</strong> Drag components from the library onto the canvas to build your pipeline.
        </p>
      </div>
    </div>
  );
}