"""
ScrollIntel Mobile SDK Generator
Automated generation of Flutter and React Native SDKs for mobile deployment
"""

import os
import json
from typing import Dict, List, Any
from pathlib import Path
import yaml

class MobileSDKGenerator:
    """Generates mobile SDKs for ScrollIntel platform"""
    
    def __init__(self):
        self.flutter_templates = {
            'api_client': self._get_flutter_api_template(),
            'models': self._get_flutter_models_template(),
            'services': self._get_flutter_services_template(),
            'widgets': self._get_flutter_widgets_template()
        }
        
        self.react_native_templates = {
            'api_client': self._get_rn_api_template(),
            'models': self._get_rn_models_template(),
            'services': self._get_rn_services_template(),
            'components': self._get_rn_components_template()
        }
    
    def generate_flutter_sdk(self, output_dir: str = "mobile_sdks/flutter") -> Dict[str, Any]:
        """Generate complete Flutter SDK"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate pubspec.yaml
            self._generate_flutter_pubspec(output_dir)
            
            # Generate API client
            self._generate_flutter_api_client(output_dir)
            
            # Generate models
            self._generate_flutter_models(output_dir)
            
            # Generate services
            self._generate_flutter_services(output_dir)
            
            # Generate widgets
            self._generate_flutter_widgets(output_dir)
            
            # Generate example app
            self._generate_flutter_example(output_dir)
            
            return {
                "status": "success",
                "sdk_type": "flutter",
                "output_directory": output_dir,
                "files_generated": self._count_generated_files(output_dir)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "sdk_type": "flutter"
            }
    
    def generate_react_native_sdk(self, output_dir: str = "mobile_sdks/react_native") -> Dict[str, Any]:
        """Generate complete React Native SDK"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate package.json
            self._generate_rn_package_json(output_dir)
            
            # Generate API client
            self._generate_rn_api_client(output_dir)
            
            # Generate models
            self._generate_rn_models(output_dir)
            
            # Generate services
            self._generate_rn_services(output_dir)
            
            # Generate components
            self._generate_rn_components(output_dir)
            
            # Generate example app
            self._generate_rn_example(output_dir)
            
            return {
                "status": "success",
                "sdk_type": "react_native",
                "output_directory": output_dir,
                "files_generated": self._count_generated_files(output_dir)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "sdk_type": "react_native"
            }
    
    def _generate_flutter_pubspec(self, output_dir: str):
        """Generate Flutter pubspec.yaml"""
        pubspec = {
            'name': 'scrollintel_flutter_sdk',
            'description': 'Flutter SDK for ScrollIntel AI-CTO Platform',
            'version': '1.0.0',
            'environment': {
                'sdk': '>=2.17.0 <4.0.0',
                'flutter': '>=3.0.0'
            },
            'dependencies': {
                'flutter': {'sdk': 'flutter'},
                'http': '^0.13.5',
                'dio': '^5.0.0',
                'json_annotation': '^4.8.0',
                'equatable': '^2.0.5',
                'provider': '^6.0.5'
            },
            'dev_dependencies': {
                'flutter_test': {'sdk': 'flutter'},
                'build_runner': '^2.3.3',
                'json_serializable': '^6.6.0',
                'flutter_lints': '^2.0.0'
            }
        }
        
        with open(f"{output_dir}/pubspec.yaml", 'w') as f:
            yaml.dump(pubspec, f, default_flow_style=False)
    
    def _generate_flutter_api_client(self, output_dir: str):
        """Generate Flutter API client"""
        api_client_code = self.flutter_templates['api_client']
        
        lib_dir = f"{output_dir}/lib"
        os.makedirs(lib_dir, exist_ok=True)
        
        with open(f"{lib_dir}/scrollintel_api_client.dart", 'w') as f:
            f.write(api_client_code)
    
    def _generate_flutter_models(self, output_dir: str):
        """Generate Flutter data models"""
        models_code = self.flutter_templates['models']
        
        models_dir = f"{output_dir}/lib/models"
        os.makedirs(models_dir, exist_ok=True)
        
        with open(f"{models_dir}/scrollintel_models.dart", 'w') as f:
            f.write(models_code)
    
    def _generate_flutter_services(self, output_dir: str):
        """Generate Flutter services"""
        services_code = self.flutter_templates['services']
        
        services_dir = f"{output_dir}/lib/services"
        os.makedirs(services_dir, exist_ok=True)
        
        with open(f"{services_dir}/scrollintel_services.dart", 'w') as f:
            f.write(services_code)
    
    def _generate_flutter_widgets(self, output_dir: str):
        """Generate Flutter widgets"""
        widgets_code = self.flutter_templates['widgets']
        
        widgets_dir = f"{output_dir}/lib/widgets"
        os.makedirs(widgets_dir, exist_ok=True)
        
        with open(f"{widgets_dir}/scrollintel_widgets.dart", 'w') as f:
            f.write(widgets_code)
    
    def _generate_flutter_example(self, output_dir: str):
        """Generate Flutter example app"""
        example_code = '''
import 'package:flutter/material.dart';
import 'package:scrollintel_flutter_sdk/scrollintel_api_client.dart';
import 'package:scrollintel_flutter_sdk/widgets/scrollintel_widgets.dart';

void main() {
  runApp(ScrollIntelExampleApp());
}

class ScrollIntelExampleApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ScrollIntel Flutter SDK Example',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: ScrollIntelDashboard(),
    );
  }
}

class ScrollIntelDashboard extends StatefulWidget {
  @override
  _ScrollIntelDashboardState createState() => _ScrollIntelDashboardState();
}

class _ScrollIntelDashboardState extends State<ScrollIntelDashboard> {
  final ScrollIntelApiClient _apiClient = ScrollIntelApiClient();
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('ScrollIntel Dashboard'),
      ),
      body: Column(
        children: [
          ScrollIntelAgentCard(
            agentName: 'ScrollCTO',
            status: 'Active',
            onTap: () => _interactWithAgent('ScrollCTO'),
          ),
          ScrollIntelMetricsWidget(),
          ScrollIntelChatInterface(),
        ],
      ),
    );
  }
  
  void _interactWithAgent(String agentName) {
    // Implement agent interaction
  }
}
'''
        
        example_dir = f"{output_dir}/example"
        os.makedirs(example_dir, exist_ok=True)
        
        with open(f"{example_dir}/main.dart", 'w') as f:
            f.write(example_code)
    
    def _generate_rn_package_json(self, output_dir: str):
        """Generate React Native package.json"""
        package_json = {
            "name": "scrollintel-react-native-sdk",
            "version": "1.0.0",
            "description": "React Native SDK for ScrollIntel AI-CTO Platform",
            "main": "index.js",
            "scripts": {
                "test": "jest",
                "lint": "eslint .",
                "build": "tsc"
            },
            "dependencies": {
                "react": "^18.2.0",
                "react-native": "^0.72.0",
                "axios": "^1.4.0",
                "@react-native-async-storage/async-storage": "^1.19.0",
                "react-native-vector-icons": "^9.2.0"
            },
            "devDependencies": {
                "@types/react": "^18.2.0",
                "@types/react-native": "^0.72.0",
                "typescript": "^5.0.0",
                "jest": "^29.5.0",
                "eslint": "^8.42.0"
            },
            "peerDependencies": {
                "react": ">=16.8.0",
                "react-native": ">=0.60.0"
            }
        }
        
        with open(f"{output_dir}/package.json", 'w') as f:
            json.dump(package_json, f, indent=2)
    
    def _generate_rn_api_client(self, output_dir: str):
        """Generate React Native API client"""
        api_client_code = self.react_native_templates['api_client']
        
        src_dir = f"{output_dir}/src"
        os.makedirs(src_dir, exist_ok=True)
        
        with open(f"{src_dir}/ScrollIntelApiClient.ts", 'w') as f:
            f.write(api_client_code)
    
    def _generate_rn_models(self, output_dir: str):
        """Generate React Native models"""
        models_code = self.react_native_templates['models']
        
        models_dir = f"{output_dir}/src/models"
        os.makedirs(models_dir, exist_ok=True)
        
        with open(f"{models_dir}/index.ts", 'w') as f:
            f.write(models_code)
    
    def _generate_rn_services(self, output_dir: str):
        """Generate React Native services"""
        services_code = self.react_native_templates['services']
        
        services_dir = f"{output_dir}/src/services"
        os.makedirs(services_dir, exist_ok=True)
        
        with open(f"{services_dir}/index.ts", 'w') as f:
            f.write(services_code)
    
    def _generate_rn_components(self, output_dir: str):
        """Generate React Native components"""
        components_code = self.react_native_templates['components']
        
        components_dir = f"{output_dir}/src/components"
        os.makedirs(components_dir, exist_ok=True)
        
        with open(f"{components_dir}/index.tsx", 'w') as f:
            f.write(components_code)
    
    def _generate_rn_example(self, output_dir: str):
        """Generate React Native example app"""
        example_code = '''
import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { ScrollIntelProvider, ScrollIntelDashboard } from 'scrollintel-react-native-sdk';

const App = () => {
  return (
    <ScrollIntelProvider apiKey="your-api-key" baseUrl="https://api.scrollintel.com">
      <View style={styles.container}>
        <Text style={styles.title}>ScrollIntel Mobile App</Text>
        <ScrollIntelDashboard />
      </View>
    </ScrollIntelProvider>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    padding: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    marginBottom: 20,
  },
});

export default App;
'''
        
        example_dir = f"{output_dir}/example"
        os.makedirs(example_dir, exist_ok=True)
        
        with open(f"{example_dir}/App.tsx", 'w') as f:
            f.write(example_code)
    
    def _count_generated_files(self, directory: str) -> int:
        """Count generated files in directory"""
        count = 0
        for root, dirs, files in os.walk(directory):
            count += len(files)
        return count
    
    def _get_flutter_api_template(self) -> str:
        """Get Flutter API client template"""
        return '''
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:dio/dio.dart';

class ScrollIntelApiClient {
  final String baseUrl;
  final String apiKey;
  final Dio _dio;
  
  ScrollIntelApiClient({
    this.baseUrl = 'https://api.scrollintel.com',
    required this.apiKey,
  }) : _dio = Dio() {
    _dio.options.baseUrl = baseUrl;
    _dio.options.headers['Authorization'] = 'Bearer $apiKey';
    _dio.options.headers['Content-Type'] = 'application/json';
  }
  
  Future<Map<String, dynamic>> executeAgent(String agentName, Map<String, dynamic> payload) async {
    try {
      final response = await _dio.post('/api/agents/$agentName/execute', data: payload);
      return response.data;
    } catch (e) {
      throw ScrollIntelApiException('Failed to execute agent: $e');
    }
  }
  
  Future<List<Map<String, dynamic>>> getAgents() async {
    try {
      final response = await _dio.get('/api/agents');
      return List<Map<String, dynamic>>.from(response.data);
    } catch (e) {
      throw ScrollIntelApiException('Failed to get agents: $e');
    }
  }
  
  Future<Map<String, dynamic>> uploadFile(String filePath) async {
    try {
      final formData = FormData.fromMap({
        'file': await MultipartFile.fromFile(filePath),
      });
      final response = await _dio.post('/api/files/upload', data: formData);
      return response.data;
    } catch (e) {
      throw ScrollIntelApiException('Failed to upload file: $e');
    }
  }
  
  Future<Map<String, dynamic>> getSystemHealth() async {
    try {
      final response = await _dio.get('/health');
      return response.data;
    } catch (e) {
      throw ScrollIntelApiException('Failed to get system health: $e');
    }
  }
}

class ScrollIntelApiException implements Exception {
  final String message;
  ScrollIntelApiException(this.message);
  
  @override
  String toString() => 'ScrollIntelApiException: $message';
}
'''
    
    def _get_flutter_models_template(self) -> str:
        """Get Flutter models template"""
        return '''
import 'package:json_annotation/json_annotation.dart';
import 'package:equatable/equatable.dart';

part 'scrollintel_models.g.dart';

@JsonSerializable()
class ScrollIntelAgent extends Equatable {
  final String id;
  final String name;
  final String description;
  final String status;
  final Map<String, dynamic>? capabilities;
  
  const ScrollIntelAgent({
    required this.id,
    required this.name,
    required this.description,
    required this.status,
    this.capabilities,
  });
  
  factory ScrollIntelAgent.fromJson(Map<String, dynamic> json) => _$ScrollIntelAgentFromJson(json);
  Map<String, dynamic> toJson() => _$ScrollIntelAgentToJson(this);
  
  @override
  List<Object?> get props => [id, name, description, status, capabilities];
}

@JsonSerializable()
class ScrollIntelTask extends Equatable {
  final String id;
  final String agentId;
  final String type;
  final Map<String, dynamic> payload;
  final String status;
  final DateTime createdAt;
  final DateTime? completedAt;
  final Map<String, dynamic>? result;
  
  const ScrollIntelTask({
    required this.id,
    required this.agentId,
    required this.type,
    required this.payload,
    required this.status,
    required this.createdAt,
    this.completedAt,
    this.result,
  });
  
  factory ScrollIntelTask.fromJson(Map<String, dynamic> json) => _$ScrollIntelTaskFromJson(json);
  Map<String, dynamic> toJson() => _$ScrollIntelTaskToJson(this);
  
  @override
  List<Object?> get props => [id, agentId, type, payload, status, createdAt, completedAt, result];
}

@JsonSerializable()
class ScrollIntelMetrics extends Equatable {
  final double cpuUsage;
  final double memoryUsage;
  final int activeAgents;
  final int completedTasks;
  final DateTime timestamp;
  
  const ScrollIntelMetrics({
    required this.cpuUsage,
    required this.memoryUsage,
    required this.activeAgents,
    required this.completedTasks,
    required this.timestamp,
  });
  
  factory ScrollIntelMetrics.fromJson(Map<String, dynamic> json) => _$ScrollIntelMetricsFromJson(json);
  Map<String, dynamic> toJson() => _$ScrollIntelMetricsToJson(this);
  
  @override
  List<Object?> get props => [cpuUsage, memoryUsage, activeAgents, completedTasks, timestamp];
}
'''
    
    def _get_flutter_services_template(self) -> str:
        """Get Flutter services template"""
        return '''
import 'dart:async';
import 'package:flutter/foundation.dart';
import 'scrollintel_api_client.dart';
import '../models/scrollintel_models.dart';

class ScrollIntelService extends ChangeNotifier {
  final ScrollIntelApiClient _apiClient;
  List<ScrollIntelAgent> _agents = [];
  ScrollIntelMetrics? _metrics;
  bool _isLoading = false;
  String? _error;
  
  ScrollIntelService(this._apiClient);
  
  List<ScrollIntelAgent> get agents => _agents;
  ScrollIntelMetrics? get metrics => _metrics;
  bool get isLoading => _isLoading;
  String? get error => _error;
  
  Future<void> loadAgents() async {
    _setLoading(true);
    try {
      final agentsData = await _apiClient.getAgents();
      _agents = agentsData.map((data) => ScrollIntelAgent.fromJson(data)).toList();
      _error = null;
    } catch (e) {
      _error = e.toString();
    } finally {
      _setLoading(false);
    }
  }
  
  Future<Map<String, dynamic>> executeAgent(String agentName, Map<String, dynamic> payload) async {
    _setLoading(true);
    try {
      final result = await _apiClient.executeAgent(agentName, payload);
      _error = null;
      return result;
    } catch (e) {
      _error = e.toString();
      rethrow;
    } finally {
      _setLoading(false);
    }
  }
  
  Future<void> loadMetrics() async {
    try {
      final healthData = await _apiClient.getSystemHealth();
      _metrics = ScrollIntelMetrics.fromJson(healthData);
      _error = null;
    } catch (e) {
      _error = e.toString();
    }
    notifyListeners();
  }
  
  void _setLoading(bool loading) {
    _isLoading = loading;
    notifyListeners();
  }
  
  void startMetricsPolling() {
    Timer.periodic(Duration(seconds: 30), (timer) {
      loadMetrics();
    });
  }
}
'''
    
    def _get_flutter_widgets_template(self) -> str:
        """Get Flutter widgets template"""
        return '''
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../services/scrollintel_services.dart';
import '../models/scrollintel_models.dart';

class ScrollIntelAgentCard extends StatelessWidget {
  final String agentName;
  final String status;
  final VoidCallback? onTap;
  
  const ScrollIntelAgentCard({
    Key? key,
    required this.agentName,
    required this.status,
    this.onTap,
  }) : super(key: key);
  
  @override
  Widget build(BuildContext context) {
    return Card(
      child: ListTile(
        leading: CircleAvatar(
          backgroundColor: _getStatusColor(status),
          child: Icon(Icons.smart_toy, color: Colors.white),
        ),
        title: Text(agentName),
        subtitle: Text('Status: $status'),
        trailing: Icon(Icons.arrow_forward_ios),
        onTap: onTap,
      ),
    );
  }
  
  Color _getStatusColor(String status) {
    switch (status.toLowerCase()) {
      case 'active':
        return Colors.green;
      case 'busy':
        return Colors.orange;
      case 'error':
        return Colors.red;
      default:
        return Colors.grey;
    }
  }
}

class ScrollIntelMetricsWidget extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Consumer<ScrollIntelService>(
      builder: (context, service, child) {
        final metrics = service.metrics;
        if (metrics == null) {
          return Card(
            child: Padding(
              padding: EdgeInsets.all(16),
              child: Text('Loading metrics...'),
            ),
          );
        }
        
        return Card(
          child: Padding(
            padding: EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('System Metrics', style: Theme.of(context).textTheme.headline6),
                SizedBox(height: 8),
                _buildMetricRow('CPU Usage', '${metrics.cpuUsage.toStringAsFixed(1)}%'),
                _buildMetricRow('Memory Usage', '${metrics.memoryUsage.toStringAsFixed(1)}%'),
                _buildMetricRow('Active Agents', '${metrics.activeAgents}'),
                _buildMetricRow('Completed Tasks', '${metrics.completedTasks}'),
              ],
            ),
          ),
        );
      },
    );
  }
  
  Widget _buildMetricRow(String label, String value) {
    return Padding(
      padding: EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label),
          Text(value, style: TextStyle(fontWeight: FontWeight.bold)),
        ],
      ),
    );
  }
}

class ScrollIntelChatInterface extends StatefulWidget {
  @override
  _ScrollIntelChatInterfaceState createState() => _ScrollIntelChatInterfaceState();
}

class _ScrollIntelChatInterfaceState extends State<ScrollIntelChatInterface> {
  final TextEditingController _messageController = TextEditingController();
  final List<ChatMessage> _messages = [];
  
  @override
  Widget build(BuildContext context) {
    return Card(
      child: Container(
        height: 300,
        child: Column(
          children: [
            Padding(
              padding: EdgeInsets.all(16),
              child: Text('Chat with ScrollIntel', style: Theme.of(context).textTheme.headline6),
            ),
            Expanded(
              child: ListView.builder(
                itemCount: _messages.length,
                itemBuilder: (context, index) {
                  final message = _messages[index];
                  return ListTile(
                    leading: CircleAvatar(
                      child: Icon(message.isUser ? Icons.person : Icons.smart_toy),
                    ),
                    title: Text(message.text),
                    subtitle: Text(message.timestamp.toString()),
                  );
                },
              ),
            ),
            Padding(
              padding: EdgeInsets.all(16),
              child: Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: _messageController,
                      decoration: InputDecoration(
                        hintText: 'Type your message...',
                        border: OutlineInputBorder(),
                      ),
                    ),
                  ),
                  SizedBox(width: 8),
                  IconButton(
                    icon: Icon(Icons.send),
                    onPressed: _sendMessage,
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
  
  void _sendMessage() {
    if (_messageController.text.isNotEmpty) {
      setState(() {
        _messages.add(ChatMessage(
          text: _messageController.text,
          isUser: true,
          timestamp: DateTime.now(),
        ));
      });
      _messageController.clear();
      
      // Simulate AI response
      Future.delayed(Duration(seconds: 1), () {
        setState(() {
          _messages.add(ChatMessage(
            text: 'I understand your request. Let me process that for you.',
            isUser: false,
            timestamp: DateTime.now(),
          ));
        });
      });
    }
  }
}

class ChatMessage {
  final String text;
  final bool isUser;
  final DateTime timestamp;
  
  ChatMessage({
    required this.text,
    required this.isUser,
    required this.timestamp,
  });
}
'''
    
    def _get_rn_api_template(self) -> str:
        """Get React Native API client template"""
        return '''
import axios, { AxiosInstance, AxiosResponse } from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';

export interface ScrollIntelConfig {
  baseUrl: string;
  apiKey: string;
  timeout?: number;
}

export class ScrollIntelApiClient {
  private client: AxiosInstance;
  private config: ScrollIntelConfig;
  
  constructor(config: ScrollIntelConfig) {
    this.config = config;
    this.client = axios.create({
      baseURL: config.baseUrl,
      timeout: config.timeout || 30000,
      headers: {
        'Authorization': `Bearer ${config.apiKey}`,
        'Content-Type': 'application/json',
      },
    });
    
    this.setupInterceptors();
  }
  
  private setupInterceptors(): void {
    this.client.interceptors.request.use(
      async (config) => {
        const token = await AsyncStorage.getItem('scrollintel_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );
    
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        if (error.response?.status === 401) {
          // Handle unauthorized access
          AsyncStorage.removeItem('scrollintel_token');
        }
        return Promise.reject(error);
      }
    );
  }
  
  async executeAgent(agentName: string, payload: any): Promise<any> {
    try {
      const response: AxiosResponse = await this.client.post(
        `/api/agents/${agentName}/execute`,
        payload
      );
      return response.data;
    } catch (error) {
      throw new ScrollIntelApiError(`Failed to execute agent: ${error}`);
    }
  }
  
  async getAgents(): Promise<any[]> {
    try {
      const response: AxiosResponse = await this.client.get('/api/agents');
      return response.data;
    } catch (error) {
      throw new ScrollIntelApiError(`Failed to get agents: ${error}`);
    }
  }
  
  async uploadFile(fileUri: string): Promise<any> {
    try {
      const formData = new FormData();
      formData.append('file', {
        uri: fileUri,
        type: 'application/octet-stream',
        name: 'upload.file',
      } as any);
      
      const response: AxiosResponse = await this.client.post(
        '/api/files/upload',
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );
      return response.data;
    } catch (error) {
      throw new ScrollIntelApiError(`Failed to upload file: ${error}`);
    }
  }
  
  async getSystemHealth(): Promise<any> {
    try {
      const response: AxiosResponse = await this.client.get('/health');
      return response.data;
    } catch (error) {
      throw new ScrollIntelApiError(`Failed to get system health: ${error}`);
    }
  }
}

export class ScrollIntelApiError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ScrollIntelApiError';
  }
}
'''
    
    def _get_rn_models_template(self) -> str:
        """Get React Native models template"""
        return '''
export interface ScrollIntelAgent {
  id: string;
  name: string;
  description: string;
  status: 'active' | 'busy' | 'error' | 'inactive';
  capabilities?: Record<string, any>;
}

export interface ScrollIntelTask {
  id: string;
  agentId: string;
  type: string;
  payload: Record<string, any>;
  status: 'pending' | 'running' | 'completed' | 'failed';
  createdAt: Date;
  completedAt?: Date;
  result?: Record<string, any>;
}

export interface ScrollIntelMetrics {
  cpuUsage: number;
  memoryUsage: number;
  activeAgents: number;
  completedTasks: number;
  timestamp: Date;
}

export interface ChatMessage {
  id: string;
  text: string;
  isUser: boolean;
  timestamp: Date;
  agentName?: string;
}

export interface FileUploadResult {
  id: string;
  filename: string;
  size: number;
  uploadedAt: Date;
  processingStatus: 'pending' | 'processing' | 'completed' | 'failed';
}

export interface SystemHealth {
  status: 'healthy' | 'degraded' | 'unhealthy';
  services: Record<string, ServiceStatus>;
  uptime: number;
  version: string;
}

export interface ServiceStatus {
  status: 'up' | 'down' | 'degraded';
  responseTime?: number;
  lastCheck: Date;
}
'''
    
    def _get_rn_services_template(self) -> str:
        """Get React Native services template"""
        return '''
import { ScrollIntelApiClient, ScrollIntelConfig } from './ScrollIntelApiClient';
import { ScrollIntelAgent, ScrollIntelMetrics, ChatMessage } from '../models';

export class ScrollIntelService {
  private apiClient: ScrollIntelApiClient;
  private agents: ScrollIntelAgent[] = [];
  private metrics: ScrollIntelMetrics | null = null;
  private isLoading: boolean = false;
  private error: string | null = null;
  
  constructor(config: ScrollIntelConfig) {
    this.apiClient = new ScrollIntelApiClient(config);
  }
  
  async loadAgents(): Promise<ScrollIntelAgent[]> {
    this.setLoading(true);
    try {
      const agentsData = await this.apiClient.getAgents();
      this.agents = agentsData;
      this.error = null;
      return this.agents;
    } catch (error) {
      this.error = error instanceof Error ? error.message : 'Unknown error';
      throw error;
    } finally {
      this.setLoading(false);
    }
  }
  
  async executeAgent(agentName: string, payload: any): Promise<any> {
    this.setLoading(true);
    try {
      const result = await this.apiClient.executeAgent(agentName, payload);
      this.error = null;
      return result;
    } catch (error) {
      this.error = error instanceof Error ? error.message : 'Unknown error';
      throw error;
    } finally {
      this.setLoading(false);
    }
  }
  
  async loadMetrics(): Promise<ScrollIntelMetrics> {
    try {
      const healthData = await this.apiClient.getSystemHealth();
      this.metrics = {
        cpuUsage: healthData.cpuUsage || 0,
        memoryUsage: healthData.memoryUsage || 0,
        activeAgents: healthData.activeAgents || 0,
        completedTasks: healthData.completedTasks || 0,
        timestamp: new Date(),
      };
      this.error = null;
      return this.metrics;
    } catch (error) {
      this.error = error instanceof Error ? error.message : 'Unknown error';
      throw error;
    }
  }
  
  async uploadFile(fileUri: string): Promise<any> {
    this.setLoading(true);
    try {
      const result = await this.apiClient.uploadFile(fileUri);
      this.error = null;
      return result;
    } catch (error) {
      this.error = error instanceof Error ? error.message : 'Unknown error';
      throw error;
    } finally {
      this.setLoading(false);
    }
  }
  
  getAgents(): ScrollIntelAgent[] {
    return this.agents;
  }
  
  getMetrics(): ScrollIntelMetrics | null {
    return this.metrics;
  }
  
  getIsLoading(): boolean {
    return this.isLoading;
  }
  
  getError(): string | null {
    return this.error;
  }
  
  private setLoading(loading: boolean): void {
    this.isLoading = loading;
  }
  
  startMetricsPolling(intervalMs: number = 30000): () => void {
    const interval = setInterval(() => {
      this.loadMetrics().catch(console.error);
    }, intervalMs);
    
    return () => clearInterval(interval);
  }
}
'''
    
    def _get_rn_components_template(self) -> str:
        """Get React Native components template"""
        return '''
import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  FlatList,
  TextInput,
  StyleSheet,
  ActivityIndicator,
} from 'react-native';
import Icon from 'react-native-vector-icons/MaterialIcons';
import { ScrollIntelService } from '../services';
import { ScrollIntelAgent, ScrollIntelMetrics, ChatMessage } from '../models';

interface ScrollIntelProviderProps {
  children: React.ReactNode;
  apiKey: string;
  baseUrl: string;
}

const ScrollIntelContext = React.createContext<ScrollIntelService | null>(null);

export const ScrollIntelProvider: React.FC<ScrollIntelProviderProps> = ({
  children,
  apiKey,
  baseUrl,
}) => {
  const [service] = useState(() => new ScrollIntelService({ apiKey, baseUrl }));
  
  return (
    <ScrollIntelContext.Provider value={service}>
      {children}
    </ScrollIntelContext.Provider>
  );
};

export const useScrollIntel = (): ScrollIntelService => {
  const service = React.useContext(ScrollIntelContext);
  if (!service) {
    throw new Error('useScrollIntel must be used within ScrollIntelProvider');
  }
  return service;
};

interface AgentCardProps {
  agent: ScrollIntelAgent;
  onPress?: (agent: ScrollIntelAgent) => void;
}

export const AgentCard: React.FC<AgentCardProps> = ({ agent, onPress }) => {
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return '#4CAF50';
      case 'busy':
        return '#FF9800';
      case 'error':
        return '#F44336';
      default:
        return '#9E9E9E';
    }
  };
  
  return (
    <TouchableOpacity
      style={styles.agentCard}
      onPress={() => onPress?.(agent)}
    >
      <View style={styles.agentHeader}>
        <View style={[styles.statusIndicator, { backgroundColor: getStatusColor(agent.status) }]} />
        <Text style={styles.agentName}>{agent.name}</Text>
      </View>
      <Text style={styles.agentDescription}>{agent.description}</Text>
      <Text style={styles.agentStatus}>Status: {agent.status}</Text>
    </TouchableOpacity>
  );
};

export const MetricsWidget: React.FC = () => {
  const service = useScrollIntel();
  const [metrics, setMetrics] = useState<ScrollIntelMetrics | null>(null);
  const [loading, setLoading] = useState(false);
  
  useEffect(() => {
    loadMetrics();
    const stopPolling = service.startMetricsPolling();
    return stopPolling;
  }, []);
  
  const loadMetrics = async () => {
    setLoading(true);
    try {
      const metricsData = await service.loadMetrics();
      setMetrics(metricsData);
    } catch (error) {
      console.error('Failed to load metrics:', error);
    } finally {
      setLoading(false);
    }
  };
  
  if (loading && !metrics) {
    return (
      <View style={styles.metricsContainer}>
        <ActivityIndicator size="large" />
        <Text>Loading metrics...</Text>
      </View>
    );
  }
  
  if (!metrics) {
    return (
      <View style={styles.metricsContainer}>
        <Text>No metrics available</Text>
      </View>
    );
  }
  
  return (
    <View style={styles.metricsContainer}>
      <Text style={styles.metricsTitle}>System Metrics</Text>
      <View style={styles.metricsGrid}>
        <View style={styles.metricItem}>
          <Text style={styles.metricLabel}>CPU Usage</Text>
          <Text style={styles.metricValue}>{metrics.cpuUsage.toFixed(1)}%</Text>
        </View>
        <View style={styles.metricItem}>
          <Text style={styles.metricLabel}>Memory Usage</Text>
          <Text style={styles.metricValue}>{metrics.memoryUsage.toFixed(1)}%</Text>
        </View>
        <View style={styles.metricItem}>
          <Text style={styles.metricLabel}>Active Agents</Text>
          <Text style={styles.metricValue}>{metrics.activeAgents}</Text>
        </View>
        <View style={styles.metricItem}>
          <Text style={styles.metricLabel}>Completed Tasks</Text>
          <Text style={styles.metricValue}>{metrics.completedTasks}</Text>
        </View>
      </View>
    </View>
  );
};

export const ScrollIntelDashboard: React.FC = () => {
  const service = useScrollIntel();
  const [agents, setAgents] = useState<ScrollIntelAgent[]>([]);
  const [loading, setLoading] = useState(false);
  
  useEffect(() => {
    loadAgents();
  }, []);
  
  const loadAgents = async () => {
    setLoading(true);
    try {
      const agentsData = await service.loadAgents();
      setAgents(agentsData);
    } catch (error) {
      console.error('Failed to load agents:', error);
    } finally {
      setLoading(false);
    }
  };
  
  const handleAgentPress = (agent: ScrollIntelAgent) => {
    console.log('Agent pressed:', agent.name);
    // Implement agent interaction
  };
  
  return (
    <View style={styles.dashboard}>
      <MetricsWidget />
      <View style={styles.agentsSection}>
        <Text style={styles.sectionTitle}>AI Agents</Text>
        {loading ? (
          <ActivityIndicator size="large" />
        ) : (
          <FlatList
            data={agents}
            keyExtractor={(item) => item.id}
            renderItem={({ item }) => (
              <AgentCard agent={item} onPress={handleAgentPress} />
            )}
          />
        )}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  dashboard: {
    flex: 1,
    padding: 16,
    backgroundColor: '#f5f5f5',
  },
  agentCard: {
    backgroundColor: 'white',
    padding: 16,
    marginVertical: 8,
    borderRadius: 8,
    elevation: 2,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  agentHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  statusIndicator: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginRight: 8,
  },
  agentName: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  agentDescription: {
    fontSize: 14,
    color: '#666',
    marginBottom: 4,
  },
  agentStatus: {
    fontSize: 12,
    color: '#999',
  },
  metricsContainer: {
    backgroundColor: 'white',
    padding: 16,
    borderRadius: 8,
    marginBottom: 16,
    elevation: 2,
  },
  metricsTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 16,
  },
  metricsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  metricItem: {
    width: '48%',
    marginBottom: 12,
  },
  metricLabel: {
    fontSize: 12,
    color: '#666',
  },
  metricValue: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
  },
  agentsSection: {
    flex: 1,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 16,
  },
});
'''

# Create the mobile SDK generator instance
mobile_sdk_generator = MobileSDKGenerator()