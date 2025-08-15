"use client";

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Play, 
  Clock, 
  User, 
  Search,
  BookOpen,
  Video,
  Star,
  Filter
} from 'lucide-react';

interface VideoTutorial {
  id: string;
  title: string;
  description: string;
  duration: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  category: string;
  thumbnail: string;
  videoUrl: string;
  views: number;
  rating: number;
  instructor: string;
  tags: string[];
  createdAt: string;
}

interface TutorialCategory {
  id: string;
  name: string;
  description: string;
  videoCount: number;
  icon: React.ReactNode;
}

const VideoTutorials: React.FC = () => {
  const [tutorials, setTutorials] = useState<VideoTutorial[]>([]);
  const [categories, setCategories] = useState<TutorialCategory[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedDifficulty, setSelectedDifficulty] = useState<string>('all');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadTutorials();
  }, []);

  const loadTutorials = async () => {
    setLoading(true);
    
    // Mock data - in real implementation, this would come from API
    const mockCategories: TutorialCategory[] = [
      {
        id: 'getting-started',
        name: 'Getting Started',
        description: 'Learn the basics of ScrollIntel',
        videoCount: 8,
        icon: <BookOpen className="h-5 w-5" />
      },
      {
        id: 'data-upload',
        name: 'Data Upload',
        description: 'How to upload and manage your data',
        videoCount: 5,
        icon: <Video className="h-5 w-5" />
      },
      {
        id: 'ai-agents',
        name: 'AI Agents',
        description: 'Working with different AI agents',
        videoCount: 12,
        icon: <User className="h-5 w-5" />
      },
      {
        id: 'api-integration',
        name: 'API Integration',
        description: 'Integrating ScrollIntel API',
        videoCount: 6,
        icon: <Play className="h-5 w-5" />
      },
      {
        id: 'advanced-features',
        name: 'Advanced Features',
        description: 'Advanced tips and techniques',
        videoCount: 9,
        icon: <Star className="h-5 w-5" />
      }
    ];

    const mockTutorials: VideoTutorial[] = [
      {
        id: '1',
        title: 'ScrollIntel Overview: Your AI-Powered CTO',
        description: 'Get an overview of ScrollIntel\'s capabilities and learn how our AI agents can replace human technical experts.',
        duration: '5:32',
        difficulty: 'beginner',
        category: 'getting-started',
        thumbnail: '/api/placeholder/320/180',
        videoUrl: 'https://example.com/video1',
        views: 15420,
        rating: 4.8,
        instructor: 'Sarah Chen',
        tags: ['overview', 'introduction', 'ai-agents'],
        createdAt: '2024-01-15'
      },
      {
        id: '2',
        title: 'Uploading Your First Dataset',
        description: 'Step-by-step guide to uploading data files and understanding the supported formats.',
        duration: '3:45',
        difficulty: 'beginner',
        category: 'data-upload',
        thumbnail: '/api/placeholder/320/180',
        videoUrl: 'https://example.com/video2',
        views: 12350,
        rating: 4.7,
        instructor: 'Mike Rodriguez',
        tags: ['upload', 'csv', 'excel', 'data'],
        createdAt: '2024-01-14'
      },
      {
        id: '3',
        title: 'Working with the CTO Agent',
        description: 'Learn how to interact with the CTO Agent for strategic technology decisions and architecture planning.',
        duration: '8:15',
        difficulty: 'intermediate',
        category: 'ai-agents',
        thumbnail: '/api/placeholder/320/180',
        videoUrl: 'https://example.com/video3',
        views: 9870,
        rating: 4.9,
        instructor: 'Dr. Emily Watson',
        tags: ['cto', 'strategy', 'architecture'],
        createdAt: '2024-01-13'
      },
      {
        id: '4',
        title: 'Data Analysis with the Data Scientist Agent',
        description: 'Discover how to perform comprehensive data analysis using our AI Data Scientist.',
        duration: '12:30',
        difficulty: 'intermediate',
        category: 'ai-agents',
        thumbnail: '/api/placeholder/320/180',
        videoUrl: 'https://example.com/video4',
        views: 11200,
        rating: 4.8,
        instructor: 'Alex Thompson',
        tags: ['data-science', 'analysis', 'statistics'],
        createdAt: '2024-01-12'
      },
      {
        id: '5',
        title: 'Building ML Models with the ML Engineer Agent',
        description: 'Complete walkthrough of building, training, and deploying machine learning models.',
        duration: '18:45',
        difficulty: 'advanced',
        category: 'ai-agents',
        thumbnail: '/api/placeholder/320/180',
        videoUrl: 'https://example.com/video5',
        views: 8650,
        rating: 4.9,
        instructor: 'Dr. James Liu',
        tags: ['machine-learning', 'models', 'deployment'],
        createdAt: '2024-01-11'
      },
      {
        id: '6',
        title: 'Creating Dashboards with the BI Agent',
        description: 'Learn to create stunning business intelligence dashboards and reports.',
        duration: '10:20',
        difficulty: 'intermediate',
        category: 'ai-agents',
        thumbnail: '/api/placeholder/320/180',
        videoUrl: 'https://example.com/video6',
        views: 7890,
        rating: 4.6,
        instructor: 'Lisa Park',
        tags: ['business-intelligence', 'dashboards', 'visualization'],
        createdAt: '2024-01-10'
      },
      {
        id: '7',
        title: 'API Integration Basics',
        description: 'Get started with the ScrollIntel API and learn authentication and basic requests.',
        duration: '7:55',
        difficulty: 'intermediate',
        category: 'api-integration',
        thumbnail: '/api/placeholder/320/180',
        videoUrl: 'https://example.com/video7',
        views: 6540,
        rating: 4.7,
        instructor: 'David Kim',
        tags: ['api', 'integration', 'authentication'],
        createdAt: '2024-01-09'
      },
      {
        id: '8',
        title: 'Advanced API Usage and SDKs',
        description: 'Deep dive into advanced API features and using our Python and JavaScript SDKs.',
        duration: '15:30',
        difficulty: 'advanced',
        category: 'api-integration',
        thumbnail: '/api/placeholder/320/180',
        videoUrl: 'https://example.com/video8',
        views: 4320,
        rating: 4.8,
        instructor: 'Rachel Green',
        tags: ['api', 'sdk', 'python', 'javascript'],
        createdAt: '2024-01-08'
      },
      {
        id: '9',
        title: 'Exporting and Sharing Results',
        description: 'Learn how to export your analysis results and share them with your team.',
        duration: '6:10',
        difficulty: 'beginner',
        category: 'getting-started',
        thumbnail: '/api/placeholder/320/180',
        videoUrl: 'https://example.com/video9',
        views: 9100,
        rating: 4.5,
        instructor: 'Tom Wilson',
        tags: ['export', 'pdf', 'excel', 'sharing'],
        createdAt: '2024-01-07'
      },
      {
        id: '10',
        title: 'Team Collaboration Features',
        description: 'Discover how to collaborate with your team using workspaces and shared projects.',
        duration: '9:40',
        difficulty: 'intermediate',
        category: 'advanced-features',
        thumbnail: '/api/placeholder/320/180',
        videoUrl: 'https://example.com/video10',
        views: 5670,
        rating: 4.7,
        instructor: 'Anna Martinez',
        tags: ['collaboration', 'teams', 'workspaces'],
        createdAt: '2024-01-06'
      }
    ];

    setCategories(mockCategories);
    setTutorials(mockTutorials);
    setLoading(false);
  };

  const filteredTutorials = tutorials.filter(tutorial => {
    const matchesCategory = selectedCategory === 'all' || tutorial.category === selectedCategory;
    const matchesDifficulty = selectedDifficulty === 'all' || tutorial.difficulty === selectedDifficulty;
    const matchesSearch = searchQuery === '' || 
      tutorial.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      tutorial.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
      tutorial.tags.some(tag => tag.toLowerCase().includes(searchQuery.toLowerCase()));
    
    return matchesCategory && matchesDifficulty && matchesSearch;
  });

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner': return 'bg-green-100 text-green-800';
      case 'intermediate': return 'bg-yellow-100 text-yellow-800';
      case 'advanced': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const formatViews = (views: number) => {
    if (views >= 1000) {
      return `${(views / 1000).toFixed(1)}K`;
    }
    return views.toString();
  };

  return (
    <div className="container mx-auto p-6 max-w-7xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Video Tutorials</h1>
        <p className="text-gray-600">Learn ScrollIntel with step-by-step video guides</p>
      </div>

      {/* Search and Filters */}
      <div className="mb-8 space-y-4">
        <div className="flex flex-col md:flex-row gap-4">
          <div className="flex-1">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-4 w-4" />
              <Input
                placeholder="Search tutorials..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10"
              />
            </div>
          </div>
          <div className="flex gap-2">
            <select
              value={selectedDifficulty}
              onChange={(e) => setSelectedDifficulty(e.target.value)}
              className="px-3 py-2 border rounded-md bg-white"
            >
              <option value="all">All Levels</option>
              <option value="beginner">Beginner</option>
              <option value="intermediate">Intermediate</option>
              <option value="advanced">Advanced</option>
            </select>
          </div>
        </div>
      </div>

      <div className="grid lg:grid-cols-4 gap-8">
        {/* Categories Sidebar */}
        <div className="lg:col-span-1">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Filter className="h-5 w-5" />
                Categories
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <Button
                variant={selectedCategory === 'all' ? 'default' : 'ghost'}
                className="w-full justify-start"
                onClick={() => setSelectedCategory('all')}
              >
                All Categories ({tutorials.length})
              </Button>
              {categories.map((category) => (
                <Button
                  key={category.id}
                  variant={selectedCategory === category.id ? 'default' : 'ghost'}
                  className="w-full justify-start"
                  onClick={() => setSelectedCategory(category.id)}
                >
                  <div className="flex items-center gap-2">
                    {category.icon}
                    <span>{category.name}</span>
                    <Badge variant="secondary" className="ml-auto">
                      {category.videoCount}
                    </Badge>
                  </div>
                </Button>
              ))}
            </CardContent>
          </Card>

          {/* Featured Tutorial */}
          <Card className="mt-6">
            <CardHeader>
              <CardTitle className="text-lg">Featured Tutorial</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="aspect-video bg-gray-200 rounded-lg flex items-center justify-center">
                  <Play className="h-8 w-8 text-gray-400" />
                </div>
                <h4 className="font-medium">Getting Started with ScrollIntel</h4>
                <p className="text-sm text-gray-600">Complete beginner's guide to using ScrollIntel</p>
                <div className="flex items-center gap-2 text-sm text-gray-500">
                  <Clock className="h-4 w-4" />
                  <span>5:32</span>
                  <Badge className="bg-green-100 text-green-800">Beginner</Badge>
                </div>
                <Button size="sm" className="w-full">
                  <Play className="h-4 w-4 mr-2" />
                  Watch Now
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Tutorials Grid */}
        <div className="lg:col-span-3">
          {loading ? (
            <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-6">
              {[...Array(6)].map((_, i) => (
                <Card key={i} className="animate-pulse">
                  <div className="aspect-video bg-gray-200 rounded-t-lg"></div>
                  <CardContent className="p-4">
                    <div className="h-4 bg-gray-200 rounded mb-2"></div>
                    <div className="h-3 bg-gray-200 rounded mb-4"></div>
                    <div className="flex justify-between">
                      <div className="h-3 bg-gray-200 rounded w-16"></div>
                      <div className="h-3 bg-gray-200 rounded w-20"></div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : (
            <>
              <div className="mb-6">
                <p className="text-gray-600">
                  Showing {filteredTutorials.length} tutorial{filteredTutorials.length !== 1 ? 's' : ''}
                  {selectedCategory !== 'all' && ` in ${categories.find(c => c.id === selectedCategory)?.name}`}
                  {selectedDifficulty !== 'all' && ` for ${selectedDifficulty} level`}
                </p>
              </div>

              <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-6">
                {filteredTutorials.map((tutorial) => (
                  <Card key={tutorial.id} className="group hover:shadow-lg transition-shadow cursor-pointer">
                    <div className="relative aspect-video bg-gray-200 rounded-t-lg overflow-hidden">
                      <img
                        src={tutorial.thumbnail}
                        alt={tutorial.title}
                        className="w-full h-full object-cover"
                      />
                      <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-30 transition-all flex items-center justify-center">
                        <Play className="h-12 w-12 text-white opacity-0 group-hover:opacity-100 transition-opacity" />
                      </div>
                      <div className="absolute bottom-2 right-2 bg-black bg-opacity-75 text-white text-xs px-2 py-1 rounded">
                        {tutorial.duration}
                      </div>
                    </div>
                    <CardContent className="p-4">
                      <div className="flex items-start justify-between mb-2">
                        <Badge className={getDifficultyColor(tutorial.difficulty)}>
                          {tutorial.difficulty}
                        </Badge>
                        <div className="flex items-center gap-1 text-sm text-gray-500">
                          <Star className="h-3 w-3 fill-yellow-400 text-yellow-400" />
                          <span>{tutorial.rating}</span>
                        </div>
                      </div>
                      
                      <h3 className="font-medium mb-2 line-clamp-2 group-hover:text-blue-600 transition-colors">
                        {tutorial.title}
                      </h3>
                      
                      <p className="text-sm text-gray-600 mb-3 line-clamp-2">
                        {tutorial.description}
                      </p>
                      
                      <div className="flex items-center justify-between text-sm text-gray-500 mb-3">
                        <div className="flex items-center gap-1">
                          <User className="h-3 w-3" />
                          <span>{tutorial.instructor}</span>
                        </div>
                        <span>{formatViews(tutorial.views)} views</span>
                      </div>
                      
                      <div className="flex flex-wrap gap-1 mb-3">
                        {tutorial.tags.slice(0, 3).map((tag) => (
                          <Badge key={tag} variant="outline" className="text-xs">
                            {tag}
                          </Badge>
                        ))}
                      </div>
                      
                      <Button className="w-full" size="sm">
                        <Play className="h-4 w-4 mr-2" />
                        Watch Tutorial
                      </Button>
                    </CardContent>
                  </Card>
                ))}
              </div>

              {filteredTutorials.length === 0 && (
                <div className="text-center py-12">
                  <Video className="h-16 w-16 mx-auto mb-4 text-gray-400" />
                  <h3 className="text-lg font-medium mb-2">No tutorials found</h3>
                  <p className="text-gray-600 mb-4">
                    Try adjusting your search criteria or browse all categories
                  </p>
                  <Button onClick={() => {
                    setSearchQuery('');
                    setSelectedCategory('all');
                    setSelectedDifficulty('all');
                  }}>
                    Clear Filters
                  </Button>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
};

export default VideoTutorials;