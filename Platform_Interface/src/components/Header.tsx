import React from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Badge } from './ui/badge';
import { Avatar, AvatarFallback, AvatarImage } from './ui/avatar';
import { Switch } from './ui/switch';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from './ui/dropdown-menu';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from './ui/popover';
import {
  Search,
  Bell,
  Menu,
  Sun,
  Moon,
  Settings,
  LogOut,
  User,
  HelpCircle,
  Zap,
  AlertCircle,
  CheckCircle
} from 'lucide-react';

interface HeaderProps {
  isCollapsed: boolean;
  isDarkMode: boolean;
  onToggleSidebar: () => void;
  onToggleTheme: () => void;
}

const notifications = [
  {
    id: 1,
    type: 'success',
    title: 'Model Training Complete',
    message: 'Customer segmentation model finished training with 94.2% accuracy',
    time: '2 min ago',
    read: false
  },
  {
    id: 2,
    type: 'warning',
    title: 'High API Usage',
    message: 'OpenAI API usage is at 85% of monthly limit',
    time: '15 min ago',
    read: false
  },
  {
    id: 3,
    type: 'info',
    title: 'System Update',
    message: 'ScrollIntel platform updated to version 2.4.1',
    time: '1 hour ago',
    read: true
  }
];

export function Header({ isCollapsed, isDarkMode, onToggleSidebar, onToggleTheme }: HeaderProps) {
  const unreadCount = notifications.filter(n => !n.read).length;

  const getNotificationIcon = (type: string) => {
    switch (type) {
      case 'success': return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'warning': return <AlertCircle className="w-4 h-4 text-yellow-500" />;
      case 'info': return <Zap className="w-4 h-4 text-blue-500" />;
      default: return <Bell className="w-4 h-4" />;
    }
  };

  return (
    <header className="h-16 border-b border-border bg-card/50 backdrop-blur supports-[backdrop-filter]:bg-card/50">
      <div className="flex items-center justify-between h-full px-4">
        {/* Left Section */}
        <div className="flex items-center gap-4">
          <Button
            variant="ghost"
            size="sm"
            onClick={onToggleSidebar}
            className="p-2"
          >
            <Menu className="w-4 h-4" />
          </Button>
          
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input
              placeholder="Search agents, prompts, or analytics..."
              className="w-96 pl-10 bg-muted/50"
            />
          </div>
        </div>

        {/* Right Section */}
        <div className="flex items-center gap-3">
          {/* Theme Toggle */}
          <div className="flex items-center gap-2">
            <Sun className="w-4 h-4 text-muted-foreground" />
            <Switch
              checked={isDarkMode}
              onCheckedChange={onToggleTheme}
            />
            <Moon className="w-4 h-4 text-muted-foreground" />
          </div>

          {/* Notifications */}
          <Popover>
            <PopoverTrigger asChild>
              <Button variant="ghost" size="sm" className="p-2 relative">
                <Bell className="w-4 h-4" />
                {unreadCount > 0 && (
                  <Badge className="absolute -top-1 -right-1 w-5 h-5 text-xs p-0 flex items-center justify-center bg-red-500">
                    {unreadCount}
                  </Badge>
                )}
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-80 p-0" align="end">
              <div className="p-4 border-b border-border">
                <h3 className="font-medium">Notifications</h3>
                <p className="text-sm text-muted-foreground">{unreadCount} unread notifications</p>
              </div>
              <div className="max-h-96 overflow-y-auto">
                {notifications.map((notification) => (
                  <div
                    key={notification.id}
                    className={`p-4 border-b border-border last:border-b-0 hover:bg-muted/50 cursor-pointer ${
                      !notification.read ? 'bg-muted/30' : ''
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      {getNotificationIcon(notification.type)}
                      <div className="flex-1">
                        <div className="flex items-center justify-between">
                          <h4 className="text-sm font-medium">{notification.title}</h4>
                          {!notification.read && (
                            <div className="w-2 h-2 bg-blue-500 rounded-full" />
                          )}
                        </div>
                        <p className="text-sm text-muted-foreground mt-1">{notification.message}</p>
                        <p className="text-xs text-muted-foreground mt-1">{notification.time}</p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
              <div className="p-4 border-t border-border">
                <Button variant="ghost" className="w-full text-sm">
                  View all notifications
                </Button>
              </div>
            </PopoverContent>
          </Popover>

          {/* Help */}
          <Button variant="ghost" size="sm" className="p-2">
            <HelpCircle className="w-4 h-4" />
          </Button>

          {/* User Menu */}
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" className="p-1">
                <Avatar className="w-8 h-8">
                  <AvatarImage src="/api/placeholder/32/32" />
                  <AvatarFallback>JD</AvatarFallback>
                </Avatar>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-56">
              <DropdownMenuLabel>
                <div>
                  <p className="font-medium">John Doe</p>
                  <p className="text-sm text-muted-foreground">john.doe@company.com</p>
                </div>
              </DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem>
                <User className="w-4 h-4 mr-2" />
                Profile
              </DropdownMenuItem>
              <DropdownMenuItem>
                <Settings className="w-4 h-4 mr-2" />
                Settings
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem>
                <LogOut className="w-4 h-4 mr-2" />
                Log out
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
    </header>
  );
}