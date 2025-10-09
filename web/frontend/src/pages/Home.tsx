import React, { useEffect, useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  CardActionArea,
  Avatar,
  Paper,
  Stack,
} from '@mui/material';
import {
  CameraAlt,
  School,
  Visibility,
  Collections,
  People,
  CheckCircle,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

interface Stats {
  persons: Array<{ name: string; count: number }>;
  totalPersons: number;
  totalImages: number;
}

const Home: React.FC = () => {
  const navigate = useNavigate();
  const [stats, setStats] = useState<Stats>({
    persons: [],
    totalPersons: 0,
    totalImages: 0,
  });

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const response = await axios.get('http://localhost:5000/api/persons');
      const persons = response.data.persons;
      const totalImages = persons.reduce((sum: number, p: any) => sum + p.count, 0);
      
      setStats({
        persons,
        totalPersons: persons.length,
        totalImages,
      });
    } catch (error) {
      console.error('Failed to fetch stats:', error);
    }
  };

  const features = [
    {
      title: 'Capture Faces',
      description: 'Capture face images using your webcam',
      icon: <CameraAlt sx={{ fontSize: 48 }} />,
      path: '/capture',
      color: '#2196f3',
    },
    {
      title: 'Train Model',
      description: 'Build recognition model from captured faces',
      icon: <School sx={{ fontSize: 48 }} />,
      path: '/training',
      color: '#4caf50',
    },
    {
      title: 'Recognition',
      description: 'Real-time face recognition with live camera',
      icon: <Visibility sx={{ fontSize: 48 }} />,
      path: '/recognition',
      color: '#ff9800',
    },
    {
      title: 'Gallery',
      description: 'View and manage captured face images',
      icon: <Collections sx={{ fontSize: 48 }} />,
      path: '/gallery',
      color: '#e91e63',
    },
  ];

  return (
    <Box>
      <Typography variant="h4" gutterBottom sx={{ mb: 4 }}>
        Welcome to Facial Recognition System
      </Typography>

      <Stack direction={{ xs: 'column', md: 'row' }} spacing={3} sx={{ mb: 4 }}>
        <Paper
          elevation={3}
          sx={{
            p: 3,
            textAlign: 'center',
            background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            flex: 1,
          }}
        >
          <People sx={{ fontSize: 48, mb: 1 }} />
          <Typography variant="h3">{stats.totalPersons}</Typography>
          <Typography variant="subtitle1">Registered People</Typography>
        </Paper>
        <Paper
          elevation={3}
          sx={{
            p: 3,
            textAlign: 'center',
            background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
            flex: 1,
          }}
        >
          <Collections sx={{ fontSize: 48, mb: 1 }} />
          <Typography variant="h3">{stats.totalImages}</Typography>
          <Typography variant="subtitle1">Total Images</Typography>
        </Paper>
        <Paper
          elevation={3}
          sx={{
            p: 3,
            textAlign: 'center',
            background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
            flex: 1,
          }}
        >
          <CheckCircle sx={{ fontSize: 48, mb: 1 }} />
          <Typography variant="h3">Ready</Typography>
          <Typography variant="subtitle1">System Status</Typography>
        </Paper>
      </Stack>

      <Typography variant="h5" gutterBottom sx={{ mb: 3 }}>
        Quick Actions
      </Typography>

      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: 'repeat(2, 1fr)', md: 'repeat(4, 1fr)' }, gap: 3 }}>
        {features.map((feature) => (
          <Card
            key={feature.title}
            sx={{
              height: '100%',
              transition: 'transform 0.2s',
              '&:hover': {
                transform: 'translateY(-4px)',
              },
            }}
          >
            <CardActionArea
              onClick={() => navigate(feature.path)}
              sx={{ height: '100%' }}
            >
              <CardContent
                sx={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  textAlign: 'center',
                  p: 3,
                }}
              >
                <Avatar
                  sx={{
                    width: 80,
                    height: 80,
                    bgcolor: feature.color,
                    mb: 2,
                  }}
                >
                  {feature.icon}
                </Avatar>
                <Typography variant="h6" gutterBottom>
                  {feature.title}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {feature.description}
                </Typography>
              </CardContent>
            </CardActionArea>
          </Card>
        ))}
      </Box>
    </Box>
  );
};

export default Home;
