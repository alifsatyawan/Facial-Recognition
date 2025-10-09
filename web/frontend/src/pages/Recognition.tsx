import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Switch,
  FormControlLabel,
  Slider,
  Button,
  Chip,
  Card,
  CardContent,
  Stack,
} from '@mui/material';
import { PlayArrow, Stop, Settings } from '@mui/icons-material';
import VideoStream from '../components/Camera/VideoStream';
import { useSocket } from '../contexts/SocketContext';
import axios from 'axios';

const Recognition: React.FC = () => {
  const { socket, fps } = useSocket();
  const [isRunning, setIsRunning] = useState(false);
  const [settings, setSettings] = useState({
    use_alignment: true,
    threshold: 0.6,
    show_fps: true,
  });

  useEffect(() => {
    // Load current settings
    fetchSettings();

    return () => {
      if (isRunning && socket) {
        socket.emit('stop_recognition');
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const fetchSettings = async () => {
    try {
      const response = await axios.get('http://localhost:5000/api/status');
      setSettings(response.data.settings);
    } catch (error) {
      console.error('Failed to fetch settings:', error);
    }
  };

  const updateSettings = async (newSettings: any) => {
    try {
      await axios.post('http://localhost:5000/api/settings', newSettings);
      setSettings({ ...settings, ...newSettings });
    } catch (error) {
      console.error('Failed to update settings:', error);
    }
  };

  const handleStart = () => {
    if (socket) {
      socket.emit('start_recognition');
      setIsRunning(true);
    }
  };

  const handleStop = () => {
    if (socket) {
      socket.emit('stop_recognition');
      setIsRunning(false);
    }
  };

  const handleAlignmentChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    updateSettings({ use_alignment: event.target.checked });
  };

  const handleThresholdChange = (_: any, value: number | number[]) => {
    updateSettings({ threshold: value as number });
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom sx={{ mb: 4 }}>
        Real-time Recognition
      </Typography>

      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '2fr 1fr' }, gap: 3 }}>
        <Paper elevation={3} sx={{ p: 2 }}>
          <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography variant="h6">Camera Feed</Typography>
            <Box>
              {settings.show_fps && fps > 0 && (
                <Chip
                  label={`FPS: ${fps.toFixed(1)}`}
                  color="success"
                  size="small"
                  sx={{ mr: 2 }}
                />
              )}
              <Button
                variant="contained"
                color={isRunning ? 'error' : 'primary'}
                startIcon={isRunning ? <Stop /> : <PlayArrow />}
                onClick={isRunning ? handleStop : handleStart}
              >
                {isRunning ? 'Stop' : 'Start'} Recognition
              </Button>
            </Box>
          </Box>
          <VideoStream mode="recognition" isActive={isRunning} />
        </Paper>

        <Stack spacing={2}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Settings sx={{ mr: 1 }} />
                <Typography variant="h6">Settings</Typography>
              </Box>

              <Box sx={{ mb: 3 }}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.use_alignment}
                      onChange={handleAlignmentChange}
                      disabled={isRunning}
                    />
                  }
                  label="Face Alignment"
                />
                <Typography variant="caption" display="block" color="text.secondary">
                  Enable dlib face alignment for better accuracy (reduces FPS)
                </Typography>
              </Box>

              <Box sx={{ mb: 3 }}>
                <Typography gutterBottom>
                  Recognition Threshold: {(settings.threshold * 100).toFixed(0)}%
                </Typography>
                <Slider
                  value={settings.threshold}
                  onChange={handleThresholdChange}
                  min={0}
                  max={1}
                  step={0.05}
                  marks={[
                    { value: 0, label: '0%' },
                    { value: 0.5, label: '50%' },
                    { value: 1, label: '100%' },
                  ]}
                  disabled={isRunning}
                />
                <Typography variant="caption" display="block" color="text.secondary">
                  Higher threshold = stricter matching
                </Typography>
              </Box>

              <Box sx={{ mb: 3 }}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={settings.show_fps}
                      onChange={(e) => updateSettings({ show_fps: e.target.checked })}
                    />
                  }
                  label="Show FPS"
                />
              </Box>
            </CardContent>
          </Card>

          <Card sx={{ mt: 2 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Performance Tips
              </Typography>
              <Typography variant="body2" paragraph>
                • Turn off alignment for +5 FPS
              </Typography>
              <Typography variant="body2" paragraph>
                • Lower threshold for easier matching
              </Typography>
              <Typography variant="body2" paragraph>
                • Good lighting improves accuracy
              </Typography>
              <Typography variant="body2">
                • Stay 2-3 feet from camera
              </Typography>
            </CardContent>
          </Card>
        </Stack>
      </Box>
    </Box>
  );
};

export default Recognition;
