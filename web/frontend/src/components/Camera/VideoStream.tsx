import React, { useRef, useEffect, useCallback, useState } from 'react';
import { Box, Alert } from '@mui/material';
import { useSocket } from '../../contexts/SocketContext';

interface Face {
  box: [number, number, number, number];
  label: string;
  confidence: number;
  threshold_met: boolean;
}

interface VideoStreamProps {
  mode: 'capture' | 'recognition';
  onCapture?: (imageData: string) => void;
  isActive?: boolean;
}

const VideoStream: React.FC<VideoStreamProps> = ({ mode, onCapture, isActive = true }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animationRef = useRef<number>(0);
  const { socket } = useSocket();
  const [faces, setFaces] = useState<Face[]>([]);
  const [error, setError] = useState<string>('');

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user',
        },
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
      }
    } catch (err) {
      setError('Failed to access camera. Please ensure camera permissions are granted.');
      console.error('Camera error:', err);
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
  };

  const drawFaceBoxes = (ctx: CanvasRenderingContext2D, faces: Face[], scale: number) => {
    faces.forEach(face => {
      const [x1, y1, x2, y2] = face.box;
      const width = x2 - x1;
      const height = y2 - y1;

      // Scale coordinates
      const sx1 = x1 * scale;
      const sy1 = y1 * scale;
      const swidth = width * scale;
      const sheight = height * scale;

      // Draw box
      ctx.strokeStyle = face.threshold_met ? '#4caf50' : '#f44336';
      ctx.lineWidth = 3;
      ctx.strokeRect(sx1, sy1, swidth, sheight);

      // Draw label
      if (face.label !== 'no_model') {
        ctx.fillStyle = face.threshold_met ? '#4caf50' : '#f44336';
        ctx.fillRect(sx1, sy1 - 25, swidth, 25);
        
        ctx.fillStyle = 'white';
        ctx.font = '16px Arial';
        ctx.fillText(
          `${face.label}: ${(face.confidence * 100).toFixed(1)}%`,
          sx1 + 5,
          sy1 - 7
        );
      }
    });
  };

  const captureFrame = useCallback(() => {
    if (!videoRef.current || !canvasRef.current || !isActive) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    if (!ctx || video.readyState !== 4) return;

    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw video frame
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Get image data
    const imageData = canvas.toDataURL('image/jpeg', 0.8);

    if (mode === 'recognition' && socket?.connected) {
      // Send frame to server for recognition
      socket.emit('video_frame', { image: imageData });
    } else if (mode === 'capture' && onCapture) {
      // Draw face boxes for capture mode
      const scale = canvas.width / 640; // Assuming server processes at 640px width
      drawFaceBoxes(ctx, faces, scale);
    }

    // Continue animation
    animationRef.current = requestAnimationFrame(captureFrame);
  }, [mode, socket, isActive, faces, onCapture]);

  useEffect(() => {
    startCamera();

    return () => {
      stopCamera();
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (isActive) {
      animationRef.current = requestAnimationFrame(captureFrame);
    } else {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isActive, captureFrame]);

  useEffect(() => {
    if (!socket) return;

    const handleDetectionResult = (data: any) => {
      setFaces(data.faces || []);
      
      // Draw faces on canvas
      if (canvasRef.current && videoRef.current) {
        const ctx = canvasRef.current.getContext('2d');
        const video = videoRef.current;
        
        if (ctx && video.readyState === 4) {
          const scale = canvasRef.current.width / 640;
          ctx.drawImage(video, 0, 0, canvasRef.current.width, canvasRef.current.height);
          drawFaceBoxes(ctx, data.faces || [], scale);
        }
      }
    };

    socket.on('detection_result', handleDetectionResult);

    return () => {
      socket.off('detection_result', handleDetectionResult);
    };
  }, [socket]);

  const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    if (mode !== 'capture' || !onCapture || !canvasRef.current) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const scale = canvasRef.current.width / rect.width;

    // Check if click is on a face
    const clickedFace = faces.find(face => {
      const [x1, y1, x2, y2] = face.box;
      const sx1 = x1 * scale;
      const sy1 = y1 * scale;
      const sx2 = x2 * scale;
      const sy2 = y2 * scale;
      
      return x >= sx1 && x <= sx2 && y >= sy1 && y <= sy2;
    });

    if (clickedFace && canvasRef.current) {
      const imageData = canvasRef.current.toDataURL('image/jpeg', 0.8);
      onCapture(imageData);
    }
  };

  return (
    <Box sx={{ position: 'relative', width: '100%', maxWidth: 800 }}>
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={{
          display: 'none',
        }}
      />
      <canvas
        ref={canvasRef}
        onClick={handleCanvasClick}
        style={{
          width: '100%',
          height: 'auto',
          cursor: mode === 'capture' ? 'pointer' : 'default',
          backgroundColor: '#000',
          borderRadius: 8,
        }}
      />
    </Box>
  );
};

export default VideoStream;
