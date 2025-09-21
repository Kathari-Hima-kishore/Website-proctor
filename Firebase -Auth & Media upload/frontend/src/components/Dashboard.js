import React, { useState, useEffect } from 'react';
import Compressor from 'compressorjs';
import { useNavigate } from 'react-router-dom';
import { onAuthStateChanged, signOut } from 'firebase/auth';
import { ref, set, onValue, remove } from 'firebase/database';
import { auth, realtimeDb } from '../firebase';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Container,
  Grid,
  Card,
  CardMedia,
  CardActions,
  IconButton,
  Box,
  CircularProgress,
  Snackbar,
  Alert,
} from '@mui/material';
import DeleteIcon from '@mui/icons-material/Delete';
import UploadIcon from '@mui/icons-material/Upload';

const Dashboard = () => {
  const [user, setUser] = useState(null);
  const [images, setImages] = useState([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'info' });
  const navigate = useNavigate();

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      if (currentUser) {
        setUser(currentUser);
        const imagesRef = ref(realtimeDb, `users/${currentUser.uid}/images`);
        onValue(imagesRef, (snapshot) => {
          const data = snapshot.val();
          if (data) {
            setImages(Object.values(data));
          } else {
            setImages([]);
          }
          setLoading(false);
        });
      } else {
        navigate('/login');
      }
    });
    return unsubscribe;
  }, [navigate]);

  const handleLogout = async () => {
    try {
      await signOut(auth);
      navigate('/login');
    } catch (err) {
      setSnackbar({ open: true, message: 'Logout failed', severity: 'error' });
    }
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      setSnackbar({ open: true, message: 'Please select an image file', severity: 'error' });
      return;
    }

    setUploading(true);
    new Compressor(file, {
      quality: 0.6,
      maxWidth: 800,
      maxHeight: 800,
      success(compressedFile) {
        const reader = new FileReader();
        reader.onload = async (event) => {
          const base64 = event.target.result;
          const id = Date.now().toString();
          try {
            await set(ref(realtimeDb, `users/${user.uid}/images/${id}`), {
              id,
              name: compressedFile.name,
              base64,
              timestamp: new Date().toISOString(),
            });
            setSnackbar({ open: true, message: 'Image uploaded successfully', severity: 'success' });
            e.target.value = ''; // Reset input
          } catch (err) {
            setSnackbar({ open: true, message: 'Upload failed', severity: 'error' });
          }
          setUploading(false);
        };
        reader.readAsDataURL(compressedFile);
      },
      error(err) {
        setSnackbar({ open: true, message: 'Compression failed', severity: 'error' });
        setUploading(false);
      },
    });
  };

  const handleDelete = async (imageId) => {
    try {
      await remove(ref(realtimeDb, `users/${user.uid}/images/${imageId}`));
      setSnackbar({ open: true, message: 'Image deleted successfully', severity: 'success' });
    } catch (err) {
      setSnackbar({ open: true, message: 'Delete failed', severity: 'error' });
    }
  };

  const handleCloseSnackbar = () => {
    setSnackbar({ ...snackbar, open: false });
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="100vh">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <>
      <AppBar position="static">
        <Toolbar>
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Dashboard
          </Typography>
          <Button color="inherit" onClick={handleLogout}>
            Logout
          </Button>
        </Toolbar>
      </AppBar>
      <Container sx={{ mt: 4 }}>
        <Box sx={{ mb: 4 }}>
          <input
            accept="image/*"
            style={{ display: 'none' }}
            id="upload-image"
            type="file"
            onChange={handleFileUpload}
          />
          <label htmlFor="upload-image">
            <Button
              variant="contained"
              component="span"
              startIcon={<UploadIcon />}
              disabled={uploading}
            >
              {uploading ? <CircularProgress size={20} /> : 'Upload Image'}
            </Button>
          </label>
        </Box>
        <Grid container spacing={3}>
          {images.map((image) => (
            <Grid item xs={12} sm={6} md={4} key={image.id}>
              <Card
                sx={{
                  transition: 'transform 0.2s',
                  '&:hover': {
                    transform: 'scale(1.05)',
                  },
                }}
              >
                <CardMedia
                  component="img"
                  height="200"
                  image={image.base64}
                  alt={image.name}
                />
                <CardActions>
                  <IconButton
                    size="small"
                    color="error"
                    onClick={() => handleDelete(image.id)}
                  >
                    <DeleteIcon />
                  </IconButton>
                </CardActions>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Container>
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={handleCloseSnackbar}
      >
        <Alert onClose={handleCloseSnackbar} severity={snackbar.severity}>
          {snackbar.message}
        </Alert>
      </Snackbar>
    </>
  );
};

export default Dashboard;
