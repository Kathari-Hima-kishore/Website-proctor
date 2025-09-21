import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { signInWithEmailAndPassword, signInWithPopup, sendPasswordResetEmail } from 'firebase/auth';
import { auth, googleAuthProvider } from '../firebase';
import { TextField, Button, Box, Typography, Container, Alert, Snackbar } from '@mui/material';
import GoogleIcon from '@mui/icons-material/Google';


const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [openSnackbar, setOpenSnackbar] = useState(false);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'info' });
  const navigate = useNavigate();
  const formRef = useRef(null);

  const handleLogin = async (e) => {
    e.preventDefault();
    setError('');

    // Basic client-side validation to avoid sending empty values to Firebase REST API.
    if (!email || !password) {
      setError('Please enter both email and password');
      setOpenSnackbar(true);
      return;
    }

    try {
      await signInWithEmailAndPassword(auth, email, password);
      navigate('/dashboard');
    } catch (err) {
      // Log full error details to console for debugging; UI shows friendly message.
      // eslint-disable-next-line no-console
      console.error('[login] signInWithEmailAndPassword error:', err);
      // eslint-disable-next-line no-console
      console.error('[login] firebase error code/message:', err.code, err.message);
      // Map common firebase auth error codes to friendlier messages.
      const friendly = (() => {
        if (!err || !err.code) return err.message || 'Login failed';
        switch (err.code) {
          case 'auth/user-not-found':
          case 'EMAIL_NOT_FOUND':
            return 'No account found with that email. Please sign up first.';
          case 'auth/wrong-password':
          case 'INVALID_PASSWORD':
            return 'Incorrect password. Please try again.';
          case 'auth/user-disabled':
          case 'USER_DISABLED':
            return 'This user account has been disabled.';
          case 'auth/too-many-requests':
          case 'TOO_MANY_ATTEMPTS_TRY_LATER':
            return 'Too many attempts. Please try again later.';
          case 'auth/operation-not-allowed':
          case 'OPERATION_NOT_ALLOWED':
            return 'Email/password sign-in is not enabled for this project.';
          default:
            return err.message || 'Login failed';
        }
      })();
      setError(friendly);
      setOpenSnackbar(true);
    }
  };

  const handleGoogleLogin = async () => {
    setError('');
    try {
      await signInWithPopup(auth, googleAuthProvider);
      navigate('/dashboard');
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error('[login] signInWithPopup error:', err);
      // eslint-disable-next-line no-console
      console.error('[login] firebase error code/message:', err.code, err.message);
      setError(err.message || 'Google login failed');
      setOpenSnackbar(true);
    }
  };

  const handleCloseSnackbar = () => {
    setOpenSnackbar(false);
  };

  useEffect(() => {
    // Animate the background and form using Anime.js from CDN with royal blue palette
    if (window.anime) {
      const bg = document.getElementById('bg-animated');
      if (bg) {
        window.anime({
          targets: bg,
          background: [
            'linear-gradient(120deg, #00aaffff 0%, #414345 100%)',
            'linear-gradient(120deg, #283e51 0%, #485563 100%)',
            'linear-gradient(120deg, #1e3c72 0%, #040405ff 100%)',
            'linear-gradient(120deg, #0f2027 0%, #2c5364 100%)',
            'linear-gradient(120deg, #1a2980 0%, #26d0ce 100%)',
          ],
          duration: 18000,
          direction: 'alternate',
          loop: true,
          easing: 'linear',
          update: anim => {
            bg.style.background = anim.animations[0].currentValue;
          }
        });
      }
      window.anime({
        targets: '.login-form',
        translateY: [-60, 0],
        opacity: [0, 1],
        duration: 1000,
        easing: 'easeOutExpo'
      });
      window.anime({
        targets: '.login-title',
        scale: [0.8, 1],
        opacity: [0, 1],
        duration: 900,
        delay: 100,
        easing: 'easeOutExpo'
      });
    }
  }, []);

  return (
    <Container component="main" maxWidth="xs" sx={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <Box
        ref={formRef}
        sx={{
          width: '100%',
          background: 'rgba(24, 34, 64, 0.98)',
          borderRadius: 6,
          p: 4,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          position: 'relative',
          border: '2px solid #3f5efb',
        }}
      >
        <Typography component="h1" variant="h4" className="login-title" sx={{ fontWeight: 800, mb: 2, color: '#3f5efb', letterSpacing: 2, textShadow: '0 0 8px #1e3c72' }}>
          LOGIN
        </Typography>
        <Typography variant="subtitle1" sx={{ mb: 3, color: '#26d0ce', fontWeight: 600, textShadow: '0 0 6px #2a5298' }}>
          Enter the grid to continue
        </Typography>
        <Box component="form" onSubmit={handleLogin} noValidate className="login-form" sx={{ width: '100%' }}>
          <TextField
            margin="normal"
            required
            fullWidth
            id="email"
            name="email"
            autoComplete="email"
            autoFocus
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="Email"
            sx={{ mb: 2, input: { color: '#fff', fontWeight: 600, letterSpacing: 1, '::placeholder': { color: '#3f5efb', opacity: 1 } }, fieldset: { borderColor: '#3f5efb' } }}
          />
          <TextField
            margin="normal"
            required
            fullWidth
            name="password"
            type="password"
            id="password"
            autoComplete="current-password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Password"
            sx={{ mb: 2, input: { color: '#fff', fontWeight: 600, letterSpacing: 1, '::placeholder': { color: '#3f5efb', opacity: 1 } }, fieldset: { borderColor: '#3f5efb' } }}
          />
          <Button
            type="submit"
            fullWidth
            variant="contained"
            sx={{
              mt: 1,
              mb: 2,
              fontWeight: 700,
              fontSize: 18,
              py: 1.5,
              borderRadius: 2,
              background: 'linear-gradient(90deg, #1e3c72 0%, #2a5298 100%)',
              color: '#fff',
              letterSpacing: 2,
              transition: 'all 0.2s cubic-bezier(.4,2,.6,1)',
              '&:hover': {
                background: 'linear-gradient(90deg, #2a5298 0%, #1e3c72 100%)',
                color: '#fff',
                transform: 'scale(1.03)',
              },
            }}
          >
            Log In
          </Button>
          <Button
            fullWidth
            variant="outlined"
            onClick={handleGoogleLogin}
            sx={{ mb: 2, fontWeight: 700, fontSize: 16, borderRadius: 2, color: '#26d0ce', borderColor: '#26d0ce', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1, textShadow: '0 0 8px #26d0ce' }}
            startIcon={<GoogleIcon />}
          >
            Continue with Google
          </Button>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: 1 }}>
            <Button
              variant="text"
              onClick={() => {
                if (email) {
                  sendPasswordResetEmail(auth, email)
                    .then(() => {
                      setSnackbar({ open: true, message: 'Password reset email sent!', severity: 'success' });
                    })
                    .catch((err) => {
                      setError(err.message);
                      setOpenSnackbar(true);
                    });
                } else {
                  setError('Please enter your email first');
                  setOpenSnackbar(true);
                }
              }}
              sx={{ color: '#3f5efb', fontWeight: 600, textShadow: '0 0 8px #3f5efb' }}
            >
              Forgot Password?
            </Button>
            <Button
              variant="text"
              onClick={() => navigate('/signup')}
              sx={{ color: '#26d0ce', fontWeight: 600, textShadow: '0 0 8px #26d0ce' }}
            >
              Create an account
            </Button>
          </Box>
          {error && (
            <Alert severity="error" sx={{ mt: 2, background: 'rgba(30,0,40,0.9)', color: '#3f5efb', border: '1px solid #3f5efb', textShadow: '0 0 8px #3f5efb' }}>
              {error}
            </Alert>
          )}
        </Box>
      </Box>
      <Snackbar
        open={openSnackbar}
        autoHideDuration={6000}
        onClose={handleCloseSnackbar}
        message={error}
      />
      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
      >
        <Alert severity={snackbar.severity} sx={{ background: 'rgba(30,0,40,0.9)', color: '#26d0ce', border: '1px solid #26d0ce', textShadow: '0 0 8px #26d0ce' }}>
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Container>
  );
};

export default Login;
