import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { createUserWithEmailAndPassword, signInWithPopup, updateProfile, sendEmailVerification } from 'firebase/auth';
import { doc, setDoc } from 'firebase/firestore';
import { auth, db, googleAuthProvider } from '../firebase';
import { TextField, Button, Box, Typography, Container, Alert, Snackbar } from '@mui/material';
import GoogleIcon from '@mui/icons-material/Google';


const Signup = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [username, setUsername] = useState('');
  const [error, setError] = useState('');
  const [openSnackbar, setOpenSnackbar] = useState(false);
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' });
  const navigate = useNavigate();

  const handleSignup = async (e) => {
    e.preventDefault();
    setError('');

    try {
      const userCredential = await createUserWithEmailAndPassword(auth, email, password);
      await updateProfile(userCredential.user, { displayName: username });
      await sendEmailVerification(userCredential.user);
      await setDoc(doc(db, 'users', userCredential.user.uid), {
        uid: userCredential.user.uid,
        email: userCredential.user.email,
        username: username,
      });
      setSnackbar({ open: true, message: 'Verification email sent! Please check your inbox.', severity: 'success' });
      navigate('/login');
    } catch (err) {
      setError(err.message);
      setOpenSnackbar(true);
    }
  };

  const handleGoogleSignup = async () => {
    setError('');
    try {
      const result = await signInWithPopup(auth, googleAuthProvider);
      const user = result.user;
      if (!user.displayName) {
        await updateProfile(user, { displayName: username || 'Google User' });
      }
      await setDoc(doc(db, 'users', user.uid), {
        uid: user.uid,
        email: user.email,
        username: user.displayName,
      }, { merge: true });
      navigate('/dashboard');
    } catch (err) {
      setError(err.message);
      setOpenSnackbar(true);
    }
  };

  const handleCloseSnackbar = () => {
    setOpenSnackbar(false);
  };

  const handleCloseSuccessSnackbar = () => {
    setSnackbar({ ...snackbar, open: false });
  };



  return (
    <Container component="main" maxWidth="xs" sx={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <Box
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
        <Typography component="h1" variant="h4" className="signup-title" sx={{ fontWeight: 800, mb: 2, color: '#3f5efb', letterSpacing: 2 }}>
          SIGN UP
        </Typography>
        <Typography variant="subtitle1" sx={{ mb: 3, color: '#26d0ce', fontWeight: 600 }}>
          Create your account
        </Typography>
        <Box component="form" onSubmit={handleSignup} noValidate className="signup-form" sx={{ width: '100%' }}>
          <TextField
            margin="normal"
            required
            fullWidth
            id="username"
            name="username"
            autoComplete="name"
            autoFocus
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="Username"
            sx={{ mb: 2, input: { color: '#fff', fontWeight: 600, letterSpacing: 1, '::placeholder': { color: '#3f5efb', opacity: 1 } }, fieldset: { borderColor: '#3f5efb' } }}
          />
          <TextField
            margin="normal"
            required
            fullWidth
            id="email"
            name="email"
            autoComplete="email"
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
            Sign Up
          </Button>
          <Button
            fullWidth
            variant="outlined"
            onClick={handleGoogleSignup}
            sx={{ mb: 2, fontWeight: 700, fontSize: 16, borderRadius: 2, color: '#26d0ce', borderColor: '#26d0ce', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 1 }}
            startIcon={<GoogleIcon />}
          >
            Continue with Google
          </Button>
          {error && (
            <Alert severity="error" sx={{ mt: 2, background: 'rgba(30,0,40,0.9)', color: '#3f5efb', border: '1px solid #3f5efb' }}>
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
        onClose={handleCloseSuccessSnackbar}
      >
        <Alert onClose={handleCloseSuccessSnackbar} severity={snackbar.severity} sx={{ background: 'rgba(30,0,40,0.9)', color: '#26d0ce', border: '1px solid #26d0ce' }}>
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Container>
  );
};

export default Signup;
