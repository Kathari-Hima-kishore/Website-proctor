import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';
import { firebaseConfigDebug } from './firebase';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

// Expose a sanitized firebase config helper for debugging in the browser console.
// Call `firebaseConfigDebug()` in console to see which keys are set vs missing.
if (typeof window !== 'undefined') {
  // eslint-disable-next-line no-console
  console.info('[app] sanitized firebase config:', firebaseConfigDebug());
  // Make callable from console
  // eslint-disable-next-line no-undef
  window.firebaseConfigDebug = firebaseConfigDebug;
}

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
