import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
import { getAuth, GoogleAuthProvider } from 'firebase/auth';
import { getFirestore } from 'firebase/firestore';
import { getDatabase } from 'firebase/database';

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyCNKRHAHaWS806hqnxlPZ2ZGyqfuTjr7qk",
  authDomain: "genai-973e8.firebaseapp.com",
  databaseURL: "https://genai-973e8-default-rtdb.asia-southeast1.firebasedatabase.app",
  projectId: "genai-973e8",
  storageBucket: "genai-973e8.firebasestorage.app",
  messagingSenderId: "775058576088",
  appId: "1:775058576088:web:67e1008f8134f479c3fdad",
  measurementId: "G-X2DX6W8R1S"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
export const analytics = getAnalytics(app);
export const auth = getAuth(app);
export const db = getFirestore(app);
export const realtimeDb = getDatabase(app);
export const googleAuthProvider = new GoogleAuthProvider();

// Export a sanitized debug helper to inspect the config without revealing sensitive keys in logs
export function firebaseConfigDebug() {
  return {
    apiKey: firebaseConfig.apiKey ? 'SET' : 'MISSING',
    authDomain: firebaseConfig.authDomain ? firebaseConfig.authDomain : 'MISSING',
    projectId: firebaseConfig.projectId ? firebaseConfig.projectId : 'MISSING',
    databaseURL: firebaseConfig.databaseURL ? firebaseConfig.databaseURL : 'MISSING',
    appId: firebaseConfig.appId ? 'SET' : 'MISSING',
  };
}
