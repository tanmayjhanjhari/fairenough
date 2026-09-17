import axios from 'axios'
import toast from 'react-hot-toast'

const getBaseURL = () => {
  if (import.meta.env.VITE_API_URL) return import.meta.env.VITE_API_URL;
  if (import.meta.env.VITE_API_BASE_URL) return import.meta.env.VITE_API_BASE_URL;
  // If running in browser on production (e.g. Vercel), route directly to Render backend
  if (typeof window !== 'undefined' && window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
    return 'https://fairenough-backend.onrender.com';
  }
  return 'http://localhost:8000';
};

const client = axios.create({
  baseURL: getBaseURL(),
  timeout: 120000,
  headers: {
    "Content-Type": "application/json",
  },
})

client.interceptors.request.use((config) => {
  // Get token from localStorage directly (avoid circular import with store)
  try {
    const authData = JSON.parse(localStorage.getItem('fairenough-auth') || localStorage.getItem('byus-auth') || '{}')
    const token = authData?.state?.token
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
  } catch (e) {}
  return config
})

client.interceptors.response.use(
  (response) => response,
  (error) => {
    const msg = error?.response?.data?.detail || error?.response?.data?.message || error?.message || 'Something went wrong.'
    if (error?.response?.status === 401) {
      // Clear auth and redirect to login only for protected routes
      const isProtected = error.config.url.includes('/api/reports/')
                       || error.config.url.includes('/api/auth/me')
      if (isProtected) {
        localStorage.removeItem('fairenough-auth')
        localStorage.removeItem('byus-auth')
        toast.error('Session expired. Please log in again.', {
          duration: 5000,
          style: {
            background: "#1E293B",
            color: "#F1F5F9",
            border: "1px solid rgba(239,68,68,0.4)",
          },
          iconTheme: { primary: "#EF4444", secondary: "#F1F5F9" },
        })
      }
    } else {
      toast.error(msg, {
        duration: 5000,
        style: {
          background: "#1E293B",
          color: "#F1F5F9",
          border: "1px solid rgba(239,68,68,0.4)",
        },
        iconTheme: { primary: "#EF4444", secondary: "#F1F5F9" },
      })
    }
    return Promise.reject(error)
  }
)

export default client
