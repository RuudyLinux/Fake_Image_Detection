import { BrowserRouter, Routes, Route, useLocation } from 'react-router-dom';
import AnimatedBackground from './components/AnimatedBackground';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import BlindDetection from './pages/BlindDetection';
import ComparativeDetection from './pages/ComparativeDetection';
import History from './pages/History';

/**
 * Keying the route subtree on pathname remounts it on navigation, which
 * restarts the entrance animation — without it, React reuses the DOM and the
 * new page snaps in with no transition.
 */
function AnimatedRoutes() {
  const location = useLocation();
  return (
    <div key={location.pathname} className="animate-fade-in">
      <Routes location={location}>
        <Route path="/" element={<Home />} />
        <Route path="/blind" element={<BlindDetection />} />
        <Route path="/compare" element={<ComparativeDetection />} />
        <Route path="/history" element={<History />} />
      </Routes>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AnimatedBackground />
      <Navbar />
      <AnimatedRoutes />
    </BrowserRouter>
  );
}
