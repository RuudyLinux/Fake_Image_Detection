import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import BlindDetection from './pages/BlindDetection';
import ComparativeDetection from './pages/ComparativeDetection';
import History from './pages/History';

export default function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/blind" element={<BlindDetection />} />
        <Route path="/compare" element={<ComparativeDetection />} />
        <Route path="/history" element={<History />} />
      </Routes>
    </BrowserRouter>
  );
}
