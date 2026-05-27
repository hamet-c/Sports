import React from 'react';
import ReactDOM from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter, Link, NavLink, Route, Routes } from 'react-router-dom';
import SlatePage from './pages/SlatePage';
import PlayerDetailPage from './pages/PlayerDetailPage';
import PerformanceLogPage from './pages/PerformanceLogPage';
import PlayerSearch from './components/PlayerSearch';
import './index.css';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

function NavBar() {
  const linkCls = ({ isActive }: { isActive: boolean }) =>
    `px-3 py-1 rounded text-sm ${
      isActive ? 'bg-slate-900 text-white' : 'text-slate-700 hover:bg-slate-100'
    }`;
  return (
    <nav className="border-b border-slate-200 bg-white px-6 py-3 flex items-center gap-6">
      <Link to="/" className="font-bold text-lg">
        NBA Props
      </Link>
      <div className="flex gap-1">
        <NavLink to="/" end className={linkCls}>
          Slate
        </NavLink>
        <NavLink to="/performance" className={linkCls}>
          Performance
        </NavLink>
      </div>
      <div className="ml-auto">
        <PlayerSearch />
      </div>
    </nav>
  );
}

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-slate-50">
        <NavBar />
        <Routes>
          <Route path="/" element={<SlatePage />} />
          <Route path="/players/:playerId" element={<PlayerDetailPage />} />
          <Route path="/performance" element={<PerformanceLogPage />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </React.StrictMode>,
);
