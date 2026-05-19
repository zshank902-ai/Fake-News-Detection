import React, { useState, useEffect } from 'react';
import { 
  Shield, 
  Activity, 
  Globe, 
  Database,
  Award,
  ChevronRight,
  Zap,
  TrendingUp,
  AlertOctagon,
  ShieldCheck,
  Wifi,
  WifiOff
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';

const Dashboard = () => {
  const [time, setTime] = useState(new Date().toLocaleTimeString());
  const [isConnected, setIsConnected] = useState(false);
  const [stats, setStats] = useState({
    scanned: "0",
    fake: "0",
    reliability: "99.2%",
    history: []
  });

  // --- REAL-TIME SYNC ENGINE ---
  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await axios.get('http://localhost:8000/stats');
        setStats(res.data);
        setIsConnected(true);
      } catch (err) {
        setIsConnected(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 3000); // Sync every 3s
    const timer = setInterval(() => setTime(new Date().toLocaleTimeString()), 1000);

    return () => {
      clearInterval(interval);
      clearInterval(timer);
    };
  }, []);

  return (
    <div className="wrapper">
      <div className="blob blob-1" />
      <div className="blob blob-2" />

      {/* --- HEADER --- */}
      <div className="header">
        <div className="flex items-center gap-5">
          <div className="w-14 h-14 border-2 border-[#00ffb4] rounded-xl flex items-center justify-center shadow-[0_0_24px_rgba(0,255,180,0.3)]">
            <Shield size={28} className="text-[#00ffb4]" />
          </div>
          <div className="header-title">
            <h1 className="glitch-text">TRUTH SHIELD</h1>
            <p>Neural Defense v3.0 &nbsp;·&nbsp; SOTA CONNECTED</p>
          </div>
        </div>
        <div className="flex items-center gap-6">
          <div className={`status-pill ${isConnected ? 'text-[#00ffb4]' : 'text-rose-500 border-rose-500/20 bg-rose-500/5'}`}>
            <div className={`pulse-dot ${!isConnected && 'bg-rose-500 shadow-[0_0_8px_#f43f5e]'}`} />
            {isConnected ? 'API CONNECTED' : 'SYSTEM OFFLINE'}
          </div>
          <div className="font-mono text-sm text-[#5a6a7a] tracking-widest">{time}</div>
        </div>
      </div>

      {/* --- STATS ROW (REAL-TIME) --- */}
      <div className="stats-row">
        <div className="stat-card">
          <div className="stat-label"><span />Scan Volume</div>
          <div className="stat-value">{stats.scanned}</div>
          <div className="flex items-center gap-1 text-[0.78rem] text-[#00ffb4] mt-2 font-bold">
            <TrendingUp size={12} /> Live Engine Data
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-label"><span />Shield Rank</div>
          <div className="stat-value">#042</div>
          <div className="flex items-center gap-1 text-[0.78rem] text-[#00ffb4] mt-2 font-bold">
            <TrendingUp size={12} /> Top 1% Global
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-label"><span />Accuracy</div>
          <div className="stat-value">{stats.reliability}</div>
          <div className="mt-4 flex gap-2">
            <span className="tag">BERT</span><span className="tag">XLMR</span><span className="tag">SOTA</span>
          </div>
        </div>
      </div>

      {/* --- MAIN GRID --- */}
      <div className="main-grid">
        
        {/* LEFT: LIVE LOG */}
        <div className="log-card min-h-[500px]">
          <div className="flex justify-between items-center mb-6">
            <h3 className="font-syne text-base font-extrabold uppercase">Live Decision Log</h3>
            <div className="flex items-center gap-2 border border-[#00ffb4]/30 bg-[#00ffb4]/5 px-3 py-1 rounded-full text-[9px] font-mono text-[#00ffb4] tracking-widest">
              <div className="w-1.5 h-1.5 rounded-full bg-[#00ffb4] animate-pulse" />
              NEURAL STREAM
            </div>
          </div>
          
          <div className="w-full">
            <div className="grid grid-cols-12 gap-4 pb-3 border-b border-white/5 font-mono text-[9px] text-[#5a6a7a] uppercase tracking-widest mb-4">
              <div className="col-span-8">Evidence Analysis</div>
              <div className="col-span-2">Verdict</div>
              <div className="col-span-2 text-right">Confidence</div>
            </div>

            <div className="space-y-1">
              <AnimatePresence>
                {stats.history.length === 0 ? (
                  <div className="text-center py-20 text-[#5a6a7a] font-mono text-xs uppercase tracking-widest">
                    Waiting for Neural Input...
                  </div>
                ) : (
                  stats.history.map(entry => (
                    <motion.div 
                      key={entry.id}
                      initial={{ x: -10, opacity: 0 }}
                      animate={{ x: 0, opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="grid grid-cols-12 gap-4 py-4 px-2 border-b border-white/[0.04] hover:bg-white/[0.02] transition-all group items-center"
                    >
                      <div className="col-span-8">
                        <p className="text-sm font-medium text-slate-300 group-hover:text-white leading-relaxed">{entry.title}</p>
                        <p className="font-mono text-[9px] text-[#5a6a7a] mt-1 uppercase">{entry.source} • {entry.time} UTC</p>
                      </div>
                      <div className="col-span-2">
                        <span className={`px-2.5 py-1 rounded-md font-mono text-[8px] font-bold border ${entry.type === 'REAL' ? 'bg-[#00ffb4]/10 text-[#00ffb4] border-[#00ffb4]/20' : 'bg-[#ff4b6e]/10 text-[#ff4b6e] border-[#ff4b6e]/20'}`}>
                          {entry.type}
                        </span>
                      </div>
                      <div className="col-span-2 text-right font-mono text-xs text-white/70">
                        {entry.conf}%
                      </div>
                    </motion.div>
                  ))
                )}
              </AnimatePresence>
            </div>
          </div>
        </div>

        {/* RIGHT: SYSTEM INFO */}
        <div className="flex flex-col gap-4">
          <div className="stat-card !p-7">
            <div className="flex justify-between items-center mb-6">
              <h3 className="font-syne text-sm font-bold uppercase">Neural Link</h3>
              <div className="status-pill !p-1.5"><div className={`pulse-dot ${!isConnected && 'bg-rose-500'}`} /></div>
            </div>
            <p className="text-xs text-[#5a6a7a] leading-relaxed mb-6 font-medium">
              API is {isConnected ? 'LIVE' : 'OFFLINE'}. {isConnected ? 'Actively syncing with gated fusion engine.' : 'Please start api/main.py to begin monitoring.'}
            </p>
            <div className="w-full h-32 bg-[#0a1018] rounded-xl border border-white/5 flex items-center justify-center relative overflow-hidden">
               <div className="absolute inset-0 opacity-10 bg-[url('https://www.transparenttextures.com/patterns/carbon-fibre.png')]" />
               <motion.div animate={{ rotate: 360 }} transition={{ duration: 10, repeat: Infinity, ease: "linear" }} className="w-[120%] h-[1px] bg-[#00ffb4]/30 absolute" />
               <Activity size={32} className={`text-[#00ffb4] opacity-40 ${!isConnected && 'text-rose-500'}`} />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="stat-card !p-5 flex flex-col items-center">
              <Globe size={18} className="text-[#00ffb4] mb-2" />
              <p className="font-mono text-[8px] text-[#5a6a7a] uppercase mb-1">Coverage</p>
              <p className="font-syne text-xs font-black">GLOBAL</p>
            </div>
            <div className="stat-card !p-5 flex flex-col items-center">
              <Database size={18} className="text-[#00c8ff] mb-2" />
              <p className="font-mono text-[8px] text-[#5a6a7a] uppercase mb-1">Engine</p>
              <p className="font-syne text-xs font-black">v3.0.4</p>
            </div>
          </div>
          
          <button className="w-full py-5 bg-[#00ffb4]/10 border border-[#00ffb4]/20 rounded-xl text-[#00ffb4] font-black font-syne text-[10px] tracking-[0.2em] uppercase hover:bg-[#00ffb4] hover:text-[#020408] transition-all">
            MANUAL SYSTEM SCAN
          </button>
        </div>

      </div>
    </div>
  );
};

export default Dashboard;
