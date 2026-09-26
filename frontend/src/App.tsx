import { useEffect, useState } from 'react'

interface HealthResponse {
  status: string
  app: string
  version: string
  database: string
  environment: string
  timestamp: string
}

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)
  const [lastChecked, setLastChecked] = useState<Date>(new Date())

  const fetchHealth = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch('/api/health')
      if (!res.ok) {
        throw new Error(`HTTP error ${res.status}: ${res.statusText}`)
      }
      const data = await res.json()
      setHealth(data)
      setLastChecked(new Date())
    } catch (err: any) {
      setError(err.message || 'Failed to connect to backend')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchHealth()
  }, [])

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col items-center justify-center p-6 selection:bg-cyan-500 selection:text-white">
      {/* Background glow effects */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 -left-40 w-96 h-96 bg-blue-600/20 rounded-full blur-3xl"></div>
        <div className="absolute top-1/2 -right-40 w-96 h-96 bg-emerald-600/15 rounded-full blur-3xl"></div>
      </div>

      <main className="relative z-10 w-full max-w-2xl bg-slate-900/80 border border-slate-800 backdrop-blur-xl rounded-2xl shadow-2xl p-8 transition-all">
        {/* Header */}
        <div className="flex items-center justify-between pb-6 border-b border-slate-800">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-semibold uppercase tracking-wider mb-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
              Phase 0 Active
            </div>
            <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-white flex items-center gap-3">
              <span>Sentinal</span>
              <span className="text-xs px-2.5 py-1 bg-slate-800 text-slate-400 border border-slate-700 rounded-md font-mono font-normal">
                v1.0.0
              </span>
            </h1>
            <p className="text-sm text-slate-400 mt-1">
              Intelligent Data Quality & EV Analytics Platform
            </p>
          </div>

          <button
            onClick={fetchHealth}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2 text-xs font-medium bg-slate-800 hover:bg-slate-700 active:scale-95 text-slate-200 border border-slate-700 rounded-xl transition-all cursor-pointer disabled:opacity-50"
            title="Refresh Health Status"
          >
            <svg
              className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
              />
            </svg>
            Refresh
          </button>
        </div>

        {/* Health Check Card */}
        <div className="mt-6 space-y-4">
          <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
            Backend & Services Health Check
          </div>

          {loading && !health && (
            <div className="p-8 text-center bg-slate-950/50 border border-slate-800 rounded-xl">
              <div className="w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin mx-auto mb-3"></div>
              <p className="text-sm text-slate-400">Pinging backend health endpoint (/api/health)...</p>
            </div>
          )}

          {error && (
            <div className="p-4 bg-red-950/40 border border-red-500/30 rounded-xl text-red-300 text-sm flex items-start gap-3">
              <span className="text-red-400 text-lg">⚠️</span>
              <div>
                <p className="font-semibold">Backend Unreachable</p>
                <p className="text-xs text-red-400/80 mt-1">{error}</p>
              </div>
            </div>
          )}

          {health && (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {/* API Status */}
              <div className="p-4 bg-slate-950/60 border border-slate-800/80 rounded-xl flex items-center justify-between">
                <div>
                  <div className="text-xs text-slate-400">FastAPI Backend</div>
                  <div className="text-sm font-semibold text-white capitalize mt-0.5">
                    {health.status}
                  </div>
                </div>
                <div
                  className={`w-3 h-3 rounded-full ${
                    health.status === 'healthy' ? 'bg-emerald-400 shadow-[0_0_8px_#34d399]' : 'bg-amber-400 shadow-[0_0_8px_#fbbf24]'
                  }`}
                ></div>
              </div>

              {/* PostgreSQL Status */}
              <div className="p-4 bg-slate-950/60 border border-slate-800/80 rounded-xl flex items-center justify-between">
                <div>
                  <div className="text-xs text-slate-400">PostgreSQL Database</div>
                  <div className="text-sm font-semibold text-white capitalize mt-0.5">
                    {health.database}
                  </div>
                </div>
                <div
                  className={`w-3 h-3 rounded-full ${
                    health.database === 'connected' ? 'bg-emerald-400 shadow-[0_0_8px_#34d399]' : 'bg-red-400 shadow-[0_0_8px_#f87171]'
                  }`}
                ></div>
              </div>

              {/* Environment */}
              <div className="p-4 bg-slate-950/60 border border-slate-800/80 rounded-xl">
                <div className="text-xs text-slate-400">Environment</div>
                <div className="text-sm font-mono text-cyan-300 mt-0.5">
                  {health.environment}
                </div>
              </div>

              {/* Timestamp */}
              <div className="p-4 bg-slate-950/60 border border-slate-800/80 rounded-xl">
                <div className="text-xs text-slate-400">Server Timestamp (UTC)</div>
                <div className="text-xs font-mono text-slate-300 mt-1 truncate" title={health.timestamp}>
                  {new Date(health.timestamp).toLocaleTimeString()} · {new Date(health.timestamp).toLocaleDateString()}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer info */}
        <div className="mt-8 pt-4 border-t border-slate-800/80 flex items-center justify-between text-xs text-slate-500">
          <span>Client proxy connected to FastAPI (:8000)</span>
          <span>Last ping: {lastChecked.toLocaleTimeString()}</span>
        </div>
      </main>
    </div>
  )
}
