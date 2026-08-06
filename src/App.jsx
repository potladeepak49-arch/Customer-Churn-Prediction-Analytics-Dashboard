import React, { useState, useEffect, useCallback } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { Search, ArrowUpDown, TrendingDown, DollarSign, AlertTriangle, Users, X } from "lucide-react";

const TIER_COLOR = {
  High: { bg: "#FCEBEB", text: "#791F1F", border: "#E24B4A" },
  Medium: { bg: "#FAEEDA", text: "#633806", border: "#EF9F27" },
  Low: { bg: "#EAF3DE", text: "#27500A", border: "#97C459" },
};

function currency(n) {
  if (n == null) return "-";
  return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 }).format(n);
}

function KpiCard({ icon: Icon, label, value, sub, accent }) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-4 flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-gray-500 uppercase tracking-wide">{label}</span>
        <Icon size={16} style={{ color: accent }} />
      </div>
      <div className="text-2xl font-semibold text-gray-900">{value ?? "…"}</div>
      {sub && <div className="text-xs text-gray-500">{sub}</div>}
    </div>
  );
}

function DetailRow({ label, value }) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-gray-100 last:border-0">
      <span className="text-xs text-gray-500">{label}</span>
      <span className="text-sm text-gray-800 font-medium">{value}</span>
    </div>
  );
}

function YesNo({ value }) {
  const yes = value === "Yes";
  return (
    <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${yes ? "bg-green-50 text-green-700" : "bg-gray-100 text-gray-500"}`}>
      {value}
    </span>
  );
}

function CustomerDrilldown({ customerId, onClose }) {
  const [customer, setCustomer] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!customerId) return;
    setCustomer(null);
    setError(null);
    fetch(`/api/customers/${customerId}`)
      .then(r => { if (!r.ok) throw new Error("Not found"); return r.json(); })
      .then(setCustomer)
      .catch(e => setError(e.message));
  }, [customerId]);

  if (!customerId) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4" style={{ background: "rgba(0,0,0,0.4)" }} onClick={onClose}>
      <div className="bg-white rounded-xl max-w-lg w-full max-h-[85vh] overflow-y-auto" onClick={e => e.stopPropagation()}>
        <div className="flex items-start justify-between p-4 border-b border-gray-100 sticky top-0 bg-white">
          <div>
            <p className="text-xs text-gray-500">Customer</p>
            <p className="text-lg font-semibold text-gray-900">{customerId}</p>
          </div>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600 p-1"><X size={18} /></button>
        </div>

        {error && <p className="p-4 text-sm text-red-600">Couldn't load this customer: {error}</p>}
        {!customer && !error && <p className="p-4 text-sm text-gray-500">Loading…</p>}

        {customer && (
          <div className="p-4 flex flex-col gap-4">
            <div className="flex items-center gap-3 rounded-lg p-3" style={{ background: TIER_COLOR[customer.risk_tier]?.bg }}>
              <div className="text-2xl font-semibold" style={{ color: TIER_COLOR[customer.risk_tier]?.text }}>
                {(customer.risk_score * 100).toFixed(0)}%
              </div>
              <div>
                <p className="text-sm font-medium" style={{ color: TIER_COLOR[customer.risk_tier]?.text }}>{customer.risk_tier} risk</p>
                <p className="text-xs" style={{ color: TIER_COLOR[customer.risk_tier]?.text }}>{customer.top_drivers}</p>
              </div>
            </div>

            <div>
              <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">Account</p>
              <DetailRow label="Contract" value={customer.contract_type} />
              <DetailRow label="Tenure" value={`${customer.tenure} months`} />
              <DetailRow label="Monthly charges" value={currency(customer.monthly_charges)} />
              <DetailRow label="Total charges to date" value={currency(customer.total_charges)} />
              <DetailRow label="Payment method" value={customer.payment_method} />
              <DetailRow label="Paperless billing" value={<YesNo value={customer.paperless_billing} />} />
            </div>

            <div>
              <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">Profile</p>
              <DetailRow label="Gender" value={customer.gender} />
              <DetailRow label="Senior citizen" value={<YesNo value={customer.senior_citizen === 1 ? "Yes" : "No"} />} />
              <DetailRow label="Has partner" value={<YesNo value={customer.partner} />} />
              <DetailRow label="Has dependents" value={<YesNo value={customer.dependents} />} />
            </div>

            <div>
              <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">Services</p>
              <DetailRow label="Internet service" value={customer.internet_service} />
              <DetailRow label="Phone service" value={<YesNo value={customer.phone_service} />} />
              <DetailRow label="Online security" value={<YesNo value={customer.online_security} />} />
              <DetailRow label="Tech support" value={<YesNo value={customer.tech_support} />} />
              <DetailRow label="Streaming TV" value={<YesNo value={customer.streaming_tv} />} />
              <DetailRow label="Streaming movies" value={<YesNo value={customer.streaming_movies} />} />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default function App() {
  const [kpis, setKpis] = useState(null);
  const [search, setSearch] = useState("");
  const [tierFilter, setTierFilter] = useState("All");
  const [sortKey, setSortKey] = useState("risk_score");
  const [sortDir, setSortDir] = useState("desc");
  const [page, setPage] = useState(0);
  const [rows, setRows] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [selectedId, setSelectedId] = useState(null);
  const pageSize = 10;

  useEffect(() => {
    fetch("/api/kpis").then(r => r.json()).then(setKpis).catch(() => {});
  }, []);

  const fetchCustomers = useCallback(() => {
    setLoading(true);
    const params = new URLSearchParams({
      sort_by: sortKey, sort_dir: sortDir, page: String(page), page_size: String(pageSize),
    });
    if (tierFilter !== "All") params.set("risk_tier", tierFilter);
    if (search) params.set("search", search);

    fetch(`/api/customers?${params}`)
      .then(r => r.json())
      .then(data => { setRows(data.results); setTotal(data.total); })
      .finally(() => setLoading(false));
  }, [search, tierFilter, sortKey, sortDir, page]);

  useEffect(() => {
    const t = setTimeout(fetchCustomers, 250); // debounce search typing
    return () => clearTimeout(t);
  }, [fetchCustomers]);

  function toggleSort(key) {
    if (sortKey === key) setSortDir(d => (d === "asc" ? "desc" : "asc"));
    else { setSortKey(key); setSortDir("desc"); }
    setPage(0);
  }

  const totalPages = Math.max(1, Math.ceil(total / pageSize));

  return (
    <div className="min-h-screen bg-gray-50 p-6 font-sans">
      <div className="max-w-6xl mx-auto flex flex-col gap-6">

        <div>
          <h1 className="text-xl font-semibold text-gray-900">Customer churn dashboard</h1>
          <p className="text-sm text-gray-500">Live data from /api &middot; {kpis?.total_customers?.toLocaleString() ?? "…"} customers scored</p>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <KpiCard icon={TrendingDown} label="High-risk %" value={kpis ? `${(kpis.high_risk_pct * 100).toFixed(1)}%` : null}
            sub="Of total customer base" accent="#D85A30" />
          <KpiCard icon={DollarSign} label="Revenue at risk" value={kpis ? currency(kpis.revenue_at_risk_monthly) : null}
            sub="Monthly, high-risk tier" accent="#E24B4A" />
          <KpiCard icon={AlertTriangle} label="High-risk accounts" value={kpis?.high_risk_count?.toLocaleString()}
            sub={kpis ? `${kpis.medium_risk_count.toLocaleString()} medium, ${kpis.low_risk_count.toLocaleString()} low` : null} accent="#EF9F27" />
          <KpiCard icon={Users} label="Total customers" value={kpis?.total_customers?.toLocaleString()}
            sub="Scored by the model" accent="#378ADD" />
        </div>

        <div className="bg-white rounded-xl border border-gray-200 p-4">
          <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
            <h2 className="text-sm font-semibold text-gray-900">At-risk customers</h2>
          </div>

          <div className="flex flex-wrap gap-2 mb-3">
            <div className="relative flex-1 min-w-[180px]">
              <Search size={14} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-gray-400" />
              <input
                value={search}
                onChange={e => { setSearch(e.target.value); setPage(0); }}
                placeholder="Search customer ID"
                className="w-full pl-8 pr-3 py-1.5 text-sm border border-gray-300 rounded-lg outline-none focus:border-gray-400"
              />
            </div>
            <select value={tierFilter} onChange={e => { setTierFilter(e.target.value); setPage(0); }}
              className="text-sm border border-gray-300 rounded-lg px-2 py-1.5">
              {["All", "High", "Medium", "Low"].map(t => <option key={t} value={t}>{t === "All" ? "All risk tiers" : `${t} risk`}</option>)}
            </select>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs text-gray-500 border-b border-gray-200">
                  <th className="py-2 pr-3 cursor-pointer select-none" onClick={() => toggleSort("customer_id")}>
                    <span className="flex items-center gap-1">Customer <ArrowUpDown size={11} /></span>
                  </th>
                  <th className="py-2 pr-3 cursor-pointer select-none" onClick={() => toggleSort("risk_score")}>
                    <span className="flex items-center gap-1">Risk score <ArrowUpDown size={11} /></span>
                  </th>
                  <th className="py-2 pr-3">Tier</th>
                  <th className="py-2">Top drivers</th>
                </tr>
              </thead>
              <tbody>
                {loading && <tr><td colSpan={4} className="py-6 text-center text-gray-400 text-sm">Loading…</td></tr>}
                {!loading && rows.map(c => {
                  const tc = TIER_COLOR[c.risk_tier];
                  return (
                    <tr key={c.customer_id} onClick={() => setSelectedId(c.customer_id)}
                      className="border-b border-gray-100 hover:bg-gray-50 cursor-pointer">
                      <td className="py-2 pr-3 font-medium text-gray-800">{c.customer_id}</td>
                      <td className="py-2 pr-3 text-gray-700">{(c.risk_score * 100).toFixed(0)}%</td>
                      <td className="py-2 pr-3">
                        <span className="text-xs font-medium px-2 py-0.5 rounded-full border"
                          style={{ background: tc.bg, color: tc.text, borderColor: tc.border }}>
                          {c.risk_tier}
                        </span>
                      </td>
                      <td className="py-2 text-gray-500 text-xs max-w-md truncate" title={c.top_drivers}>{c.top_drivers}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="flex items-center justify-between mt-3 text-xs text-gray-500">
            <span>{total.toLocaleString()} customers matched</span>
            <div className="flex items-center gap-2">
              <button disabled={page === 0} onClick={() => setPage(p => p - 1)}
                className="px-2 py-1 border border-gray-300 rounded disabled:opacity-40">Prev</button>
              <span>Page {page + 1} of {totalPages}</span>
              <button disabled={page >= totalPages - 1} onClick={() => setPage(p => p + 1)}
                className="px-2 py-1 border border-gray-300 rounded disabled:opacity-40">Next</button>
            </div>
          </div>
        </div>
      </div>

      <CustomerDrilldown customerId={selectedId} onClose={() => setSelectedId(null)} />
    </div>
  );
}
