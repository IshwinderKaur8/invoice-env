import { useMemo, useState } from "react";
import {
  BarElement,
  CategoryScale,
  Chart as ChartJS,
  Legend,
  LinearScale,
  Tooltip,
} from "chart.js";
import { Bar } from "react-chartjs-2";

import AgentOutput from "../components/AgentOutput.jsx";
import InvoiceViewer from "../components/InvoiceViewer.jsx";
import Navbar from "../components/Navbar.jsx";
import ProgressBar from "../components/ProgressBar.jsx";
import ScoreBoard from "../components/ScoreBoard.jsx";
import { getResults, resetEnv, runAgent, stepEnv } from "../services/api.js";

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend);

function heuristicAction(observation) {
  if (!observation) {
    return {
      extracted_fields: { vendor_name: "", invoice_date: "" },
      category: "Misc",
      anomaly_flag: false,
    };
  }

  const vendor = String(observation.vendor_name || "").toLowerCase();
  let category = "Misc";

  if (["uber", "lyft", "airlines", "marriott"].some((token) => vendor.includes(token))) {
    category = "Travel";
  } else if (["amazon", "staples", "ikea"].some((token) => vendor.includes(token))) {
    category = "Office Supplies";
  } else if (["electricity", "water", "internet", "gas"].some((token) => vendor.includes(token))) {
    category = "Utilities";
  }

  return {
    extracted_fields: {
      vendor_name: observation.vendor_name,
      invoice_date: observation.invoice_date,
    },
    category,
    anomaly_flag: Number(observation.amount || 0) > 2500,
  };
}

function Dashboard() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [observation, setObservation] = useState(null);
  const [reward, setReward] = useState(null);
  const [lastResult, setLastResult] = useState(null);
  const [state, setState] = useState(null);
  const [latestRun, setLatestRun] = useState(null);

  const activeMode = latestRun?.mode || null;

  const progress = useMemo(() => {
    const pointer = state?.state?.pointer ?? 0;
    const total = pointer + (state?.state?.remaining ?? 0);
    return { current: pointer, total };
  }, [state]);

  const chartData = useMemo(() => {
    const scores = latestRun?.results?.map((item, index) => ({
      label: `#${index + 1}`,
      extraction: item.reward.details.extraction,
      category: item.reward.details.category,
      anomaly: item.reward.details.anomaly,
      total: item.reward.score,
    })) || [];

    return {
      labels: scores.map((item) => item.label),
      datasets: [
        { label: "Total", data: scores.map((item) => item.total), backgroundColor: "#1d4ed8" },
        { label: "Extraction", data: scores.map((item) => item.extraction), backgroundColor: "#16a34a" },
        { label: "Category", data: scores.map((item) => item.category), backgroundColor: "#f97316" },
        { label: "Anomaly", data: scores.map((item) => item.anomaly), backgroundColor: "#e11d48" },
      ],
    };
  }, [latestRun]);

  const refreshResults = async () => {
    const resultData = await getResults(10);
    setLatestRun(resultData.latest_run);
  };

  const onReset = async () => {
    setLoading(true);
    setError("");
    try {
      const data = await resetEnv(12);
      setObservation(data.observation);
      setReward(null);
      setLastResult(null);
      setState({ state: data.state });
      await refreshResults();
    } catch (err) {
      setError(err?.response?.data?.detail || "Failed to reset environment");
    } finally {
      setLoading(false);
    }
  };

  const onNextStep = async () => {
    if (!observation) return;

    setLoading(true);
    setError("");
    try {
      const action = heuristicAction(observation);
      const data = await stepEnv(action);
      setObservation(data.observation);
      setReward(data.reward);
      setLastResult({ ...data, action });
      setState((prev) => ({ ...prev, state: { ...(prev?.state || {}), pointer: progress.current + 1, remaining: Math.max(progress.total - (progress.current + 1), 0) } }));
      if (data.done) {
        await refreshResults();
      }
    } catch (err) {
      setError(err?.response?.data?.detail || "Failed to execute step");
    } finally {
      setLoading(false);
    }
  };

  const onRunAgent = async () => {
    setLoading(true);
    setError("");
    try {
      const run = await runAgent(12, "auto");
      setLatestRun(run);
      const final = run.results?.[run.results.length - 1];
      if (final) {
        setObservation(final.observation);
        setReward(final.reward);
        setLastResult(final);
      }
      await refreshResults();
    } catch (err) {
      setError(err?.response?.data?.detail || "Failed to run agent");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-slate-100">
      <Navbar />

      <main className="mx-auto max-w-7xl p-6 space-y-6">
        <section className="card p-4 flex flex-wrap gap-3 items-center">
          <button onClick={onReset} disabled={loading} className="px-4 py-2 rounded-xl bg-brand-500 text-white font-medium hover:bg-brand-700 disabled:opacity-50">
            Reset Environment
          </button>
          <button onClick={onRunAgent} disabled={loading} className="px-4 py-2 rounded-xl bg-emerald-600 text-white font-medium hover:bg-emerald-700 disabled:opacity-50">
            Run Agent
          </button>
          <button onClick={onNextStep} disabled={loading || !observation} className="px-4 py-2 rounded-xl bg-slate-900 text-white font-medium hover:bg-slate-700 disabled:opacity-50">
            Next Step
          </button>
          <span className="text-sm text-slate-500">{loading ? "Processing..." : "Ready"}</span>
          {activeMode && (
            <span
              className={`text-xs px-3 py-1 rounded-full font-medium ${
                activeMode === "openai"
                  ? "bg-indigo-100 text-indigo-700"
                  : "bg-amber-100 text-amber-700"
              }`}
            >
              Mode: {activeMode}
            </span>
          )}
        </section>

        {error && <section className="rounded-xl border border-rose-200 bg-rose-50 text-rose-700 px-4 py-3">{error}</section>}

        <ProgressBar current={progress.current} total={progress.total} />

        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          <InvoiceViewer observation={observation} />
          <AgentOutput result={lastResult} />
        </div>

        <ScoreBoard reward={reward} />

        <section className="card p-6">
          <h2 className="text-lg font-semibold text-slate-900 mb-4">Results Chart</h2>
          {latestRun?.results?.length ? (
            <Bar
              data={chartData}
              options={{
                responsive: true,
                plugins: { legend: { position: "top" } },
                scales: { y: { beginAtZero: true, max: 1 } },
              }}
            />
          ) : (
            <p className="text-slate-500">Run an agent to generate analytics.</p>
          )}
        </section>
      </main>
    </div>
  );
}

export default Dashboard;
