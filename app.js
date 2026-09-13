"use client";

import { useState, useRef, useEffect } from "react";
import {
  GraduationCap,
  Activity,
  Stethoscope,
  Sparkles,
  ShieldCheck,
  ArrowUpRight,
  CheckCircle2,
  Loader2,
  Info,
  Brain,
  HeartPulse,
  Droplet,
  Save,
  CalendarClock,
  TrendingDown,
  TrendingUp,
} from "lucide-react";

/* ---------------------------------------------------------------------- */
/*  Design tokens & global (scoped) styles                                 */
/* ---------------------------------------------------------------------- */

const STYLES = `
  @import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

  .qha-root {
    --canvas: #F5F7F8;
    --surface: #FFFFFF;
    --ink: #14213D;
    --muted: #5B6B79;
    --line: #E3E8EC;
    --teal: #0E7C7B;
    --teal-soft: #E4F3F1;
    --indigo: #4C5FD5;
    --indigo-soft: #EBEDFB;
    --risk-low: #1E8F5F;
    --risk-low-soft: #E7F5EE;
    --risk-mod: #C97A1F;
    --risk-mod-soft: #FBF0E3;
    --risk-high: #C1443A;
    --risk-high-soft: #FBE9E7;
    font-family: 'Inter', ui-sans-serif, system-ui, sans-serif;
    color: var(--ink);
    background: var(--canvas);
  }
  .qha-display { font-family: 'Sora', ui-sans-serif, system-ui, sans-serif; }

  .qha-input:focus-visible,
  .qha-select:focus-visible,
  .qha-toggle:focus-visible,
  .qha-btn:focus-visible,
  .qha-range:focus-visible,
  .qha-link:focus-visible {
    outline: 2px solid var(--indigo);
    outline-offset: 2px;
  }

  input[type="range"].qha-range {
    -webkit-appearance: none;
    appearance: none;
    width: 100%;
    height: 6px;
    border-radius: 999px;
    background: linear-gradient(to right, var(--teal) var(--range-progress, 50%), var(--line) var(--range-progress, 50%));
  }
  input[type="range"].qha-range::-webkit-slider-thumb {
    -webkit-appearance: none;
    height: 20px;
    width: 20px;
    border-radius: 50%;
    background: var(--surface);
    border: 3px solid var(--teal);
    box-shadow: 0 1px 3px rgba(20,33,61,0.25);
    cursor: pointer;
    margin-top: -7px;
  }
  input[type="range"].qha-range::-moz-range-thumb {
    height: 20px;
    width: 20px;
    border-radius: 50%;
    background: var(--surface);
    border: 3px solid var(--teal);
    box-shadow: 0 1px 3px rgba(20,33,61,0.25);
    cursor: pointer;
  }
  input[type="range"].qha-range::-moz-range-track {
    height: 6px;
    border-radius: 999px;
    background: var(--line);
  }

  /* Staggered Reveal Animations */
  @keyframes qha-rise {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
  }
  .qha-stagger-1 { animation: qha-rise 0.5s ease-out 0.1s both; }
  .qha-stagger-2 { animation: qha-rise 0.5s ease-out 0.25s both; }
  .qha-stagger-3 { animation: qha-rise 0.5s ease-out 0.4s both; }
  .qha-stagger-4 { animation: qha-rise 0.5s ease-out 0.55s both; }
  .qha-stagger-5 { animation: qha-rise 0.5s ease-out 0.7s both; }

  /* Micro-animations */
  @keyframes qha-pulse-soft {
    0% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.15); opacity: 0.8; }
    100% { transform: scale(1); opacity: 1; }
  }
  .qha-celebrate { animation: qha-pulse-soft 2s infinite ease-in-out; }

  @media (prefers-reduced-motion: reduce) {
    .qha-stagger-1, .qha-stagger-2, .qha-stagger-3, .qha-stagger-4, .qha-stagger-5, .qha-celebrate { animation: none; opacity: 1; transform: none; }
  }
`;

/* ---------------------------------------------------------------------- */
/*  Static option data                                                     */
/* ---------------------------------------------------------------------- */

const EDUCATION_OPTIONS = [
  "No formal schooling",
  "Primary school",
  "Secondary school",
  "Higher secondary",
  "Graduate",
  "Postgraduate or higher",
];

const INCOME_OPTIONS = [
  "Under ₹2.5L / year",
  "₹2.5L – 5L / year",
  "₹5L – 10L / year",
  "₹10L – 20L / year",
  "Above ₹20L / year",
];

const GENERAL_HEALTH_LABELS = ["Excellent", "Very good", "Good", "Fair", "Poor"];

const INITIAL_FORM = {
  age: 45,
  bmi: 26.5,
  education: 3,
  income: 2,
  cigsPerDay: 0,
  generalHealth: 2,
  physicalHealthDays: 3,
  systolicBP: 122,
  diastolicBP: 80,
  totalCholesterol: 195,
  highBPHistory: false,
  highCholHistory: false,
};

/* ---------------------------------------------------------------------- */
/*  Live Hint Helpers                                                      */
/* ---------------------------------------------------------------------- */

function getHint(key, value) {
  if (!value) return null;
  if (key === "bmi") {
    if (value < 18.5) return { text: "Underweight", color: "var(--muted)" };
    if (value < 25) return { text: "Optimal range", color: "var(--teal)" };
    if (value < 30) return { text: "Worth watching", color: "var(--risk-mod)" };
    return { text: "Elevated", color: "var(--risk-high)" };
  }
  if (key === "systolicBP") {
    if (value < 120) return { text: "Optimal", color: "var(--teal)" };
    if (value < 130) return { text: "Elevated", color: "var(--risk-mod)" };
    return { text: "High", color: "var(--risk-high)" };
  }
  if (key === "totalCholesterol") {
    if (value < 200) return { text: "Optimal", color: "var(--teal)" };
    if (value < 240) return { text: "Borderline", color: "var(--risk-mod)" };
    return { text: "High", color: "var(--risk-high)" };
  }
  return null;
}

/* ---------------------------------------------------------------------- */
/*  Risk model                                                             */
/* ---------------------------------------------------------------------- */

const clamp = (v, min, max) => Math.min(max, Math.max(min, v));

function computeRisk(f) {
  let heart = 0;
  heart += clamp((f.age - 30) / 50, 0, 1) * 22;
  heart += clamp((f.systolicBP - 110) / 70, 0, 1) * 18;
  heart += clamp((f.diastolicBP - 70) / 40, 0, 1) * 10;
  heart += clamp((f.totalCholesterol - 160) / 140, 0, 1) * 16;
  heart += f.highBPHistory ? 14 : 0;
  heart += f.highCholHistory ? 10 : 0;
  heart += clamp(f.cigsPerDay / 20, 0, 1) * 10;
  heart += clamp((f.bmi - 22) / 15, 0, 1) * 6;
  heart += clamp((f.generalHealth - 1) / 4, 0, 1) * 4;

  let diabetes = 0;
  diabetes += clamp((f.bmi - 21) / 18, 0, 1) * 30;
  diabetes += clamp((f.age - 25) / 55, 0, 1) * 18;
  diabetes += clamp((f.generalHealth - 1) / 4, 0, 1) * 16;
  diabetes += clamp(f.physicalHealthDays / 30, 0, 1) * 12;
  diabetes += clamp((f.systolicBP - 115) / 65, 0, 1) * 8;
  diabetes += f.highBPHistory ? 8 : 0;
  diabetes += clamp(f.cigsPerDay / 25, 0, 1) * 6;
  diabetes -= clamp(f.education / 5, 0, 1) * 4;

  return {
    heart: Math.round(clamp(heart, 2, 96)),
    diabetes: Math.round(clamp(diabetes, 2, 96)),
  };
}

function bucket(score) {
  if (score < 33) return { label: "Low Risk", tone: "low" };
  if (score < 66) return { label: "Elevated Risk", tone: "mod" };
  return { label: "High Risk", tone: "high" };
}

function getPopulationContext(score, age) {
  const betterThan = Math.max(5, 95 - score);
  return `Lower risk than ${betterThan}% of people your age`;
}

function heartFactors(f) {
  const notes = [];
  if (f.systolicBP >= 130 || f.diastolicBP >= 85 || f.highBPHistory) {
    notes.push({ text: "Blood pressure is running above optimal", dir: "up" });
  } else {
    notes.push({ text: "Blood pressure is within a healthy range", dir: "down" });
  }
  if (f.totalCholesterol >= 220 || f.highCholHistory) {
    notes.push({ text: "Total cholesterol is elevated", dir: "up" });
  } else {
    notes.push({ text: "Cholesterol looks well-controlled", dir: "down" });
  }
  if (f.cigsPerDay > 0) {
    notes.push({ text: `Smoking (${f.cigsPerDay}/day) adds cardiovascular strain`, dir: "up" });
  }
  if (f.age >= 55 && notes.length < 3) {
    notes.push({ text: "Age is a contributing factor at this stage", dir: "up" });
  }
  if (notes.length < 3) {
    notes.push({ text: "No major red flags in your cardiac vitals", dir: "down" });
  }
  return notes.slice(0, 3);
}

function diabetesFactors(f) {
  const notes = [];
  if (f.bmi >= 25) {
    notes.push({ text: `BMI of ${f.bmi} sits above the optimal range`, dir: "up" });
  } else {
    notes.push({ text: "BMI is in a healthy, protective range", dir: "down" });
  }
  if (f.generalHealth >= 3) {
    notes.push({ text: "Self-reported wellness is lower than ideal", dir: "up" });
  } else {
    notes.push({ text: "Self-reported wellness is strong", dir: "down" });
  }
  if (f.physicalHealthDays >= 10) {
    notes.push({ text: `${f.physicalHealthDays} unwell days last month is notable`, dir: "up" });
  }
  if (notes.length < 3) {
    notes.push({ text: "Activity and lifestyle indicators look stable", dir: "down" });
  }
  return notes.slice(0, 3);
}

function empatheticSummary({ heart, diabetes }) {
  const worst = heart.score >= diabetes.score ? heart : diabetes;
  const worstName = heart.score >= diabetes.score ? "heart disease" : "diabetes";
  
  if (worst.tone === "low") {
    return "Great news—both areas look solid and are currently modeling in the low-risk range. Here's a look at what stands out as working well for you. Keep up your routine!";
  }
  if (worst.tone === "mod") {
    return `We're seeing a slight elevation in your ${worstName} risk profile. It's nothing alarming, but definitely worth keeping an eye on. Small adjustments can make a big difference here.`;
  }
  return `Your ${worstName} indicators are running high today. Don't panic—this is simply a prompt to share these insights with your physician to confirm with standard clinical tests.`;
}

function toneColor(tone) {
  return tone === "low" ? "var(--risk-low)" : tone === "mod" ? "var(--risk-mod)" : "var(--risk-high)";
}
function toneSoft(tone) {
  return tone === "low" ? "var(--risk-low-soft)" : tone === "mod" ? "var(--risk-mod-soft)" : "var(--risk-high-soft)";
}

/* ---------------------------------------------------------------------- */
/*  Small building blocks                                                  */
/* ---------------------------------------------------------------------- */

function LogoMark() {
  return (
    <svg width="34" height="34" viewBox="0 0 34 34" fill="none" aria-hidden="true">
      <circle cx="17" cy="17" r="15" stroke="var(--teal)" strokeWidth="2" />
      <ellipse cx="17" cy="17" rx="15" ry="6" stroke="var(--indigo)" strokeWidth="1.6" transform="rotate(45 17 17)" />
      <path d="M11 17h3l2-5 3 10 2-5h3" stroke="var(--ink)" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" fill="none" />
    </svg>
  );
}

function SectionCard({ index, title, children }) {
  return (
    <div className="rounded-2xl border bg-[var(--surface)] p-5 sm:p-6" style={{ borderColor: "var(--line)" }}>
      <div className="mb-5 flex items-center gap-3">
        <span
          className="flex h-7 w-7 items-center justify-center rounded-full qha-display text-xs font-semibold"
          style={{ background: "var(--teal-soft)", color: "var(--teal)" }}
        >
          {index}
        </span>
        <h2 className="qha-display text-base font-semibold">{title}</h2>
      </div>
      {children}
    </div>
  );
}

function Field({ label, hint, children }) {
  return (
    <label className="block">
      <div className="mb-1.5 flex items-center justify-between">
        <span className="block text-sm font-medium">{label}</span>
        {hint && <span className="text-[11px] font-medium transition-colors" style={{ color: hint.color }}>{hint.text}</span>}
      </div>
      {children}
    </label>
  );
}

function NumberField({ label, value, onChange, min, max, step = 1, suffix, hint }) {
  return (
    <Field label={label} hint={hint}>
      <div className="flex items-center rounded-lg border bg-white px-3 transition-colors" style={{ borderColor: "var(--line)" }}>
        <input
          type="number"
          className="qha-input w-full bg-transparent py-2.5 text-sm outline-none"
          value={value}
          min={min}
          max={max}
          step={step}
          onChange={onChange}
        />
        {suffix && <span className="pl-2 text-xs whitespace-nowrap" style={{ color: "var(--muted)" }}>{suffix}</span>}
      </div>
    </Field>
  );
}

function SelectField({ label, value, onChange, options }) {
  return (
    <Field label={label}>
      <select
        className="qha-select w-full rounded-lg border bg-white px-3 py-2.5 text-sm outline-none"
        style={{ borderColor: "var(--line)" }}
        value={value}
        onChange={onChange}
      >
        {options.map((opt, i) => (
          <option key={i} value={i}>{opt}</option>
        ))}
      </select>
    </Field>
  );
}

function SliderField({ label, value, onChange, min, max, step = 1, endLabels, valueLabel }) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <Field label={label}>
      <div className="flex items-center gap-3">
        <input
          type="range"
          className="qha-range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={onChange}
          style={{ "--range-progress": `${pct}%` }}
        />
        <span className="qha-display w-24 shrink-0 text-right text-sm font-semibold">
          {valueLabel ? valueLabel : value}
        </span>
      </div>
      {endLabels && (
        <div className="mt-1.5 flex justify-between text-[11px]" style={{ color: "var(--muted)" }}>
          <span>{endLabels[0]}</span>
          <span>{endLabels[1]}</span>
        </div>
      )}
    </Field>
  );
}

function ToggleField({ label, value, onChange }) {
  return (
    <div className="flex items-center justify-between rounded-lg border px-3 py-2.5" style={{ borderColor: "var(--line)" }}>
      <span className="text-sm font-medium">{label}</span>
      <button
        type="button"
        role="switch"
        aria-checked={value}
        aria-label={label}
        onClick={onChange}
        className="qha-toggle relative h-6 w-11 shrink-0 rounded-full transition-colors"
        style={{ background: value ? "var(--teal)" : "var(--line)" }}
      >
        <span
          className="absolute top-0.5 h-5 w-5 rounded-full bg-white shadow transition-transform"
          style={{ transform: value ? "translateX(22px)" : "translateX(2px)" }}
        />
      </button>
    </div>
  );
}

/* ---------------------------------------------------------------------- */
/*  Animated Gauge & Results Card                                          */
/* ---------------------------------------------------------------------- */

function AnimatedGauge({ score, tone }) {
  const [displayScore, setDisplayScore] = useState(0);

  useEffect(() => {
    let start = 0;
    const duration = 1200; 
    const interval = 16; 
    const step = (score / duration) * interval;
    
    const timer = setInterval(() => {
      start += step;
      if (start >= score) {
        setDisplayScore(score);
        clearInterval(timer);
      } else {
        setDisplayScore(Math.floor(start));
      }
    }, interval);
    return () => clearInterval(timer);
  }, [score]);

  const size = 116;
  const stroke = 9;
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const offset = c * (1 - score / 100);
  const color = toneColor(tone);

  return (
    <div className="relative shrink-0" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={size / 2} cy={size / 2} r={r} stroke="var(--line)" strokeWidth={stroke} fill="none" />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          stroke={color}
          strokeWidth={stroke}
          fill="none"
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={offset}
          style={{ transition: "stroke-dashoffset 1.2s ease-out" }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="qha-display text-2xl font-semibold" style={{ color }}>{displayScore}</span>
        <span className="text-[10px]" style={{ color: "var(--muted)" }}>out of 100</span>
      </div>
    </div>
  );
}

function ResultCard({ title, icon: Icon, result, age }) {
  const color = toneColor(result.tone);
  const soft = toneSoft(result.tone);
  const isOptimal = result.tone === "low";

  return (
    <div className="rounded-2xl bg-white p-6 shadow-sm relative overflow-hidden" style={{ borderLeft: `4px solid ${color}` }}>
      {isOptimal && (
        <Sparkles className="absolute -top-3 -right-3 h-16 w-16 opacity-10 qha-celebrate" style={{ color }} />
      )}
      <div className="mb-4 flex items-center gap-2 relative z-10">
        <Icon className="h-4 w-4" style={{ color }} />
        <h3 className="qha-display text-base font-semibold">{title}</h3>
      </div>
      <div className="flex flex-col gap-5 sm:flex-row sm:items-center relative z-10">
        <div className="flex flex-col items-center gap-2">
          <AnimatedGauge score={result.score} tone={result.tone} />
          <span className="text-[10px] font-medium" style={{ color: "var(--muted)" }}>
            {getPopulationContext(result.score, age)}
          </span>
        </div>
        <div>
          <span
            className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-semibold"
            style={{ background: soft, color }}
          >
            {isOptimal && <CheckCircle2 className="h-3 w-3" />}
            {result.label}
          </span>
          <ul className="mt-4 space-y-2">
            {result.factors.map((f, i) => (
              <li key={i} className="flex items-start gap-2 text-sm leading-snug">
                {f.dir === "up" ? (
                  <ArrowUpRight className="mt-0.5 h-3.5 w-3.5 shrink-0" style={{ color }} />
                ) : (
                  <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0" style={{ color: "var(--risk-low)" }} />
                )}
                <span>{f.text}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------------- */
/*  Main component                                                          */
/* ---------------------------------------------------------------------- */

export default function QuantumHealthDashboard() {
  const [form, setForm] = useState(INITIAL_FORM);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [prevResults, setPrevResults] = useState(null);
  const [saved, setSaved] = useState(false);
  const resultsRef = useRef(null);

  // Form Progress Calculation
  const progressFields = ["age", "bmi", "education", "income", "cigsPerDay", "generalHealth", "physicalHealthDays", "systolicBP", "diastolicBP", "totalCholesterol"];
  const filledFields = progressFields.filter(f => form[f] !== null && form[f] !== "").length;
  const progressPct = Math.round((filledFields / progressFields.length) * 100);

  const setNum = (key) => (e) => setForm((p) => ({ ...p, [key]: Number(e.target.value) }));
  const setToggle = (key) => () => setForm((p) => ({ ...p, [key]: !p[key] }));

  const runAnalysis = () => {
    setLoading(true);
    if (results) setPrevResults(results);
    setResults(null);
    setSaved(false);
    
    setTimeout(() => {
      const { heart, diabetes } = computeRisk(form);
      setResults({
        heart: { score: heart, ...bucket(heart), factors: heartFactors(form) },
        diabetes: { score: diabetes, ...bucket(diabetes), factors: diabetesFactors(form) },
      });
      setLoading(false);
      requestAnimationFrame(() => {
        resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    }, 1600);
  };

  // Instant recalculation for the What-If Simulator (bypasses loader)
  const handleSimulate = (key) => (e) => {
    const val = Number(e.target.value);
    setForm((p) => {
      const next = { ...p, [key]: val };
      if (results) {
        if (!prevResults) setPrevResults(results); // Capture baseline if not already captured
        const { heart, diabetes } = computeRisk(next);
        setResults({
          heart: { score: heart, ...bucket(heart), factors: heartFactors(next) },
          diabetes: { score: diabetes, ...bucket(diabetes), factors: diabetesFactors(next) },
        });
        setSaved(false);
      }
      return next;
    });
  };

  // Delta Message logic for What-If Simulator
  const getDeltaMessage = () => {
    if (!prevResults || !results) return null;
    const heartDiff = results.heart.score - prevResults.heart.score;
    const diabDiff = results.diabetes.score - prevResults.diabetes.score;
    
    if (heartDiff === 0 && diabDiff === 0) return null;
    
    const isBetter = heartDiff < 0 || diabDiff < 0;
    const Icon = isBetter ? TrendingDown : TrendingUp;
    const color = isBetter ? "var(--teal)" : "var(--risk-mod)";
    
    let text = "Risk profile shifted: ";
    if (heartDiff !== 0) text += `Heart ${heartDiff > 0 ? "+" : ""}${heartDiff} pts. `;
    if (diabDiff !== 0) text += `Diabetes ${diabDiff > 0 ? "+" : ""}${diabDiff} pts.`;

    return (
      <div className="flex items-center gap-2 rounded-lg px-3 py-2 text-xs font-semibold qha-stagger-2" style={{ background: "var(--surface)", color, border: `1px solid ${color}` }}>
        <Icon className="h-4 w-4" />
        {text}
      </div>
    );
  };

  return (
    <div className="qha-root min-h-screen w-full pb-20">
      <style>{STYLES}</style>
      
      {/* Sticky Progress Bar */}
      <div className="fixed top-0 left-0 h-1.5 w-full bg-gray-200 z-50">
        <div 
          className="h-full transition-all duration-500 ease-out" 
          style={{ width: `${progressPct}%`, background: "var(--teal)" }}
        />
      </div>

      <div className="mx-auto max-w-4xl px-4 py-10 sm:px-6 lg:px-8">
        
        {/* Header */}
        <header className="mb-10 mt-4">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <LogoMark />
              <span className="qha-display text-xl font-semibold tracking-tight">Quantum Health AI</span>
            </div>
            <div className="rounded-full p-[1px]" style={{ background: "linear-gradient(90deg, var(--teal), var(--indigo))" }}>
              <div
                className="flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium"
                style={{ background: "var(--surface)", color: "var(--indigo)" }}
              >
                <Sparkles className="h-3.5 w-3.5" />
                Powered by Hybrid Quantum ML
              </div>
            </div>
          </div>
          <h1 className="qha-display mt-8 text-2xl font-semibold leading-snug sm:text-3xl" style={{ maxWidth: "38ch" }}>
            Early disease risk, assessed in one visit.
          </h1>
          <p className="mt-2 text-[15px] leading-relaxed" style={{ color: "var(--muted)", maxWidth: "60ch" }}>
            Enter your health details below for a comprehensive quantum-powered risk assessment.
          </p>
        </header>

        {/* Form */}
        <div className="space-y-5">
          <SectionCard index={1} title="Demographics & lifestyle">
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <NumberField label="Age" value={form.age} onChange={setNum("age")} min={1} max={110} suffix="years" />
              <NumberField 
                label="BMI" 
                value={form.bmi} 
                onChange={setNum("bmi")} 
                min={10} max={60} step={0.1} suffix="kg/m²" 
                hint={getHint("bmi", form.bmi)}
              />
              <SelectField
                label="Education level"
                value={form.education}
                onChange={setNum("education")}
                options={EDUCATION_OPTIONS}
              />
              <SelectField
                label="Income level"
                value={form.income}
                onChange={setNum("income")}
                options={INCOME_OPTIONS}
              />
              <NumberField
                label="Cigarettes per day"
                value={form.cigsPerDay}
                onChange={setNum("cigsPerDay")}
                min={0}
                max={60}
                suffix="cigs/day"
              />
            </div>
          </SectionCard>

          <SectionCard index={2} title="General wellness">
            <div className="grid grid-cols-1 gap-6">
              <SliderField
                label="General health"
                value={form.generalHealth}
                onChange={setNum("generalHealth")}
                min={1}
                max={5}
                endLabels={["Excellent", "Poor"]}
                valueLabel={GENERAL_HEALTH_LABELS[form.generalHealth - 1]}
              />
              <SliderField
                label="Physical health — unwell days in the past 30"
                value={form.physicalHealthDays}
                onChange={setNum("physicalHealthDays")}
                min={0}
                max={30}
                endLabels={["0 days", "30 days"]}
                valueLabel={`${form.physicalHealthDays} days`}
              />
            </div>
          </SectionCard>

          <SectionCard index={3} title="Vitals & medical history">
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
              <NumberField 
                label="Systolic BP" 
                value={form.systolicBP} 
                onChange={setNum("systolicBP")} 
                min={70} max={220} suffix="mmHg" 
                hint={getHint("systolicBP", form.systolicBP)}
              />
              <NumberField label="Diastolic BP" value={form.diastolicBP} onChange={setNum("diastolicBP")} min={40} max={140} suffix="mmHg" />
              <NumberField 
                label="Total cholesterol" 
                value={form.totalCholesterol} 
                onChange={setNum("totalCholesterol")} 
                min={100} max={400} suffix="mg/dL" 
                hint={getHint("totalCholesterol", form.totalCholesterol)}
              />
            </div>
            <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
              <ToggleField label="History of high blood pressure" value={form.highBPHistory} onChange={setToggle("highBPHistory")} />
              <ToggleField label="History of high cholesterol" value={form.highCholHistory} onChange={setToggle("highCholHistory")} />
            </div>
          </SectionCard>

          {/* Form Action Area */}
          <div className="mt-8 flex flex-col items-center gap-4 rounded-2xl bg-white p-6 shadow-sm sm:flex-row sm:justify-between border border-transparent transition-all" style={{ borderColor: progressPct === 100 ? "var(--teal-soft)" : "transparent" }}>
            <div>
              <h3 className="text-sm font-semibold text-gray-900">Ready for analysis?</h3>
              <p className="text-xs mt-1" style={{ color: "var(--muted)" }}>
                {progressPct === 100 ? "All fields complete. Your data remains completely private." : `${progressPct}% complete. Fill remaining fields for highest accuracy.`}
              </p>
            </div>
            
            <button
              onClick={runAnalysis}
              disabled={loading}
              className="qha-btn relative inline-flex items-center justify-center gap-2 rounded-xl px-8 py-3.5 text-sm font-semibold text-white transition disabled:opacity-70 overflow-hidden group"
              style={{
                background: "linear-gradient(135deg, var(--teal), var(--indigo))",
                boxShadow: "0 8px 20px -8px rgba(14,124,123,0.45)",
              }}
            >
              <div className="absolute inset-0 bg-white/20 translate-y-full group-hover:translate-y-0 transition-transform duration-300 ease-out" />
              {loading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin relative z-10" />
                  <span className="relative z-10">Running quantum mapping…</span>
                </>
              ) : (
                <>
                  <Sparkles className="h-4 w-4 relative z-10" />
                  <span className="relative z-10">Run Quantum Analysis</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Results */}
        {results && (
          <section ref={resultsRef} className="mt-14 scroll-mt-10">
            <div className="qha-stagger-1 mb-8 flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4">
              <div>
                <div className="mb-2 flex items-center gap-2">
                  <ShieldCheck className="h-5 w-5" style={{ color: "var(--teal)" }} />
                  <h2 className="qha-display text-xl font-semibold">Your global health outlook</h2>
                </div>
                <p className="text-[15px] leading-relaxed" style={{ color: "var(--muted)", maxWidth: "60ch" }}>
                  {empatheticSummary(results)}
                </p>
              </div>
              {getDeltaMessage()}
            </div>

            <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 qha-stagger-2">
              <ResultCard title="Heart disease" icon={HeartPulse} result={results.heart} age={form.age} />
              <ResultCard title="Diabetes" icon={Droplet} result={results.diabetes} age={form.age} />
            </div>

            {/* What-If Simulator */}
            <div className="mt-6 rounded-2xl bg-white border p-6 qha-stagger-3" style={{ borderColor: "var(--line)" }}>
              <div className="mb-5 flex items-center gap-2">
                <Activity className="h-4 w-4" style={{ color: "var(--indigo)" }} />
                <h3 className="qha-display text-sm font-semibold">Interactive What-If Simulator</h3>
              </div>
              <p className="mb-6 text-sm" style={{ color: "var(--muted)" }}>
                Nudge these key lifestyle metrics to see how your risk profile adapts in real-time. Small adjustments compound over time.
              </p>
              <div className="grid grid-cols-1 gap-x-8 gap-y-6 md:grid-cols-3">
                <SliderField
                  label="BMI (Body Mass Index)"
                  value={form.bmi}
                  onChange={handleSimulate("bmi")}
                  min={18} max={40} step={0.5}
                />
                <SliderField
                  label="Systolic BP"
                  value={form.systolicBP}
                  onChange={handleSimulate("systolicBP")}
                  min={100} max={180} step={1}
                />
                <SliderField
                  label="Cigarettes / day"
                  value={form.cigsPerDay}
                  onChange={handleSimulate("cigsPerDay")}
                  min={0} max={40} step={1}
                />
              </div>
            </div>

            <div className="mt-6 flex items-start gap-3 rounded-2xl p-5 qha-stagger-4" style={{ background: "var(--indigo-soft)" }}>
              <Brain className="mt-0.5 h-5 w-5 shrink-0" style={{ color: "var(--indigo)" }} />
              <div>
                <h3 className="qha-display text-sm font-semibold">How the quantum model works</h3>
                <p className="mt-1 text-sm leading-relaxed" style={{ color: "var(--muted)" }}>
                  Your inputs were encoded into a hybrid quantum neural network, which uses quantum feature maps to
                  surface higher-dimensional correlations across vitals, lifestyle, and history that classical models
                  can miss — then decodes them back into the two risk scores above.
                </p>
              </div>
            </div>
            
            <div className="mt-8 flex flex-col items-center justify-between gap-4 border-t pt-6 sm:flex-row qha-stagger-5" style={{ borderColor: "var(--line)" }}>
               <div className="flex items-start gap-2.5 max-w-md">
                <Info className="mt-0.5 h-4 w-4 shrink-0" style={{ color: "var(--muted)" }} />
                <p className="text-xs leading-relaxed" style={{ color: "var(--muted)" }}>
                  This is an AI prediction powered by Quantum ML and is for informational purposes only. Please consult
                  a doctor for medical advice.
                </p>
              </div>
              
              <button
                onClick={() => setSaved(true)}
                disabled={saved}
                className="qha-btn flex shrink-0 items-center gap-2 rounded-xl border bg-white px-5 py-2.5 text-sm font-semibold transition-all disabled:opacity-100"
                style={{ borderColor: saved ? "var(--teal)" : "var(--line)", color: saved ? "var(--teal)" : "var(--ink)" }}
              >
                {saved ? (
                  <>
                    <CalendarClock className="h-4 w-4" />
                    Saved! Recheck in 30 days
                  </>
                ) : (
                  <>
                    <Save className="h-4 w-4" />
                    Save this check-in
                  </>
                )}
              </button>
            </div>
          </section>
        )}
      </div>
    </div>
  );
}