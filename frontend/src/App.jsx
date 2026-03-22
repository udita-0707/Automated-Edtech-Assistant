import { useState, useEffect } from 'react';
import axios from 'axios';
import { CheckCircle2, AlertCircle, XCircle, ChevronDown, History, BookOpen, BarChart3, Loader2 } from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs) {
  return twMerge(clsx(inputs));
}

export default function App() {
  const [activeTab, setActiveTab] = useState('grade');

  return (
    <div className="min-h-screen w-full text-slate-100 flex flex-col items-center">
      <header className="w-full max-w-4xl pt-12 pb-8 px-6 text-center">
        <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent transform transition-all hover:scale-[1.02]">
          Automated EdTech Assistant
        </h1>
        <p className="mt-4 text-slate-300 text-lg md:text-xl font-light">
          AI-powered semantic grading and insightful feedback
        </p>
      </header>
      
      <main className="flex-1 w-full max-w-4xl px-4 md:px-0 pb-12">
        {/* Navigation Tabs */}
        <div className="flex justify-center space-x-2 mb-8 bg-slate-900/50 backdrop-blur-md p-2 rounded-2xl w-max mx-auto border border-slate-700/50">
          <button 
            className={cn("px-6 py-2.5 rounded-xl font-medium transition-all flex items-center gap-2", activeTab === 'grade' ? "bg-blue-600 shadow-lg shadow-blue-900/40 text-white" : "text-slate-400 hover:text-white hover:bg-slate-800")}
            onClick={() => setActiveTab('grade')}
          >
            <BookOpen size={18} /> Grade Answer
          </button>
          <button 
            className={cn("px-6 py-2.5 rounded-xl font-medium transition-all flex items-center gap-2", activeTab === 'history' ? "bg-purple-600 shadow-lg shadow-purple-900/40 text-white" : "text-slate-400 hover:text-white hover:bg-slate-800")}
            onClick={() => setActiveTab('history')}
          >
            <History size={18} /> History
          </button>
          <button 
            className={cn("px-6 py-2.5 rounded-xl font-medium transition-all flex items-center gap-2", activeTab === 'analytics' ? "bg-emerald-600 shadow-lg shadow-emerald-900/40 text-white" : "text-slate-400 hover:text-white hover:bg-slate-800")}
            onClick={() => setActiveTab('analytics')}
          >
            <BarChart3 size={18} /> Analytics
          </button>
        </div>

        {/* Dynamic Content */}
        <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
          {activeTab === 'grade' && <GradePanel />}
          {activeTab === 'history' && <HistoryPanel />}
          {activeTab === 'analytics' && <AnalyticsPanel />}
        </div>
      </main>
    </div>
  );
}

// --------------------------------------------------------------------------

function GradePanel() {
  const [question, setQuestion] = useState('What is the function of the mitochondria in a cell?');
  const [reference, setReference] = useState('The mitochondria are organelles that act like a digestive system which takes in nutrients, breaks them down, and creates energy rich molecules for the cell. The biochemical processes of the cell are known as cellular respiration. Many of the reactions involved in cellular respiration happen in the mitochondria. Mitochondria are the working organelles that keep the cell full of energy.');
  const [student, setStudent] = useState('It is the powerhouse of the cell, generating energy from nutrients.');
  
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const handleGrade = async () => {
    if (!question || !reference || !student) {
      setError('Please fill in all fields.');
      return;
    }
    
    setError('');
    setLoading(true);
    setResult(null);

    try {
      const { data } = await axios.post('http://localhost:3000/api/grade', {
        question,
        reference_answer: reference,
        student_answer: student
      });
      setResult(data);
    } catch (err) {
      console.error(err);
      setError('Failed to grade answer. Please ensure the backend is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
      {/* Input Form */}
      <div className="bg-slate-800/60 backdrop-blur-xl border border-slate-700/50 rounded-3xl p-6 lg:p-8 shadow-2xl">
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2 text-slate-100">
          <BookOpen className="text-blue-400" /> Assessment Details
        </h2>
        
        <div className="space-y-5">
          <div>
            <label className="block text-sm font-medium text-slate-400 mb-2">Question</label>
            <textarea 
              rows="2"
              className="w-full bg-slate-900/80 border border-slate-700 rounded-xl px-4 py-3 focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all resize-none font-medium"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-400 mb-2">Reference Answer</label>
            <textarea 
              rows="4"
              className="w-full bg-slate-900/80 border border-slate-700 rounded-xl px-4 py-3 focus:ring-2 focus:ring-green-500 focus:border-transparent outline-none transition-all resize-none leading-relaxed"
              value={reference}
              onChange={(e) => setReference(e.target.value)}
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-slate-400 mb-2">Student Answer</label>
            <textarea 
              rows="4"
              className="w-full bg-slate-900/80 border border-slate-700 rounded-xl px-4 py-3 focus:ring-2 focus:ring-purple-500 focus:border-transparent outline-none transition-all resize-none leading-relaxed"
              value={student}
              onChange={(e) => setStudent(e.target.value)}
            />
          </div>
          
          {error && <div className="text-red-400 text-sm py-2 px-4 bg-red-900/30 rounded-lg">{error}</div>}
          
          <button 
            className="w-full py-4 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-bold text-lg shadow-lg shadow-blue-900/50 transition-all flex justify-center items-center gap-2 disabled:opacity-70 group"
            onClick={handleGrade}
            disabled={loading}
          >
            {loading ? <Loader2 className="animate-spin" /> : <span>Evaluate Response</span>}
            {!loading && <ChevronDown className="group-hover:translate-y-1 transition-transform" size={18} />}
          </button>
        </div>
      </div>

      {/* Results Dashboard */}
      <div className="bg-slate-800/60 backdrop-blur-xl border border-slate-700/50 rounded-3xl p-6 lg:p-8 shadow-2xl flex flex-col justify-center min-h-[400px]">
        {!result && !loading && (
          <div className="text-center text-slate-500 flex flex-col items-center">
            <div className="w-24 h-24 bg-slate-800 rounded-full flex items-center justify-center mb-6 border border-slate-700">
              <BookOpen size={40} className="text-slate-600" />
            </div>
            <p className="text-lg">Submit a response for evaluation</p>
            <p className="text-sm mt-2 opacity-70">The AI will analyze semantic similarity to grade the answer.</p>
          </div>
        )}
        
        {loading && (
          <div className="text-center flex flex-col items-center justify-center space-y-4">
            <Loader2 size={48} className="text-blue-500 animate-spin" />
            <p className="text-blue-400 font-medium animate-pulse">Running ML pipeline...</p>
          </div>
        )}

        {result && !loading && (
          <div className="space-y-6 animate-in zoom-in-95 duration-300">
            <div className="text-center">
              <h3 className="text-slate-400 uppercase tracking-widest text-sm font-bold mb-2">Final Evaluation</h3>
              
              <div className="flex justify-center mb-4">
                {result.predicted_label === 'correct' && <div className="bg-green-500/20 text-green-400 p-4 rounded-full border border-green-500/50"><CheckCircle2 size={48} /></div>}
                {result.predicted_label === 'partially correct' && <div className="bg-yellow-500/20 text-yellow-400 p-4 rounded-full border border-yellow-500/50"><AlertCircle size={48} /></div>}
                {result.predicted_label === 'incorrect' && <div className="bg-red-500/20 text-red-400 p-4 rounded-full border border-red-500/50"><XCircle size={48} /></div>}
              </div>
              
              <div className={cn(
                "text-3xl font-black uppercase tracking-wider mb-2",
                result.predicted_label === 'correct' ? "text-green-400 block" : "hidden"
              )}>Correct</div>
              <div className={cn(
                "text-3xl font-black uppercase tracking-wider mb-2",
                result.predicted_label === 'partially correct' ? "text-yellow-400 block" : "hidden"
              )}>Partially Correct</div>
              <div className={cn(
                "text-3xl font-black uppercase tracking-wider mb-2",
                result.predicted_label === 'incorrect' ? "text-red-400 block" : "hidden"
              )}>Incorrect</div>
            </div>

            <div className="bg-slate-900/70 p-6 rounded-2xl border border-slate-700/50">
              <div className="flex justify-between items-center mb-4 border-b border-slate-700/50 pb-4">
                <span className="text-slate-400 font-medium">Similarity Score</span>
                <span className="text-xl font-bold bg-slate-800 py-1 px-3 rounded-lg text-blue-300">
                  {(result.similarity_score * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex justify-between items-center mb-4 border-b border-slate-700/50 pb-4">
                <span className="text-slate-400 font-medium">Confidence Level</span>
                <span className="text-xl font-bold bg-slate-800 py-1 px-3 rounded-lg text-purple-300">
                  {(result.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <div className="mt-4">
                <span className="text-slate-400 font-medium block mb-2">Detailed Feedback</span>
                <p className="text-slate-200 leading-relaxed bg-blue-900/20 p-4 rounded-xl border border-blue-800/50">
                  {result.feedback}
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// --------------------------------------------------------------------------

function HistoryPanel() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios.get('http://localhost:3000/api/history')
      .then(res => {
        setHistory(res.data);
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setLoading(false);
      });
  }, []);

  if (loading) return <div className="flex justify-center p-12"><Loader2 className="animate-spin text-blue-500" size={32} /></div>;

  return (
    <div className="bg-slate-800/60 backdrop-blur-xl border border-slate-700/50 rounded-3xl p-6 shadow-2xl overflow-hidden">
      <h2 className="text-2xl font-bold mb-6 text-slate-100 flex items-center gap-2">
        <History className="text-purple-400" /> Recent Submissions
      </h2>
      
      {history.length === 0 ? (
        <div className="text-center py-12 text-slate-400">No submissions found. Go grade some answers!</div>
      ) : (
        <div className="space-y-4 max-h-[600px] overflow-y-auto pr-2 custom-scrollbar">
          {history.map(item => (
            <div key={item.id} className="bg-slate-900/60 p-5 rounded-2xl border border-slate-700/50 transition-all hover:bg-slate-900 hover:border-slate-600">
              <div className="flex justify-between items-start mb-3">
                <span className={cn(
                  "text-xs font-bold uppercase py-1 px-3 rounded-full tracking-wider",
                  item.predicted_label === 'correct' ? "bg-green-500/20 text-green-400" :
                  item.predicted_label === 'partially correct' ? "bg-yellow-500/20 text-yellow-400" :
                  "bg-red-500/20 text-red-400"
                )}>
                  {item.predicted_label}
                </span>
                <span className="text-xs text-slate-500 font-mono">
                  {new Date(item.created_at).toLocaleString()}
                </span>
              </div>
              <div className="text-sm font-medium text-slate-300 mb-2 truncate">Q: {item.question}</div>
              <div className="text-sm italic text-slate-400 px-4 py-2 border-l-2 border-slate-700 bg-slate-900/50 mb-3 line-clamp-2">
                "{item.student_answer}"
              </div>
              <div className="flex gap-4 text-xs font-semibold">
                <div className="bg-slate-800 px-3 py-1.5 rounded-lg text-blue-300">Sim: {(item.similarity_score * 100).toFixed(1)}%</div>
                <div className="bg-slate-800 px-3 py-1.5 rounded-lg text-purple-300">Conf: {(item.confidence * 100).toFixed(1)}%</div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// --------------------------------------------------------------------------

function AnalyticsPanel() {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    axios.get('http://localhost:3000/api/analytics')
      .then(res => {
        setStats(res.data);
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setLoading(false);
      });
  }, []);

  if (loading) return <div className="flex justify-center p-12"><Loader2 className="animate-spin text-blue-500" size={32} /></div>;
  if (!stats) return <div className="text-center py-12 text-slate-400">Failed to load analytics</div>;

  return (
    <div className="bg-slate-800/60 backdrop-blur-xl border border-slate-700/50 rounded-3xl p-6 shadow-2xl">
      <h2 className="text-2xl font-bold mb-8 text-slate-100 flex items-center gap-2">
        <BarChart3 className="text-emerald-400" /> Platform Insights
      </h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <div className="bg-slate-900/70 p-6 rounded-2xl border border-slate-700/50 flex flex-col justify-center items-center">
          <div className="text-slate-400 font-medium mb-2 uppercase tracking-wider text-sm">Total Graded</div>
          <div className="text-5xl font-black text-white">{stats.total_submissions}</div>
        </div>
        <div className="bg-slate-900/70 p-6 rounded-2xl border border-blue-900/30 shadow-[inset_0_0_20px_rgba(59,130,246,0.1)] flex flex-col justify-center items-center">
          <div className="text-blue-400 font-medium mb-2 uppercase tracking-wider text-sm">Avg Similarity</div>
          <div className="text-5xl font-black text-blue-400">{(stats.average_similarity * 100).toFixed(1)}%</div>
        </div>
      </div>

      <h3 className="text-lg font-bold mb-4 text-slate-300 px-2 border-b border-slate-700 pb-2">Grade Distribution</h3>
      <div className="space-y-4 px-2">
        <DistributionBar label="Correct" count={stats.distribution["correct"] || 0} total={stats.total_submissions} color="bg-green-500" />
        <DistributionBar label="Partially Correct" count={stats.distribution["partially correct"] || 0} total={stats.total_submissions} color="bg-yellow-500" />
        <DistributionBar label="Incorrect" count={stats.distribution["incorrect"] || 0} total={stats.total_submissions} color="bg-red-500" />
      </div>
    </div>
  );
}

function DistributionBar({ label, count, total, color }) {
  const percentage = total === 0 ? 0 : (count / total) * 100;
  
  return (
    <div>
      <div className="flex justify-between text-sm font-medium mb-1.5">
        <span className="text-slate-300">{label} ({count})</span>
        <span className="text-slate-400">{percentage.toFixed(1)}%</span>
      </div>
      <div className="w-full bg-slate-900 rounded-full h-3 overflow-hidden border border-slate-700">
        <div className={cn("h-3 rounded-full transition-all duration-1000", color)} style={{ width: `${percentage}%` }}></div>
      </div>
    </div>
  );
}
