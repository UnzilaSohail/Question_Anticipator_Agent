import React, { useState } from 'react';
import { AlertCircle, Brain, CheckCircle, Loader2, Database, Zap, TrendingUp } from 'lucide-react';

/* ---------- TYPES ---------- */
interface PastPaper { year: string; questions: string[]; }
interface ExamPattern { mcqs: number; short_questions: number; long_questions: number; }
interface Weightage { [unit: string]: number; }
interface InputData {
  syllabus: string[];
  past_papers: PastPaper[];
  exam_pattern: ExamPattern;
  weightage: Weightage;
  difficulty_preference: string;
}
interface PredictedQuestion {
  topic: string;
  question_text: string;
  difficulty_level: string;
  question_type: string;
  probability_score: number;
    options?: string[];
}
interface AgentOutput {
  predicted_questions: PredictedQuestion[];
  from_memory?: boolean;
  processing_time?: number;
  memory_hash?: string;
  cached_timestamp?: string;
}

/* ---------- COMPONENT ---------- */
const QuestionAnticipatorAgent: React.FC = () => {
  const [inputData, setInputData] = useState<InputData>({
    syllabus: [],
    past_papers: [],
    exam_pattern: { mcqs: 0, short_questions: 0, long_questions: 0 },
    weightage: {},
    difficulty_preference: 'medium'
  });

  const [syllabusInput, setSyllabusInput] = useState('');
  const [paperYear, setPaperYear] = useState('');
  const [paperQuestions, setPaperQuestions] = useState('');
  const [unitName, setUnitName] = useState('');
  const [unitWeight, setUnitWeight] = useState('');

  const [output, setOutput] = useState<AgentOutput | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [demoMode, setDemoMode] = useState(true);

  /* ---------- LOGIC ---------- */
  const addSyllabusTopic = () => {
    if (syllabusInput.trim()) {
      setInputData(prev => ({ ...prev, syllabus: [...prev.syllabus, syllabusInput.trim()] }));
      setSyllabusInput('');
    }
  };
  const removeSyllabusTopic = (i: number) => setInputData(prev => ({ ...prev, syllabus: prev.syllabus.filter((_, idx) => idx !== i) }));

  const addPastPaper = () => {
    if (paperYear && paperQuestions) {
      setInputData(prev => ({ ...prev, past_papers: [...prev.past_papers, { year: paperYear, questions: paperQuestions.split('\n').filter(q => q.trim()) }] }));
      setPaperYear(''); setPaperQuestions('');
    }
  };
  const removePastPaper = (i: number) => setInputData(prev => ({ ...prev, past_papers: prev.past_papers.filter((_, idx) => idx !== i) }));

  const addWeightage = () => {
    if (unitName && unitWeight) {
      setInputData(prev => ({ ...prev, weightage: { ...prev.weightage, [unitName]: parseInt(unitWeight) } }));
      setUnitName(''); setUnitWeight('');
    }
  };
  const removeWeightage = (unit: string) => setInputData(prev => { const w = { ...prev.weightage }; delete w[unit]; return { ...prev, weightage: w }; });

  const loadSampleData = () => setInputData({
    syllabus: ['Machine Learning Basics', 'Neural Networks', 'Deep Learning', 'Natural Language Processing', 'Computer Vision'],
    past_papers: [
      { year: '2023', questions: ['Explain supervised learning', 'What is backpropagation?', 'Describe CNNs', 'Compare RNN and LSTM'] },
      { year: '2022', questions: ['Define ML and its types', 'Explain activation functions', 'What is transfer learning?', 'Describe attention mechanism'] }
    ],
    exam_pattern: { mcqs: 10, short_questions: 5, long_questions: 3 },
    weightage: { 'Machine Learning': 30, 'Neural Networks': 25, 'Deep Learning': 25, 'NLP': 10, 'Computer Vision': 10 },
    difficulty_preference: 'medium'
  });

  const handleSubmit = async () => {
    setLoading(true); setError('');
    try {
      if (!demoMode) {
        const res = await fetch('http://localhost:8000/api/predict-questions', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(inputData) });
        if (!res.ok) throw new Error(`API Error: ${res.statusText}`);
        const result: AgentOutput = await res.json();
        setOutput(result);
      } else {
        await new Promise(r => setTimeout(r, 2000));
        setOutput({
          predicted_questions: [
            { topic: 'Machine Learning Basics', question_text: 'Explain the difference between supervised and unsupervised learning with real-world examples.', difficulty_level: 'medium', question_type: 'long_question', probability_score: 0.92 },
            { topic: 'Neural Networks', question_text: 'What is the role of backpropagation in training neural networks?', difficulty_level: 'medium', question_type: 'short_question', probability_score: 0.88 },
            { topic: 'Deep Learning', question_text: 'Which activation function is commonly used in hidden layers of deep neural networks?', difficulty_level: 'easy', question_type: 'mcq', probability_score: 0.85 },
            { topic: 'Machine Learning Basics', question_text: 'Define overfitting and explain methods to prevent it.', difficulty_level: 'medium', question_type: 'short_question', probability_score: 0.87 },
            { topic: 'Neural Networks', question_text: 'What is the vanishing gradient problem?', difficulty_level: 'medium', question_type: 'mcq', probability_score: 0.83 },
            { topic: 'Deep Learning', question_text: 'Describe the architecture and applications of Convolutional Neural Networks (CNNs).', difficulty_level: 'hard', question_type: 'long_question', probability_score: 0.81 },
            { topic: 'Machine Learning Basics', question_text: 'What is the purpose of cross-validation?', difficulty_level: 'easy', question_type: 'mcq', probability_score: 0.79 },
            { topic: 'Neural Networks', question_text: 'Explain the concept of dropout in neural networks.', difficulty_level: 'medium', question_type: 'short_question', probability_score: 0.86 },
            { topic: 'Deep Learning', question_text: 'Compare and contrast RNN and LSTM architectures with their use cases.', difficulty_level: 'hard', question_type: 'long_question', probability_score: 0.78 }
          ],
          from_memory: false, processing_time: 2.4, memory_hash: 'a3f5d91c'
        });
      }
    } catch (err: any) { setError(err?.message || 'Unknown error'); } finally { setLoading(false); }
  };

  /* ---------- UI ---------- */
  return (
    <>
      <style>{`@keyframes spin{to{transform:rotate(360deg)}} .spin{animation:spin 1s linear infinite}`}</style>
      <div style={{ minHeight: '100vh', background: 'linear-gradient(135deg,#eef2ff 0%,#e0e7ff 50%,#ede9fe 100%)', padding: 24 }}>
        <div style={{ maxWidth: 1280, margin: '0 auto' }}>
          {/* header */}
          <div style={{ background: '#fff', borderRadius: 16, boxShadow: '0 10px 25px rgba(0,0,0,.1)', padding: 32, marginBottom: 24 }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <Brain size={40} color="#4f46e5" />
                <div>
                  <h1 style={{ fontSize: 32, fontWeight: 700, color: '#111' }}>Question Anticipator Agent</h1>
                  <p style={{ color: '#6b7280' }}>AI-powered exam question prediction with long-term memory</p>
                </div>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 14 }}>
                  <input type="checkbox" checked={demoMode} onChange={e => setDemoMode(e.target.checked)} />
                  Demo Mode
                </label>
                <button onClick={loadSampleData} style={{ padding: '6px 12px', background: '#f3f4f6', borderRadius: 8, border: 'none', cursor: 'pointer' }}>Load Sample Data</button>
              </div>
            </div>
            {demoMode && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: 12, background: '#eff6ff', border: '1px solid #bfdbfe', borderRadius: 8, fontSize: 13, color: '#1d4ed8' }}>
                <Zap size={16} /> Demo mode enabled – using simulated responses. Disable to connect to API.
              </div>
            )}
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 24 }}>
            {/* ---------- INPUT SECTION ---------- */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
              {/* Syllabus */}
              <div style={{ background: '#fff', borderRadius: 12, boxShadow: '0 4px 12px rgba(0,0,0,.08)', padding: 24 }}>
                <label style={{ display: 'block', fontWeight: 600, marginBottom: 12 }}>📚 Syllabus Topics</label>
                <div style={{ display: 'flex', gap: 8 }}>
                  <input
                    value={syllabusInput}
                    onChange={e => setSyllabusInput(e.target.value)}
                    onKeyDown={e => { if (e.key === 'Enter') addSyllabusTopic(); }}
                    placeholder="Enter topic and press Enter"
                    style={{ flex: 1, padding: '8px 12px', border: '1px solid #d1d5db', borderRadius: 8 }}
                  />
                  <button onClick={addSyllabusTopic} style={{ padding: '8px 16px', background: '#4f46e5', color: '#fff', border: 'none', borderRadius: 8, cursor: 'pointer' }}>Add</button>
                </div>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 12 }}>
                  {inputData.syllabus.map((t, i) => (
                    <span key={i} style={{ padding: '4px 12px', background: '#e0e7ff', color: '#3730a3', borderRadius: 16, fontSize: 13 }}>
                      {t}
                      <button onClick={() => removeSyllabusTopic(i)} style={{ marginLeft: 6, cursor: 'pointer', background: 'none', border: 'none', color: '#3730a3' }}>×</button>
                    </span>
                  ))}
                </div>
              </div>

              {/* Past Papers */}
              <div style={{ background: '#fff', borderRadius: 12, boxShadow: '0 4px 12px rgba(0,0,0,.08)', padding: 24 }}>
                <label style={{ display: 'block', fontWeight: 600, marginBottom: 12 }}>📝 Past Papers</label>
                <input
                  value={paperYear}
                  onChange={e => setPaperYear(e.target.value)}
                  placeholder="Year (e.g., 2023)"
                  style={{ width: '100%', padding: 8, border: '1px solid #d1d5db', borderRadius: 8, marginBottom: 8 }}
                />
                <textarea
                  value={paperQuestions}
                  onChange={e => setPaperQuestions(e.target.value)}
                  placeholder="Questions (one per line)"
                  rows={3}
                  style={{ width: '100%', padding: 8, border: '1px solid #d1d5db', borderRadius: 8, marginBottom: 8 }}
                />
                <button onClick={addPastPaper} style={{ width: '100%', padding: '8px 16px', background: '#4f46e5', color: '#fff', border: 'none', borderRadius: 8, cursor: 'pointer' }}>Add Past Paper</button>
                <div style={{ marginTop: 12, display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {inputData.past_papers.map((p, i) => (
                    <div key={i} style={{ display: 'flex', justifyContent: 'space-between', padding: 8, background: '#f9fafb', borderRadius: 8, fontSize: 13 }}>
                      <span>{p.year} – {p.questions.length} questions</span>
                      <button onClick={() => removePastPaper(i)} style={{ color: '#dc2626', background: 'none', border: 'none', cursor: 'pointer' }}>Remove</button>
                    </div>
                  ))}
                </div>
              </div>

              {/* Exam Pattern */}
              <div style={{ background: '#fff', borderRadius: 12, boxShadow: '0 4px 12px rgba(0,0,0,.08)', padding: 24 }}>
                <label style={{ display: 'block', fontWeight: 600, marginBottom: 12 }}>📊 Exam Pattern</label>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 }}>
                  {(['mcqs', 'short_questions', 'long_questions'] as const).map(type => (
                    <div key={type}>
                      <label style={{ fontSize: 12, color: '#6b7280' }}>{type.replace('_', ' ')}</label>
                      <input
                        type="number"
                        value={inputData.exam_pattern[type] || ''}
                        onChange={e => setInputData(prev => ({ ...prev, exam_pattern: { ...prev.exam_pattern, [type]: parseInt(e.target.value) || 0 } }))}
                        style={{ width: '100%', padding: 8, border: '1px solid #d1d5db', borderRadius: 8 }}
                      />
                    </div>
                  ))}
                </div>
              </div>

              {/* Weightage */}
              <div style={{ background: '#fff', borderRadius: 12, boxShadow: '0 4px 12px rgba(0,0,0,.08)', padding: 24 }}>
                <label style={{ display: 'block', fontWeight: 600, marginBottom: 12 }}>⚖️ Weightage by Unit</label>
                <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
                  <input
                    value={unitName}
                    onChange={e => setUnitName(e.target.value)}
                    placeholder="Unit name"
                    style={{ flex: 1, padding: 8, border: '1px solid #d1d5db', borderRadius: 8 }}
                  />
                  <input
                    value={unitWeight}
                    onChange={e => setUnitWeight(e.target.value)}
                    placeholder="Weight %"
                    type="number"
                    style={{ width: 80, padding: 8, border: '1px solid #d1d5db', borderRadius: 8 }}
                  />
                  <button onClick={addWeightage} style={{ padding: '8px 16px', background: '#4f46e5', color: '#fff', border: 'none', borderRadius: 8, cursor: 'pointer' }}>Add</button>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {Object.entries(inputData.weightage).map(([unit, w]) => (
                    <div key={unit} style={{ display: 'flex', justifyContent: 'space-between', padding: 8, background: '#f9fafb', borderRadius: 8 }}>
                      <span style={{ fontSize: 14 }}>{unit}</span>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span style={{ fontWeight: 600, color: '#4f46e5' }}>{w}%</span>
                        <button onClick={() => removeWeightage(unit)} style={{ color: '#dc2626', background: 'none', border: 'none', cursor: 'pointer' }}>×</button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Difficulty */}
              <div style={{ background: '#fff', borderRadius: 12, boxShadow: '0 4px 12px rgba(0,0,0,.08)', padding: 24 }}>
                <label style={{ display: 'block', fontWeight: 600, marginBottom: 12 }}>🎯 Difficulty Preference</label>
                <select
                  value={inputData.difficulty_preference}
                  onChange={e => setInputData(prev => ({ ...prev, difficulty_preference: e.target.value }))}
                  style={{ width: '100%', padding: 8, border: '1px solid #d1d5db', borderRadius: 8 }}
                >
                  <option value="easy">Easy</option>
                  <option value="medium">Medium</option>
                  <option value="hard">Hard</option>
                  <option value="mixed">Mixed</option>
                </select>
              </div>

              {/* Submit */}
              <button
                onClick={handleSubmit}
                disabled={loading || inputData.syllabus.length === 0}
                style={{
                  width: '100%',
                  padding: '16px 24px',
                  background: 'linear-gradient(135deg,#4f46e5 0%,#7c3aed 100%)',
                  color: '#fff',
                  fontWeight: 600,
                  border: 'none',
                  borderRadius: 12,
                  cursor: loading || inputData.syllabus.length === 0 ? 'not-allowed' : 'pointer',
                  opacity: loading || inputData.syllabus.length === 0 ? 0.5 : 1
                }}
              >
                {loading ? (
                  <>
                    <Loader2 size={20} className="spin" style={{ marginRight: 6 }} />
                    Generating...
                  </>
                ) : (
                  <>
                    <Brain size={20} style={{ marginRight: 6 }} />
                    Predict Questions
                  </>
                )}
              </button>

              {error && (
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: 16, background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#b91c1c', fontSize: 14 }}>
                  <AlertCircle size={20} /> {error}
                </div>
              )}
            </div> {/* ---------- OUTPUT SECTION ---------- */}
              <div style={{ width: '700px',      // fixed width
      height: '100%',     // fixed height
      overflowY: 'auto', }}>
              <div style={{ background: '#fff', borderRadius: 12, boxShadow: '0 4px 12px rgba(0,0,0,.08)', padding: 24 }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
                  <h2 style={{ fontSize: 20, fontWeight: 700 }}>Predicted Questions</h2>
                  {output && (
                    <div style={{ fontSize: 13 }}>
                      {output.from_memory ? (
                        <span style={{ color: '#7c3aed' }}><Database size={16} style={{ verticalAlign: 'middle', marginRight: 4 }} />From Memory</span>
                      ) : (
                        <span style={{ color: '#16a34a' }}><Zap size={16} style={{ verticalAlign: 'middle', marginRight: 4 }} />Newly Generated</span>
                      )}
                    </div>
                  )}
                </div>

                {!output ? (
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: 384, color: '#9ca3af' }}>
                    <Brain size={80} style={{ marginBottom: 16 }} />
                    <p style={{ fontSize: 18, fontWeight: 500 }}>No predictions yet</p>
                    <p style={{ fontSize: 14 }}>Fill in the form and click Predict Questions</p>
                  </div>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 16, maxHeight: 800, overflowY: 'auto', paddingRight: 8 }}>
                    {output.predicted_questions.map((q, i) => (
                      <div key={i} style={{ border: '1px solid #e5e7eb', borderRadius: 12, padding: 16, background: '#fff' }}>
                        <div style={{ display: 'flex', alignItems: 'start', justifyContent: 'space-between', marginBottom: 8 }}>
                          <span style={{ fontSize: 12, fontWeight: 600, color: '#4f46e5', display: 'flex', alignItems: 'center', gap: 4 }}><TrendingUp size={12} /> {q.topic}</span>
                          <span style={{ fontSize: 14, fontWeight: 700, color: q.probability_score >= 0.85 ? '#16a34a' : q.probability_score >= 0.7 ? '#eab308' : '#ea580c' }}>{(q.probability_score * 100).toFixed(0)}%</span>
                        </div>

                        <p style={{ marginBottom: 12, fontWeight: 500, lineHeight: 1.5, whiteSpace: 'pre-line' }}>{q.question_text}</p>

                        {/* MCQ options */}
                        {q.question_type === 'mcq' && q.options && (
                          <ul style={{ marginBottom: 12, paddingLeft: 20 }}>
                            {q.options.map((opt, idx) => <li key={idx}>{opt}</li>)}
                          </ul>
                        )}

                        <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
                          <span style={{ padding: '4px 12px', borderRadius: 16, fontSize: 12, fontWeight: 600, border: '1px solid', background: q.difficulty_level === 'easy' ? '#dcfce7' : q.difficulty_level === 'medium' ? '#fef9c3' : '#fee2e2', color: q.difficulty_level === 'easy' ? '#166534' : q.difficulty_level === 'medium' ? '#854d0e' : '#b91c1c' }}>{q.difficulty_level}</span>
                          <span style={{ padding: '4px 12px', borderRadius: 16, fontSize: 12, fontWeight: 600, border: '1px solid', background: '#dbeafe', color: '#1d4ed8' }}>{q.question_type.replace('_', ' ')}</span>
                        </div>

                        <div style={{ width: '100%', background: '#e5e7eb', borderRadius: 999, height: 8 }}>
                          <div style={{ height: 8, borderRadius: 999, background: q.probability_score >= 0.85 ? '#16a34a' : q.probability_score >= 0.7 ? '#eab308' : '#ea580c', width: `${q.probability_score * 100}%` }} />
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {output && (
                <div style={{ background: '#fff', borderRadius: 12, boxShadow: '0 4px 12px rgba(0,0,0,.08)', padding: 24 }}>
                  <h3 style={{ display: 'flex', alignItems: 'center', gap: 8, fontWeight: 600, marginBottom: 12 }}><CheckCircle size={20} color="#16a34a" /> JSON Output Preview</h3>
                  <pre style={{ background: '#f9fafb', padding: 16, borderRadius: 8, fontSize: 12, border: '1px solid #e5e7eb', maxHeight: 256, overflow: 'auto' }}>
                    {JSON.stringify(output, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default QuestionAnticipatorAgent;