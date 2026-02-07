import { useState } from "react";
import "./App.css";

function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const predict = async () => {
    setLoading(true);
    setResult(null);

    const res = await fetch("http://localhost:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    const data = await res.json();
    setResult(data);
    setLoading(false);
  };

  return (
    <div className="bg">
      <div className="glass-card">
        <h1 className="title">
          <span>FAKE</span> NEWS DETECTOR
        </h1>

        <p className="subtitle">
          Trust the news. Let AI detect misinformation.
        </p>

        <textarea
          placeholder="Paste a news article, tweet, or headline..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        />

        <button onClick={predict} disabled={!text || loading}>
          {loading ? "Scanning Intelligence..." : "Run AI Analysis"}
        </button>

        {result && (
          <div className="result-box">
            <div
              className={`label ${
                result.label === "FAKE" ? "fake" : "real"
              }`}
            >
              {result.label}
            </div>

            <div className="confidence">
              <div
                className="confidence-bar"
                style={{ width: result.confidence }}
              ></div>
            </div>

            <p className="confidence-text">
              Confidence Level: <strong>{result.confidence}</strong>
            </p>
          </div>
        )}
      </div>

      <footer>
        ⚙️ FastAPI · MLflow · DVC · Docker · React
      </footer>
    </div>
  );
}

export default App;
