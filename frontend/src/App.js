import { useState, useEffect } from "react"
import { Card } from "./components/ui/Card"
import { Button } from "./components/ui/Button"
import { Input } from "./components/ui/Input"

function App() {
  const [text, setText] = useState("")
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [history, setHistory] = useState([])

  const BACKEND_URL = "http://127.0.0.1:8000"

  // Fetch history on mount
  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const res = await fetch(`${BACKEND_URL}/history?limit=20`)
        if (!res.ok) throw new Error("Failed to fetch history")
        const data = await res.json()
        setHistory(
          data.map((item) => ({
            text: item.text,
            label: item.label,
            confidence: item.confidence,
          }))
        )
      } catch (err) {
        console.error(err)
      }
    }
    fetchHistory()
  }, [])

  // Predict
  const handlePredict = async () => {
    if (!text.trim()) return
    setLoading(true)
    setResult(null)
    try {
      const res = await fetch(`${BACKEND_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      })
      if (!res.ok) throw new Error("Prediction failed")
      const data = await res.json()
      setResult(data)
      setHistory((prev) => [
        {
          text,
          label: data.label,
          confidence: data.confidence,
        },
        ...prev,
      ])
      setText("")
    } catch (err) {
      alert("Backend not reachable")
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  // Delete a history item
  const handleDelete = async (textToDelete) => {
    try {
      const res = await fetch(`${BACKEND_URL}/history?text=${encodeURIComponent(textToDelete)}`, {
        method: "DELETE",
      })
      if (!res.ok) throw new Error("Failed to delete history item")

      // Remove from frontend state
      setHistory((prev) => prev.filter((item) => item.text !== textToDelete))
    } catch (err) {
      alert("Failed to delete history item")
      console.error(err)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 to-slate-800 text-white flex">
      
      {/* LEFT — MAIN APP */}
      <div className="flex-1 flex items-center justify-center p-6">
        <Card className="w-full max-w-xl">
          <h1 className="text-2xl font-bold mb-4 text-center">
            📰 Fake News Detector
          </h1>

          <Input
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste a news article here..."
          />

          <Button
            onClick={handlePredict}
            disabled={loading}
            className="mt-4 w-full"
          >
            {loading ? "Analyzing..." : "Predict"}
          </Button>

          {result && (
            <div className="mt-4 p-4 rounded-lg bg-slate-800 border border-slate-600">
              <p className="text-lg">
                Result:{" "}
                <span
                  className={
                    result.label === "FAKE"
                      ? "text-red-500 text-xl font-extrabold"
                      : "text-green-500 text-xl font-extrabold"
                  }
                >
                  {result.label}
                </span>
              </p>
              <p className="text-gray-300 mt-1">
                Confidence: {(result.confidence * 100).toFixed(2)}%
              </p>
            </div>
          )}
        </Card>
      </div>

      {/* RIGHT — HISTORY */}
      <div className="w-80 bg-slate-900 border-l border-slate-700 p-4 overflow-y-auto">
        <h2 className="text-lg font-semibold mb-3">History</h2>

        {history.length === 0 && (
          <p className="text-gray-400 text-sm">No predictions yet</p>
        )}

        {history.map((item, idx) => (
          <div
            key={idx}
            className="mb-3 p-3 rounded-lg bg-slate-800 border border-slate-700 text-sm relative"
          >
            <p className="truncate text-gray-200">{item.text}</p>
            <p
              className={
                item.label === "FAKE"
                  ? "text-red-400 font-bold mt-1"
                  : "text-green-400 font-bold mt-1"
              }
            >
              {item.label} — {(item.confidence * 100).toFixed(1)}%
            </p>

            <Button
              onClick={() => handleDelete(item.text)}
              className="absolute top-2 right-2 bg-red-600 text-white text-xs px-2 py-1 rounded hover:bg-red-700"
            >
              Delete
            </Button>
          </div>
        ))}
      </div>
    </div>
  )
}

export default App
