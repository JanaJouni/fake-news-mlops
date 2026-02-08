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


  // --------------------------
  // Load history
  // --------------------------
  useEffect(() => {
    fetch(`${BACKEND_URL}/history?limit=20`)
      .then((res) => res.json())
      .then(setHistory)
      .catch(console.error)
  }, [])

  // --------------------------
  // Predict
  // --------------------------
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

      if (!res.ok) throw new Error()

      const data = await res.json()

      setResult(data)
      setHistory((prev) => [
        {
          id: Date.now(), // temporary until reload
          text,
          label: data.label,
          confidence: data.confidence,
        },
        ...prev,
      ])

      setText("")
    } catch {
      alert("Backend not reachable")
    } finally {
      setLoading(false)
    }
  }

  // --------------------------
  // Delete
  // --------------------------
  const handleDelete = async (id) => {
    try {
      const res = await fetch(`${BACKEND_URL}/history/${id}`, {
        method: "DELETE",
      })
      if (!res.ok) throw new Error()

      setHistory((prev) => prev.filter((item) => item.id !== id))
    } catch {
      alert("Failed to delete item")
    }
  }

  return (
    <div className="min-h-screen bg-slate-900 text-white flex">
      {/* MAIN */}
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
            <div className="mt-4 p-4 bg-slate-800 rounded">
              <p className="text-lg">
                Result:{" "}
                <span
                  className={
                    result.label === "FAKE"
                      ? "text-red-500 font-bold"
                      : "text-green-500 font-bold"
                  }
                >
                  {result.label}
                </span>
              </p>
              <p className="text-gray-300">
                Confidence: {(result.confidence * 100).toFixed(2)}%
              </p>
            </div>
          )}
        </Card>
      </div>

      {/* HISTORY */}
      <div className="w-80 bg-slate-800 p-4 overflow-y-auto">
        <h2 className="font-semibold mb-3">History</h2>

        {history.map((item) => (
          <div key={item.id} className="mb-3 bg-slate-700 p-3 rounded relative">
            <p className="truncate">{item.text}</p>
            <p className="font-bold mt-1">
              {item.label} — {(item.confidence * 100).toFixed(1)}%
            </p>

            <Button
              onClick={() => handleDelete(item.id)}
              className="absolute top-2 right-2 bg-red-600 text-xs"
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
