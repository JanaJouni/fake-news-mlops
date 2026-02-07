export function Card({ children }) {
  return (
    <div className="bg-white/80 backdrop-blur-lg rounded-xl shadow-lg p-6 border border-gray-200">
      {children}
    </div>
  )
}
