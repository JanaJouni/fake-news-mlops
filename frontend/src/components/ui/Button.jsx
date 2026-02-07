export function Button({ children, onClick, disabled }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="w-full bg-gradient-to-r from-indigo-600 to-purple-600 text-white font-semibold py-2 px-4 rounded-lg hover:opacity-90 transition disabled:opacity-50"
    >
      {children}
    </button>
  )
}
