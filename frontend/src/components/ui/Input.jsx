export function Input({ value, onChange, placeholder }) {
  return (
    <textarea
      value={value}
      onChange={onChange}
      placeholder={placeholder}
      rows={6}
      className="
        w-full
        p-3
        rounded-lg
        bg-slate-800
        text-white
        placeholder-gray-400
        border
        border-slate-600
        focus:outline-none
        focus:ring-2
        focus:ring-indigo-500
      "
    />
  )
}
