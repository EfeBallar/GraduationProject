import { useState } from "react";

function InputBox({ onSendMessage, loading }) {
  const [input, setInput] = useState("");

  function handleSend() {
    if (input.trim()) {
      onSendMessage(input);
      setInput("");
    }
  }

  function handleKeyDown(event) {
    // console.log(event.key)
    if (event.key === "Enter") {
      handleSend();
    }
  }

  return (
    <div className="flex justify-center p-4 min-h-20">
      <div className="flex align-center w-1/2 bg-gray-300 rounded-full px-5 py-1">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Type here"
          className="flex-1 border-none outline-none bg-transparent accent-pink-400"
        />
        <button
          onClick={() => handleSend()}
          className={`rounded-full bg-gray-200 p-3 flex items-center justify-center ${
            loading ? "cursor-pointer" : ""
          }`}
          disabled={loading}
        >
          {loading ? (
            <span className="loading loading-spinner loading-sm"></span>
          ) : (
            <img src="/arrow.png" alt="" className="size-4" />
          )}
        </button>
      </div>
    </div>
  );
}

export default InputBox;
