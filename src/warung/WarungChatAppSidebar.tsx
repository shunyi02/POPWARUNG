import { useEffect, useRef, useState } from "react";
import { HorizontaLDots } from "../icons";
import { TrashBinIcon } from "../icons"; // Import Trash icon

type Message = {
  id: number;
  from: "user" | "bot";
  text: string;
};

import { useChatSidebar } from "../context/ChatSidebarContext"; // Import useChatSidebar

const ChatSidebar: React.FC = () => {
  const { isChatOpen, closeChat } = useChatSidebar(); // Use useChatSidebar and destructure closeChat
  
  // Chat state
  const [messages, setMessages] = useState<Message[]>([
    { id: 1, from: "bot", text: "Hello! How can I help you today?" },
  ]);
  const [inputValue, setInputValue] = useState("");
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  // Scroll chat to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Handle sending a message
  const handleSendMessage = () => {
    if (!inputValue.trim()) return;

    const newUserMessage: Message = {
      id: Date.now(),
      from: "user",
      text: inputValue.trim(),
    };

    setMessages((prev) => [...prev, newUserMessage]);
    setInputValue("");

    // Dummy bot response after delay
    setTimeout(() => {
      const botReply: Message = {
        id: Date.now() + 1,
        from: "bot",
        text: "Thanks for your message! I'm here to help.",
      };
      setMessages((prev) => [...prev, botReply]);
    }, 1000);
  };

  // Handle enter key for sending message
  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      e.preventDefault();
      handleSendMessage();
    }
  };

  // Handle clearing messages
  const handleClearMessages = () => {
    setMessages([
      { id: 1, from: "bot", text: "Hello! How can I help you today?" },
    ]);
  };

  // Determine visibility based on isChatOpen state
  const visible = isChatOpen;

  return (
    <aside
      className={`flex flex-col bg-white dark:bg-gray-900 border-l border-gray-200 dark:border-gray-800
        transition-all duration-300 ease-in-out fixed right-0 top-0 z-40 ${isChatOpen ? "w-[400px]" : "w-0 overflow-hidden"}`}
      style={{ height: '100vh', margin: 0, padding: 0 }}
    >

      {/* Header */}
      <div
        className="flex items-center justify-between border-b border-gray-200 dark:border-gray-700 p-4 sticky top-0 bg-white dark:bg-gray-900 h-19" // Added sticky top-0 and background for sticky
        
      >
        {(visible) ? (
          <>
            <h2 className="text-lg font-semibold text-gray-700 dark:text-gray-200">
              Chatbot
            </h2>
            <div className="flex items-center gap-2"> {/* Container for buttons */}
              {/* Clear messages button */}
              <button
                onClick={handleClearMessages}
                aria-label="Clear messages"
                className="text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 focus:outline-none"
              >
                <TrashBinIcon className="w-5 h-5" /> {/* Bin icon */}
              </button>
              {/* Close button */}
              <button
                onClick={closeChat}
                aria-label="Close chat sidebar"
                className="text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 focus:outline-none"
                style={{ fontSize: 20, fontWeight: "bold", lineHeight: 1 }}
              >
                &times;
              </button>
            </div>
          </>
        ) : (
          <HorizontaLDots className="w-6 h-6 text-gray-400" />
        )}
      </div>

      {/* Messages Area */}
      {visible && (
        <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3 bg-gray-50 dark:bg-gray-800">
          {messages.map(({ id, from, text }) => (
            <div
              key={id}
              className={`flex ${from === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[80%] p-3 rounded-lg whitespace-pre-wrap break-words ${
                  from === "user"
                    ? "bg-gray-300 dark:bg-gray-800 text-gray-900 dark:text-gray-100 rounded-br-none" // User message is very light grey, text is black/dark mode equivalent
                    : "bg-red-200 dark:bg-red-700 text-gray-900 dark:text-gray-100 rounded-bl-none" // Bot message is light red with black/dark mode equivalent text
                }`}
              >
                {text}
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
      )}

      {/* Input Area */}
      {visible && (
        <div className="border-t border-gray-200 dark:border-gray-700 px-4 py-3 flex items-center gap-2 bg-white dark:bg-gray-900">
          <input
            type="text"
            placeholder="Type a message..."
            className="flex-1 rounded-md border border-red-300 dark:border-red-600 bg-white dark:bg-gray-700 px-3 py-2 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-red-300" // Input focus ring is light red, border is light red
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
          />
          <button
            onClick={handleSendMessage}
            className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-md transition" // Send button is red
            aria-label="Send message"
          >
            Send
          </button>
        </div>
      )}
    </aside>
  );
};

export default ChatSidebar;
