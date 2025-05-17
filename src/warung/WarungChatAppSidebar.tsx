import { useEffect, useRef, useState } from "react";
import { HorizontaLDots } from "../icons";
import { TrashBinIcon } from "../icons"; // Import Trash icon
import ReactMarkdown from 'react-markdown';
import Button from '../components/ui/button/Button'; // Import the Button component
import { useNavigate } from 'react-router-dom'; // Import useNavigate

type Message = {
  id: number;
  from: "user" | "bot";
  text: string;
  showAnalysisButton?: boolean; // Make the new property optional
};

import { useChatSidebar } from "../context/ChatSidebarContext"; // Import useChatSidebar

const ChatSidebar: React.FC = () => {
  const { isChatOpen, closeChat } = useChatSidebar(); // Use useChatSidebar and destructure closeChat
  const navigate = useNavigate(); // Get the navigate function
  
  // Chat state
  const [messages, setMessages] = useState<Message[]>([
    { id: 1, from: "bot", text: "Hello! How can I help you today?" },
  ]);
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false); // Loading state for API call
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
    setIsLoading(true); // Set loading to true before API call

    // Call API for bot response
    const fetchBotResponse = async () => {
      const url = "http://localhost:3333/api/chatbot/";
      const payload = {
        prompt: inputValue.trim(),
        inventory: {
          "Product A": 10,
          "Product B": 3,
          "Product C": 0
        },
        lstm: {
          "Product A": 30,
          "Product B": 20,
          "Product C": 5
        },
        date: new Date().toISOString().split('T')[0] // Use current date
      };

      try {
        const response = await fetch(url, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(payload),
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();

        let botResponseText = data.response;
        let showAnalysisButton = false;

        // Check for and remove the forecast tag
        if (botResponseText.includes("(##$FORCAST$##)")) {
          botResponseText = botResponseText.replace("(##$FORCAST$##)", "").trim();
          showAnalysisButton = true;
        }

        const botReply: Message = {
          id: Date.now() + 1,
          from: "bot",
          text: botResponseText,
          showAnalysisButton: showAnalysisButton, // Add new property
        };
        setMessages((prev) => [...prev, botReply]);
        setIsLoading(false); // Set loading to false after successful response

      } catch (error) {
        console.error("Error fetching chatbot response:", error);
        const errorReply: Message = {
          id: Date.now() + 1,
          from: "bot",
          text: "Sorry, I couldn't get a response from the chatbot.",
        };
        setMessages((prev) => [...prev, errorReply]);
        setIsLoading(false); // Set loading to false on error
      }
    };

    fetchBotResponse();
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
        transition-all duration-300 ease-in-out fixed right-0 top-0 z-40 ${isChatOpen ? "w-[410px]" : "w-0 overflow-hidden"}`}
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
        <div className="flex-1 overflow-y-auto overflow-x-hidden px-4 py-3 space-y-3 bg-gray-50 dark:bg-gray-800">
          {messages.map(({ id, from, text, showAnalysisButton }) => (
            <div
              key={id}
              className={`flex ${from === "user" ? "justify-end" : "justify-start"}`}
            >
              <div className={`flex flex-col ${from === "user" ? "items-end" : "items-start"}`}>
                <div
                  className={`p-3 rounded-lg break-words max-w-xl w-fit ${
                    from === "user"
                      ? "bg-gray-300 dark:bg-gray-800 text-gray-900 dark:text-gray-100 rounded-br-none"
                      : "bg-red-200 dark:bg-red-700 text-gray-900 dark:text-gray-100 rounded-bl-none"
                  }`}
                >
                {from === "bot" ? <ReactMarkdown>{text}</ReactMarkdown> : text}
                </div>
                {from === "bot" && showAnalysisButton && (
                  <Button
                    size="sm"
                    variant="primary"
                    className="mt-2"
                    onClick={() => navigate('/insight')} // Add onClick handler
                  >
                    In-Depth Analysis
                  </Button>
                )}
              </div> {/* End of new wrapper div */}
            </div>
          ))}
          {isLoading && (
            <div className="flex justify-start">
              <div className="max-w-[90%] p-3 rounded-lg bg-red-200 dark:bg-red-700 text-gray-900 dark:text-gray-100 rounded-bl-none flex items-center space-x-1">
                <span>Typing</span>
                <div className="loading-dots">
                  <span className="dot dot-1">.</span>
                  <span className="dot dot-2">.</span>
                  <span className="dot dot-3">.</span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      )}

      <style>{`
        .loading-dots {
          display: flex;
          align-items: center;
        }
        .dot {
          animation: bounce 1s infinite ease-in-out;
        }
        .dot-1 { animation-delay: -0.32s; }
        .dot-2 { animation-delay: -0.16s; }
        .dot-3 { animation-delay: 0s; }

        @keyframes bounce {
          0%, 80%, 100% {
            transform: translateY(0);
          }
          40% {
            transform: translateY(-4px);
          }
        }
      `}</style>

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
