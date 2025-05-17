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
  const handleSendMessage = (event: React.MouseEvent<HTMLButtonElement> | React.KeyboardEvent<HTMLInputElement>) => {
    event.preventDefault(); // Prevent default form submission and page reload
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
      const chatbotUrl = "http://localhost:3334/api/chatbot/";
      const inventoryUrl = "http://localhost:3334/inventory";

      try {
        console.log("Fetching inventory data...");
        // Fetch inventory data
        const inventoryResponse = await fetch(inventoryUrl);
        if (!inventoryResponse.ok) {
          throw new Error(`HTTP error fetching inventory! status: ${inventoryResponse.status}`);
        }
        const inventoryData = await inventoryResponse.json();
        console.log("Inventory data fetched:", inventoryData);

        // Format inventory data for the payload
        const formattedInventory: { [key: string]: number } = {};
        inventoryData.forEach((item: { productID: string; CurrentStock: number }) => {
          formattedInventory[item.productID] = item.CurrentStock;
        });
        console.log("Formatted inventory:", formattedInventory);

        console.log("Fetching prediction data...");
        // Fetch prediction data from /api/predict
        const predictResponse = await fetch('http://localhost:3334/api/predict', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ date: new Date().toISOString().split('T')[0] }), // Send current date
        });

        if (!predictResponse.ok) {
          throw new Error(`HTTP error fetching predictions! status: ${predictResponse.status}`);
        }

        const predictionData = await predictResponse.json();
        console.log("Prediction data fetched:", predictionData);


        if (predictionData.status !== 'success' || !predictionData.predictions) {
           throw new Error(predictionData.message || 'Failed to get prediction data.');
        }

        // Process prediction data: change negative Forecasted to 0, NaN Weeks to 0, and ensure keys match Pydantic model
        const processedPredictions = predictionData.predictions.map((p: any) => ({
            productID: p.ProductID, // Ensure lowercase 'd'
            days: p.Days,           // Ensure lowercase 'd'
            forecasted: Math.max(0, p.Forecasted), // Ensure lowercase 'f'
            current_stock: p["Current Stock"],
            stock_status: p["Stock Status"],
            weeks: isNaN(p.Weeks) ? 0 : p.Weeks,   // Ensure lowercase 'w'
        }));
        console.log("Processed predictions for InsertPred:", processedPredictions);

        // Filter for Day 1 and Day 3 predictions
        const filteredPredictions = processedPredictions.filter(
            (p: any) => p.days === 1 || p.days === 3
        );
        console.log("Filtered predictions (Day 1 & 3):", filteredPredictions);

        // Restructure filtered predictions into a dictionary by ProductID
        const lstmPayload: { [key: string]: any[] } = {};
        filteredPredictions.forEach((prediction: any) => {
            if (!lstmPayload[prediction.productID]) { // Use productID (lowercase)
                lstmPayload[prediction.productID] = [];
            }
            lstmPayload[prediction.productID].push(prediction);
        });
        console.log("LSTM payload for chatbot (Day 1 & 3 only):", lstmPayload);

        // Send processed predictions to /InsertPred endpoint
        console.log("Calling /InsertPred endpoint...");
        const insertPredUrl = "http://localhost:3334/InsertPred";
        try {
          const insertResponse = await fetch(insertPredUrl, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify(processedPredictions), // Send the array of processed predictions
          });
          console.log("/InsertPred API response status:", insertResponse.status);
          if (!insertResponse.ok) {
            const errorText = await insertResponse.text();
            console.error("/InsertPred API error response text:", errorText);
            // Decide if this error should stop the chatbot call or just be logged
            // For now, just log it and continue to chatbot
          } else {
            const insertResult = await insertResponse.json();
            console.log("/InsertPred API response data:", insertResult);
          }
        } catch (insertError: any) {
          console.error("Error calling /InsertPred API:", insertError);
          // Decide if this error should stop the chatbot call or just be logged
        }


        const payload = {
          prompt: inputValue.trim(),
          inventory: formattedInventory,
          lstm: lstmPayload, // Use the restructured dictionary
          date: new Date().toISOString().split('T')[0] // Use current date
        };
        console.log("Final payload for chatbot API:", payload);
        console.log("Type of lstm in payload:", typeof payload.lstm);
        console.log("Is lstm in payload undefined?:", payload.lstm === undefined);


        // Call API for bot response
        console.log("Calling chatbot API...");
        let response;
        try {
          const requestBody = JSON.stringify(payload);
          console.log("Request body for chatbot API:", requestBody);
          response = await fetch(chatbotUrl, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: requestBody,
          });
          console.log("Chatbot API response status:", response.status);
        } catch (networkError: any) {
          console.error("Network error calling chatbot API:", networkError);
          throw new Error(`Network error calling chatbot API: ${networkError.message}`);
        }
        

        if (!response.ok) {
          const errorText = await response.text();
          console.error("Chatbot API error response text:", errorText);
          throw new Error(`HTTP error! status: ${response.status} - ${errorText}`);
        }

        const data = await response.json();
        console.log("Chatbot API response data:", data);


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

      } catch (error: any) { // Catch any error in the async function
        console.error("An error occurred in fetchBotResponse:", error);
        const errorReply: Message = {
          id: Date.now() + 1,
          from: "bot",
          text: `Sorry, an error occurred: ${error.message || 'Unknown error'}`,
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
      handleSendMessage(e); // Pass the event object
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
                  className={`p-3 rounded-lg break-words max-w-full w-fit ${ // Changed max-w-xl to max-w-full
                    from === "user"
                      ? "bg-gray-300 dark:bg-gray-800 text-gray-900 dark:text-gray-100 rounded-br-none"
                      : "bg-red-200 dark:bg-red-700 text-gray-900 dark:text-gray-100 rounded-bl-none"
                  }`}
                >
                {from === "bot" ? <div className="prose dark:prose-invert max-w-none"><ReactMarkdown>{text}</ReactMarkdown></div> : text} 
                </div>
                {from === "bot" && showAnalysisButton && (
                  <Button
                    size="sm"
                    variant="primary"
                    className="mt-2"
                    onClick={() => navigate('/')} // Add onClick handler
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
            onClick={(event) => handleSendMessage(event)} // Pass the event object
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
