import { ChatIcon } from "../icons";
import { useChatSidebar } from "../context/ChatSidebarContext";

const ChatSidebarToggle: React.FC = () => {
    const { isChatOpen, toggleChat } = useChatSidebar();
  return (
    <>
      {!isChatOpen && (
        <div className="fixed bottom-6 right-6 z-50 pulsing-button-container">
          <button
            onClick={toggleChat}
            className="p-4 rounded-full bg-red-800 text-white shadow-lg hover:bg-red-900 focus:outline-none focus:ring-2 focus:ring-red-400 relative z-10" // Add relative z-10 to keep button above pseudo-element
            aria-label={isChatOpen ? "Close Chat Sidebar" : "Open Chat Sidebar"}
          >
            <ChatIcon className="w-6 h-6" />
          </button>
          <style>{`
            .pulsing-button-container {
              position: fixed;
              bottom: 1.5rem; /* Equivalent to bottom-6 */
              right: 1.5rem; /* Equivalent to right-6 */
              z-index: 50;
              display: flex;
              justify-content: center;
              align-items: center;
            }

            .pulsing-button-container::before {
              content: '';
              position: absolute;
              top: 0;
              left: 0;
              right: 0;
              bottom: 0;
              border-radius: 50%;
              background-color: rgba(239, 68, 68, 0.7); /* Red with some transparency */
              animation: pulse 2s infinite;
              z-index: 0;
            }

            @keyframes pulse {
              0% {
                transform: scale(1);
                opacity: 0.7;
              }
              50% {
                transform: scale(1.5);
                opacity: 0.3;
              }
              100% {
                transform: scale(1.8);
                opacity: 0;
              }
            }
          `}</style>
        </div>
      )}
    </>
  );
};

export default ChatSidebarToggle;
