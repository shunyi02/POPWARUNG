import { ChatIcon } from "../icons";
import { useChatSidebar } from "../context/ChatSidebarContext";

const ChatSidebarToggle: React.FC = () => {
    const { isChatOpen, toggleChat } = useChatSidebar();
  return (
    <>
      {!isChatOpen && (
        <button
          onClick={toggleChat}
          className="fixed bottom-6 right-6 z-50 p-4 rounded-full bg-red-500 text-white shadow-lg hover:bg-red-600 focus:outline-none focus:ring-2 focus:ring-red-400"
          aria-label={isChatOpen ? "Close Chat Sidebar" : "Open Chat Sidebar"}
        >
          <ChatIcon className="w-6 h-6" />
        </button>
      )}
    </>
  );
};

export default ChatSidebarToggle;
