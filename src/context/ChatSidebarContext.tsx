import { createContext, useContext, useState, useEffect } from "react";

type ChatSidebarContextType = {
  isChatOpen: boolean;
  isChatMobile: boolean;
  isChatHovered: boolean;
  toggleChat: () => void;
  openChat: () => void;
  closeChat: () => void;
  setChatHovered: (hovered: boolean) => void;
};

const ChatSidebarContext = createContext<ChatSidebarContextType | undefined>(undefined);

export const useChatSidebar = () => {
  const context = useContext(ChatSidebarContext);
  if (!context) {
    throw new Error("useChatSidebar must be used within a ChatSidebarProvider");
  }
  return context;
};

export const ChatSidebarProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [isChatMobile, setIsChatMobile] = useState(false);
  const [isChatHovered, setIsChatHovered] = useState(false);

  useEffect(() => {
    const handleResize = () => {
      const mobile = window.innerWidth < 768;
      setIsChatMobile(mobile);
    };

    handleResize();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  const toggleChat = () => setIsChatOpen((prev) => !prev);
  const openChat = () => setIsChatOpen(true);
  const closeChat = () => setIsChatOpen(false);
  const setChatHovered = (hovered: boolean) => setIsChatHovered(hovered);

  return (
    <ChatSidebarContext.Provider
      value={{
        isChatOpen,
        isChatMobile,
        isChatHovered,
        toggleChat,
        openChat,
        closeChat,
        setChatHovered,
      }}
    >
      {children}
    </ChatSidebarContext.Provider>
  );
};
