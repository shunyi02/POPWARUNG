import { SidebarProvider, useSidebar } from "../context/SidebarContext";
import { ChatSidebarProvider } from "../context/ChatSidebarContext"; // Added ChatSidebarProvider import
import { Outlet } from "react-router";
import AppHeader from "./AppHeader";
import Backdrop from "./Backdrop";
import AppSidebar from "./AppSidebar";
import ChatSidebarToggle from "../warung/WarungChat";
import ChatSidebar from "../warung/WarungChatAppSidebar";

import { useChatSidebar } from "../context/ChatSidebarContext"; // Import useChatSidebar

const LayoutContent: React.FC = () => {
  const { isExpanded, isHovered, isMobileOpen } = useSidebar();
  const { isChatOpen } = useChatSidebar(); // Get isChatOpen from useChatSidebar

  return (
    <div className="min-h-screen xl:flex">
      <div>
        <AppSidebar />
        <Backdrop />
      </div>
      <div
        className={`flex-1 transition-all duration-300 ease-in-out ${
          isExpanded || isHovered ? "lg:ml-[290px]" : "lg:ml-[90px]"
        } ${isMobileOpen ? "ml-0" : ""} ${isChatOpen ? "mr-[400px]" : ""}`} // Add right margin when chat is open
      >
        <AppHeader />
        <div className="p-4 mx-auto max-w-(--breakpoint-2xl) md:p-6">
          <Outlet />
        </div>
      </div>

      {/* Add your Chat Sidebar toggle */}
      <ChatSidebarToggle /> {/* Removed onToggle and isOpen props */}

      {/* ✅ Actual Sidebar */}
      <ChatSidebar /> {/* Removed isOpen and onClose props */}
    </div>
  );
};

const AppLayout: React.FC = () => {
  return (
    <SidebarProvider>
      <ChatSidebarProvider> {/* Wrapped LayoutContent with ChatSidebarProvider */}
        <LayoutContent />
      </ChatSidebarProvider>
    </SidebarProvider>
  );
};

export default AppLayout;
