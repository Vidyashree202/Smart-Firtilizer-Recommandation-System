import React from 'react';
import { Message, MessageSender } from '../types';

interface ChatBubbleProps {
  message: Message;
}

const ChatBubble: React.FC<ChatBubbleProps> = ({ message }) => {
  const isUser = message.sender === MessageSender.USER;
  const time = new Date(message.id).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

  const bubbleClasses = isUser
    ? 'bg-green-700 text-white self-end rounded-xl rounded-br-none'
    : 'bg-white text-gray-800 self-start rounded-xl rounded-bl-none border border-gray-200';

  const containerClasses = isUser ? 'flex justify-end' : 'flex justify-start items-start space-x-3';

  return (
    <div className={containerClasses}>
      {!isUser && (
         <div className="w-9 h-9 bg-green-700 rounded-full flex items-center justify-center flex-shrink-0">
            <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 256 256"><path d="M12,2c5.523,0,10,4.477,10,10s-4.477,10-10,10S2,17.523,2,12S6.477,2,12,2Zm2.553,13.658c.278-.278.447-.66.447-1.075,0-.828-.672-1.5-1.5-1.5s-1.5.672-1.5,1.5c0,.415.169.797.447,1.075C11.305,16.542,10,16.059,10,14.5c0-1.381,1.119-2.5,2.5-2.5s2.5,1.119,2.5,2.5c0,1.559-1.305,2.042-2.447,1.158ZM12.02,11h-.04C9.248,11,7,8.752,7,6c0-1.58,1.47-2.83,3.22-2.83.21,0,.42,0,.62.05,1.58.19,2.94,1.4,3.14,2.95.01.1.02.2.02.31C14,8.752,11.772,11,8.98,11h-.04C10.081,9.742,11.02,9,12,9s1.919.742,3.02,2Z" transform="scale(10.66)"></path></svg>
         </div>
      )}
      <div className={`max-w-md lg:max-w-xl px-4 py-3 shadow-sm relative ${bubbleClasses}`}>
        <p className="text-sm whitespace-pre-wrap">{message.text}</p>
        <div className={`text-xs mt-2 ${isUser ? 'text-green-200 text-right' : 'text-gray-400 text-left'}`}>
          {time}
        </div>
      </div>
    </div>
  );
};

export default ChatBubble;