import React, { useState, useEffect, useRef, useCallback } from 'react';
import { GoogleGenAI, Chat } from '@google/genai';
import { Message, MessageSender } from './types';
import { SYSTEM_INSTRUCTION } from './constants';
import ChatBubble from './components/ChatBubble';
import ChatInput from './components/ChatInput';
import Header from './components/Header';
import TypingIndicator from './components/TypingIndicator';
import Footer from './components/Footer';

interface Chip {
  text: string;
  icon?: string;
}

const DEFAULT_CHIPS: Chip[] = [
  { text: "ಸಹಾಯ", icon: "❓" },
  { text: "ಮರುಪ್ರಾರಂಭ", icon: "🔄" },
  { text: "ಬೆಳೆಗಳು", icon: "🌱" }
];

// SuggestionChips Component
const SuggestionChips: React.FC<{ chips: Chip[]; onChipSelect: (text: string) => void }> = ({ chips, onChipSelect }) => {
  if (!chips || chips.length === 0) {
    return null;
  }

  return (
    <div className="px-4 py-3 bg-[#F0FDF4] border-b border-gray-200 flex items-center justify-center flex-wrap gap-2">
      {chips.map(chip => (
        <button
          key={chip.text}
          onClick={() => onChipSelect(chip.text)}
          className="px-4 py-2 text-sm font-medium text-green-800 bg-white border border-green-200 rounded-full hover:bg-green-50 transition-colors"
        >
          {chip.icon && <span className="mr-2">{chip.icon}</span>}
          {chip.text}
        </button>
      ))}
    </div>
  );
};



const App: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isThinking, setIsThinking] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [suggestionChips, setSuggestionChips] = useState<Chip[]>(DEFAULT_CHIPS);
  const chatSessionRef = useRef<Chat | null>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  const initializeChat = useCallback(async () => {
    try {
      if (!process.env.API_KEY) {
        throw new Error("API_KEY environment variable not set.");
      }
      const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
      const chat = ai.chats.create({
        model: 'gemini-2.5-flash',
        config: {
          systemInstruction: SYSTEM_INSTRUCTION,
        },
      });
      chatSessionRef.current = chat;

      const initialMessage: Message = {
        id: Date.now(),
        text: "ನಮಸ್ಕಾರ! ನಾನು ನಿಮ್ಮ ಸ್ಮಾರ್ಟ್ ಕೃಷಿ ಸಹಾಯಕ. ನಾನು ನಿಮ್ಮ ಬೆಳೆಗೆ ತಕ್ಕ ಗೊಬ್ಬರವನ್ನು ಹುಡುಕಲು ಸಹಾಯ ಮಾಡುತ್ತೇನೆ. ನೀವು ಯಾವ ಬೆಳೆಯನ್ನು ಬೆಳೆಯಲು ಯೋಚಿಸುತ್ತಿದ್ದೀರಿ?",
        sender: MessageSender.BOT
      };
      setMessages([initialMessage]);
      setSuggestionChips([
          { text: "ಭತ್ತ" }, // Paddy/Rice
          { text: "ರಾಗಿ" }, // Ragi/Finger Millet
          { text: "ಕಬ್ಬು" }, // Sugarcane
      ]);
    } catch (e) {
      console.error(e);
      setError("Failed to initialize the chat session. Please check your API key and refresh the page.");
    } finally {
      setIsThinking(false);
    }
  }, []);

  useEffect(() => {
    initializeChat();
  }, [initializeChat]);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages, isThinking]);

  const handleSendMessage = async (userInput: string) => {
    if (!userInput.trim()) return;

    setSuggestionChips([]);
    const userMessage: Message = {
      id: Date.now(),
      text: userInput,
      sender: MessageSender.USER,
    };
    setMessages(prev => [...prev, userMessage]);
    setIsThinking(true);
    setError(null);

    try {
      if (!chatSessionRef.current) {
        throw new Error("Chat session is not initialized.");
      }
      
      const stream = await chatSessionRef.current.sendMessageStream({ message: userInput });
      
      let firstChunk = true;
      const botMessageId = Date.now() + 1;
      let fullBotResponse = '';
      let suggestionsFound = false;

      for await (const chunk of stream) {
        const textChunk = chunk.text;
        fullBotResponse += textChunk;
        
        let displayableText = fullBotResponse;
        const suggestionRegex = /\[SUGGESTIONS:\s*([^\]]+)\]/s;
        const match = fullBotResponse.match(suggestionRegex);
        
        if (match && match[1]) {
            try {
                const suggestionsText: string[] = JSON.parse(`[${match[1]}]`);
                const newChips: Chip[] = suggestionsText.map(text => ({ text }));
                setSuggestionChips(newChips);
                displayableText = fullBotResponse.replace(suggestionRegex, '').trim();
                suggestionsFound = true;
            } catch (e) {
                console.error("Failed to parse suggestions:", match[1]);
                setSuggestionChips([]);
            }
        }

        if (firstChunk) {
          setIsThinking(false);
          const botMessagePlaceholder: Message = {
            id: botMessageId,
            text: displayableText,
            sender: MessageSender.BOT,
          };
          setMessages(prev => [...prev, botMessagePlaceholder]);
          firstChunk = false;
        } else {
          setMessages(prev => prev.map(msg => 
            msg.id === botMessageId 
                ? { ...msg, text: displayableText } 
                : msg
          ));
        }
      }
      if (!suggestionsFound) {
        setSuggestionChips(DEFAULT_CHIPS);
      }

    } catch (e) {
      console.error(e);
      const errorMessage = "Sorry, I encountered an error. Please try again.";
      setError(errorMessage);
       setMessages(prev => [...prev, { id: Date.now() + 1, text: errorMessage, sender: MessageSender.BOT }]);
       setIsThinking(false);
       setSuggestionChips(DEFAULT_CHIPS);
    }
  };

  const handleChipSelect = (chipText: string) => {
    if (chipText === "ಮರುಪ್ರಾರಂಭ") {
      setIsThinking(true);
      setMessages([]);
      setSuggestionChips([]);
      initializeChat();
    } else {
      handleSendMessage(chipText);
    }
  };

  return (
    <div className="flex justify-center items-center h-screen bg-gray-100 font-sans">
      <div className="w-full max-w-2xl h-full sm:h-[95vh] sm:max-h-[800px] flex flex-col bg-white shadow-2xl rounded-2xl border border-gray-200">
        <Header />
        <SuggestionChips chips={suggestionChips} onChipSelect={handleChipSelect} />
        <div ref={chatContainerRef} className="flex-1 p-6 overflow-y-auto space-y-6 bg-[#F0FDF4]">
          {messages.map(msg => (
            <ChatBubble key={msg.id} message={msg} />
          ))}
          {isThinking && <TypingIndicator />}
        </div>
        {error && <div className="p-4 text-center text-red-600 bg-red-100">{error}</div>}
        <ChatInput onSendMessage={handleSendMessage} isLoading={isThinking} />
        <Footer />
      </div>
    </div>
  );
};

export default App;