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

// Basic offline FAQ entries (Kannada + simple English fallbacks)
const OFFLINE_FAQ: { patterns: RegExp[]; answer: string }[] = [
  {
    patterns: [/(hello|hi|namaskara|namaskar|ನಮಸ್ಕಾರ)/i],
    answer:
      "ನಮಸ್ಕಾರ! ಇಂಟರ್ನೆಟ್ ಸಂಪರ್ಕ ಇಲ್ಲ. ನಾನು ಮೂಲಭೂತ ಪ್ರಶ್ನೆಗಳಿಗೆ ಉತ್ತರಿಸಬಹುದು. \nHello! You're offline. I can answer a few basic questions.",
  },
  {
    patterns: [/(who are you|ನೀವು ಯಾರು|about you)/i],
    answer:
      "ನಾನು ಸ್ಮಾರ್ಟ್ ಕೃಷಿ ಸಹಾಯಕ. ಗೊಬ್ಬರ ಸಲಹೆಗಳಿಗಾಗಿ ತಯಾರಿಸಲಾಗಿದೆ. \nI'm a Smart Agriculture assistant for basic fertilizer guidance.",
  },
  {
    patterns: [/(help|ಸಹಾಯ)/i],
    answer:
      "ಆಫ್‌ಲೈನ್‌ನಲ್ಲಿ ನಾನು: ಶುಭಾಶಯ, ನನ್ನ ಬಗ್ಗೆ, ಬೆಳೆಗಳ ಪಟ್ಟಿ, ಗೊಬ್ಬರದ ಮೂಲ ಸೂಚನೆಗಳನ್ನು ನೀಡಬಲ್ಲೆ. \nOffline I can answer: greetings, about, list of crops, basic fertilizer tips.",
  },
  {
    patterns: [/(crops|ಬೆಳೆಗಳು)/i],
    answer:
      "ಉದಾಹರಣೆ ಬೆಳೆಗಳು: ಭತ್ತ, ರಾಗಿ, ಜೋಳ, ಗೋಧಿ, ಕಬ್ಬು. \nExample crops: rice, ragi, maize, wheat, sugarcane.",
  },
  {
    patterns: [/(fertilizer|ಗೊಬ್ಬರ|NPK)/i],
    answer:
      "ಮೂಲ ಸಲಹೆ: ಮಣ್ಣಿನ ಪರೀಕ್ಷೆ ಮಾಡಿಸಿ N-P-K ಅವಶ್ಯಕತೆಯಂತೆ ಗೊಬ್ಬರ ಬಳಸಿ. ಆನ್‌ಲೈನ್ ಆದಲ್ಲಿ ಹೆಚ್ಚಿನ ಮಾರ್ಗದರ್ಶನ ನೀಡುತ್ತೇನೆ. \nBasic tip: test soil and apply N-P-K per crop need. When online, I can give detailed guidance.",
  },
  {
    patterns: [/(what is fertilizer|ಗೊಬ್ಬರ ಎಂದರೇನು)/i],
    answer:
      "ಗೊಬ್ಬರ ಎಂದರೆ ಸಸ್ಯಗಳ ಬೆಳವಣಿಗೆಗೆ ಬೇಕಾದ ಪೋಷಕಾಂಶಗಳನ್ನು ಒದಗಿಸುವ ವಸ್ತು. ಇದು ಮಣ್ಣಿನ ಫಲವತ್ತತೆಯನ್ನು ಹೆಚ್ಚಿಸುತ್ತದೆ. \nA fertilizer is a substance that provides nutrients for plant growth, improving soil fertility.",
  },
  {
    patterns: [/(what is npk|npk ಎಂದರೇನು|NPK)/i],
    answer:
      "NPK ಎಂದರೆ ಸಾರಜನಕ (N), ರಂಜಕ (P), ಮತ್ತು ಪೊಟ್ಯಾಸಿಯಮ್ (K). ಇವು ಸಸ್ಯಗಳ ಬೆಳವಣಿಗೆಗೆ ಅತ್ಯಗತ್ಯವಾದ ಮೂರು ಮುಖ್ಯ ಪೋಷಕಾಂಶಗಳು. \nNPK stands for Nitrogen (N), Phosphorus (P), and Potassium (K) - the three primary nutrients for plant growth.",
  },
  {
    patterns: [/(nitrogen|ಸಾರಜನಕ)/i],
    answer: "ಸಾರಜನಕ (N) ಎಲೆಗಳ ಬೆಳವಣಿಗೆಗೆ ಮತ್ತು ಸಸ್ಯಕ್ಕೆ ಹಸಿರು ಬಣ್ಣ ನೀಡಲು ಮುಖ್ಯವಾಗಿದೆ. \nNitrogen (N) is vital for leaf growth and gives plants their green color.",
  },
  {
    patterns: [/(phosphorus|ರಂಜಕ)/i],
    answer: "ರಂಜಕ (P) ಬೇರು, ಹೂವು ಮತ್ತು ಹಣ್ಣುಗಳ ಬೆಳವಣಿಗೆಗೆ ಸಹಾಯ ಮಾಡುತ್ತದೆ. \nPhosphorus (P) helps in the development of roots, flowers, and fruits.",
  },
  {
    patterns: [/(potassium|ಪೊಟ್ಯಾಸಿಯಮ್)/i],
    answer: "ಪೊಟ್ಯಾಸಿಯಮ್ (K) ಸಸ್ಯದ ಒಟ್ಟಾರೆ ಆರೋಗ್ಯ ಮತ್ತು ರೋಗ ನಿರೋಧಕ ಶಕ್ತಿಯನ್ನು ಹೆಚ್ಚಿಸುತ್ತದೆ. \nPotassium (K) boosts a plant's overall health and disease resistance.",
  },
  {
    patterns: [/(paddy|ಭತ್ತ|sugarcane|ಕಬ್ಬು|wheat|ಗೋಧಿ)/i],
    answer:
      "ಪ್ರತಿ ಬೆಳೆಗೆ ನಿರ್ದಿಷ್ಟ ಗೊಬ್ಬರದ ಅವಶ್ಯಕತೆ ಇರುತ್ತದೆ. ಉದಾಹರಣೆಗೆ, ಭತ್ತಕ್ಕೆ ಸಾರಜನಕ ಹೆಚ್ಚು ಬೇಕು. ನಿಖರವಾದ ಶಿಫಾರಸುಗಳಿಗಾಗಿ ದಯವಿಟ್ಟು ಆನ್‌ಲೈನ್‌ಗೆ ಬನ್ನಿ. \nEach crop has specific fertilizer needs. For example, paddy (rice) often requires more nitrogen. For precise recommendations, please connect to the internet.",
  },
  {
    patterns: [/(soil type|ಮಣ್ಣಿನ ವಿಧ|red soil|black soil)/i],
    answer:
      "ಮಣ್ಣಿನಲ್ಲಿ ಹಲವು ವಿಧಗಳಿವೆ, ಉದಾಹರಣೆಗೆ ಕೆಂಪು ಮಣ್ಣು, ಕಪ್ಪು ಮಣ್ಣು, ಮತ್ತು ಮರಳು ಮಣ್ಣು. ಪ್ರತಿಯೊಂದು ವಿಧವು ವಿಭಿನ್ನ ಬೆಳೆಗಳಿಗೆ ಸೂಕ್ತವಾಗಿದೆ. \nThere are many soil types, like red soil, black soil, and sandy soil. Each type is suitable for different crops.",
  },
  {
    patterns: [/(how to use|ಬಳಸುವುದು ಹೇಗೆ|use this|guide)/i],
    answer:
      "ಸಂದೇಶವನ್ನು ಟೈಪ್ ಮಾಡಿ Enter ಒತ್ತಿ. ಇಂಟರ್ನೆಟ್ ಇದ್ದರೆ ವಿವರವಾದ ಉತ್ತರ; ಇಲ್ಲದಿದ್ದರೆ ಮೂಲಭೂತ ಉತ್ತರ. \nType your question and press Enter. Detailed answers when online; basic answers offline.",
  },
  {
    patterns: [/(reset|ಮರುಪ್ರಾರಂಭ|restart|clear chat)/i],
    answer:
      "ಮತ್ತೆ ಪ್ರಾರಂಭಿಸಲು 'ಮರುಪ್ರಾರಂಭ' ಚಿಪ್ ಒತ್ತಿ ಅಥವಾ 'ಮರುಪ್ರಾರಂಭ' ಎಂದು ಬರೆಯಿರಿ. \nTo start fresh, click the 'ಮರುಪ್ರಾರಂಭ' chip or type 'reset'.",
  },
  {
    patterns: [/(language|ಭಾಷೆ|Kannada|English)/i],
    answer:
      "ನಾನು ಕನ್ನಡ ಮತ್ತು ಇಂಗ್ಲಿಷ್ ಎರಡನ್ನೂ ಬಲ್ಲೆ. ನಿಮಗೆ ಇಷ್ಟವಾದ ಭಾಷೆಯಲ್ಲಿ ಬರೆಯಿರಿ. \nI understand Kannada and English. Use whichever you prefer.",
  },
  {
    patterns: [/(privacy|data|ಗೋಪ್ಯತೆ|ಡೇಟಾ)/i],
    answer:
      "ಆಫ್‌ಲೈನ್ ಉತ್ತರಗಳು ನಿಮ್ಮ ಸಾಧನದಲ್ಲೇ ಇರುತ್ತವೆ. ಆನ್‌ಲೈನ್ ವೇಳೆ ಮಾತ್ರ ಸರ್ವರ್ ಸಂಪರ್ಕ ಬೇಕಾಗುತ್ತದೆ. \nOffline replies stay on your device. Online queries contact the server.",
  },
  {
    patterns: [/(soil test|ಮಣ್ಣಿನ ಪರೀಕ್ಷೆ|pH)/i],
    answer:
      "ಮಣ್ಣಿನ ಪರೀಕ್ಷೆ ಮಾಡಿ pH, N-P-K, ಕಾರ್ಬನ್ ಮಟ್ಟ ತಿಳಿದುಕೊಳ್ಳಿ. ಸ್ಥಳೀಯ ಕೃಷಿ ಕಚೇರಿಯಿಂದ ವರದಿ ಪಡೆಯಿರಿ. \nDo a soil test for pH, N-P-K, organic carbon via local agri lab.",
  },
  {
    patterns: [/(dosage|ಮಾತ್ರೆ|apply how much|per acre)/i],
    answer:
      "ಮಾತ್ರೆ ಬೆಳೆ, ಮಣ್ಣು ಮತ್ತು ಹಂತದ ಮೇಲೆ ಅವಲಂಬಿತ. ಆನ್‌ಲೈನ್ ಆದಾಗ ಬೆಳೆ/ಏಕರೆ ವಿವರ ಕೊಡಿ, ನಿಖರ ಸಲಹೆ ನೀಡುತ್ತೇನೆ. \nDosage depends on crop, soil and stage. When online, provide crop/acre for precise rates.",
  },
  {
    patterns: [/(weather|ಹವಾಮಾನ|rain|ಮಳೆ)/i],
    answer:
      "ಆಫ್‌ಲೈನ್‌ನಲ್ಲಿ ಹವಾಮಾನ ಮಾಹಿತಿ ಲಭ್ಯವಿಲ್ಲ. ಆನ್‌ಲೈನ್‌ನಲ್ಲಿ ಸ್ಥಳದ ಹೆಸರು ನೀಡಿ, ತಾಜಾ ಮಾಹಿತಿ ಕೊಡುತ್ತೇನೆ. \nWeather data needs internet. When online, share your location for updates.",
  },
  {
    patterns: [/(contact|support|ಸಂಪರ್ಕ|helpdesk)/i],
    answer:
      "ಸಹಾಯಕ್ಕಾಗಿ ಸ್ಥಳೀಯ ಕೃಷಿ ಸಹಾಯವಾಣಿ ಅಥವಾ ಕೃಷಿ ಅಧಿಕಾರಿ ಸಂಪರ್ಕಿಸಿ. ಆನ್‌ಲೈನ್‌ನಲ್ಲಿ ನಾನು ಹೆಚ್ಚಿನ ಮಾರ್ಗದರ್ಶನ ನೀಡುತ್ತೇನೆ. \nContact your local agri helpline/officer; I'll guide more when online.",
  },
  {
    patterns: [/(hours|time|ಸಮಯ)/i],
    answer:
      "ನಾನು 24x7 ಲಭ್ಯ. ಆದರೆ ವಿವರವಾದ ಉತ್ತರಗಳಿಗೆ ಇಂಟರ್ನೆಟ್ ಅಗತ್ಯ. \nAvailable 24x7; detailed answers require internet.",
  },
  {
    patterns: [/(where|location|ಸ್ಥಳ|office)/i],
    answer:
      "ನಿಮ್ಮ ತಾಲೂಕು ಕೃಷಿ ಕಚೇರಿಯ ವಿಳಾಸಕ್ಕಾಗಿ ಸ್ಥಳೀಯ ಸರ್ಕಾರದ ವೆಬ್‌ಸೈಟ್ ನೋಡಿ. \nFind your taluk agriculture office on your state agriculture website.",
  },
  {
    patterns: [/(thanks|ಧನ್ಯವಾದ|thank you|bye|goodbye|ಬಾಯ್)/i],
    answer:
      "ಧನ್ಯವಾದಗಳು! ಮತ್ತೆ ಭೇಟಿ ಆಗೋಣ. \nThank you! See you again.",
  },
  {
    patterns: [/(irrigation|ನೆರವು|watering)/i],
    answer:
      "ಬಿತ್ತನೆ ನಂತರ ಹಗುರ ನೀರಾವರಿ ನೀಡಿ. ಮಣ್ಣಿನ ತೇವಮಾನ ಕಾಯ್ದುಕೊಳ್ಳಿ; ಅತಿಯಾಗಿ ನೀರು ಹಾಕಬೇಡಿ. \nProvide light irrigation post-sowing; keep soil moist, avoid overwatering.",
  },
  {
    patterns: [/(pest|ರೋಗ|ಹುಳು|disease)/i],
    answer:
      "ರೋಗ/ಹುಳು ಗುರುತು ಮಾಡಿ. ಲೇಬಲ್ ಪ್ರಕಾರ ಜೈವಿಕ ಅಥವಾ ರಾಸಾಯನಿಕ ನಿಯಂತ್ರಣ. ಆನ್‌ಲೈನ್‌ನಲ್ಲಿ ಲಕ್ಷಣಗಳು ನೀಡಿದರೆ ವಿಶೇಷ ಸಲಹೆ. \nIdentify pest/disease; use labeled bio/chemical control. Online I can tailor advice.",
  },
  {
    patterns: [/(safety|ಸುರಕ್ಷತೆ|PPE)/i],
    answer:
      "ಗೊಬ್ಬರ/ಕೀಟನಾಶಕ ಬಳಕೆಯ ವೇಳೆ ಗ್ಲೋವ್ಸ್, ಮಾಸ್ಕ್, ಬೂಟ್ಸ್ ಧರಿಸಿ. ಕೈಗಳನ್ನು ಚೆನ್ನಾಗಿ ತೊಳೆಯಿರಿ. \nWear gloves, mask, boots when handling inputs; wash hands after use.",
  },
  {
    patterns: [/(storage|ಸಂಗ್ರಹ|store fertilizers)/i],
    answer:
      "ಗೊಬ್ಬರಗಳನ್ನು ಒಣ, ತಂಪು, ವಾತಾಯಿತ ಸ್ಥಳದಲ್ಲಿ ಮಕ್ಕಳಿಂದ ದೂರ ಸಂಗ್ರಹಿಸಿ. \nStore fertilizers cool, dry, ventilated and away from children.",
  },
  {
    patterns: [/(organic farming|ಸಾವಯವ ಕೃಷಿ)/i],
    answer: "ಸಾವಯವ ಕೃಷಿ ರಾಸಾಯನಿಕಗಳಿಲ್ಲದೆ ನೈಸರ್ಗಿಕ ವಿಧಾನಗಳನ್ನು (ಉದಾ: ಕೊಟ್ಟಿಗೆ ಗೊಬ್ಬರ) ಬಳಸಿ ಬೆಳೆ ಬೆಳೆಯುವ ಪದ್ಧತಿ. \nOrganic farming is a method of growing crops using natural methods (e.g., manure) without synthetic chemicals.",
  },
  {
    patterns: [/(crop rotation|ಬೆಳೆ ಪರಿವರ್ತನೆ)/i],
    answer: "ಬೆಳೆ ಪರಿವರ್ತನೆ ಎಂದರೆ ಒಂದೇ ಜಾಗದಲ್ಲಿ ಬೇರೆ ಬೇರೆ ಬೆಳೆಗಳನ್ನು ಕ್ರಮವಾಗಿ ಬೆಳೆಯುವುದು. ಇದು ಮಣ್ಣಿನ ಆರೋಗ್ಯವನ್ನು ಕಾಪಾಡುತ್ತದೆ ಮತ್ತು ರೋಗಗಳನ್ನು ಕಡಿಮೆ ಮಾಡುತ್ತದೆ. \nCrop rotation is growing different crops in the same area in sequence. It maintains soil health and reduces diseases.",
  },
];

function getOfflineAnswer(userInput: string): string | null {
  for (const entry of OFFLINE_FAQ) {
    if (entry.patterns.some((re) => re.test(userInput))) {
      return entry.answer;
    }
  }
  return null;
}

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

    // If offline, answer locally and return (single reply, no duplicates)
    if (typeof navigator !== 'undefined' && navigator && navigator.onLine === false) {
      const offlineAnswer = getOfflineAnswer(userInput);
      const botResponse = offlineAnswer || [
        "ಆಫ್‌ಲೈನ್‌ ಮೋಡ್. ",
        "Offline mode: I can answer basic topics like:",
        "- ಶುಭಾಶಯ / Greetings (hello)",
        "- ನನ್ನ ಬಗ್ಗೆ / About",
        "- ಬೆಳೆಗಳು / Crops",
        "- ಗೊಬ್ಬರ / Fertilizer/NPK",
        "- ಮಣ್ಣಿನ ಪರೀಕ್ಷೆ / Soil test",
        "- ಸಾವಯವ ಕೃಷಿ / Organic Farming",
        "- ಬೆಳೆ ಪರಿವರ್ತನೆ / Crop Rotation",
      ].join("\n");

      setMessages(prev => {
        const lastBot = [...prev].reverse().find(m => m.sender === MessageSender.BOT);
        if (lastBot && lastBot.text.trim() === botResponse.trim()) {
          return prev; // skip duplicate
        }
        return [...prev, { id: Date.now() + 1, text: botResponse, sender: MessageSender.BOT }];
      });
      setIsThinking(false);
      setSuggestionChips([
        { text: "ಸಹಾಯ", icon: "❓" },
        { text: "ಬೆಳೆಗಳು", icon: "🌱" },
        { text: "ಮಣ್ಣಿನ ಪರೀಕ್ಷೆ" },
        { text: "ಮಾತ್ರೆ" },
        { text: "ಮರುಪ್ರಾರಂಭ", icon: "🔄" },
      ]);
      return;
    }

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
      // On network/AI error, try offline fallback
      const offline = getOfflineAnswer(userInput);
      const errorMessage = offline || "ಆಫ್‌ಲೈನ್‌/ದೋಷ: ದಯವಿಟ್ಟು ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ. \nOffline/error: Please try again later.";
      setError(offline ? null : errorMessage);
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