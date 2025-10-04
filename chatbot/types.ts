
export enum MessageSender {
  USER = 'user',
  BOT = 'bot',
}

export interface Message {
  id: number;
  text: string;
  sender: MessageSender;
}
