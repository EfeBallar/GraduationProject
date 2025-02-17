import { useState, useRef, useEffect } from "react";
import axios from "axios";
import SideBar from "./SideBar";
import Message from "./Message";
import InputBox from "./InputBox";

function MainScreen() {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [course, setCourse] = useState(null);
  const [chat, setChat] = useState(null);
  const [historyArr, setHistoryArr] = useState([]);
  const [selectedHistory, setSelectedHistory] = useState(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    async function getChatContent() {
      try {
        const response = await axios.get(
          "http://localhost:5000/getChatContent",
          {
            params: {
              chat_id: chat,
            },
          }
        );
        setMessages(response.data.chat.messages);
        setCourse(response.data.chat.course)
      } catch (error) {
        console.log("Error fetching data: ", error);
      }
    }

    if(chat) {
      getChatContent();
    }
  }, [chat]);

  useEffect(() => {
    async function getHistory() {
      try {
        const response = await axios.get("http://localhost:5000/getUserChats", {
          params: {
            user_name: "674b169e502419ebf6cfb296",
          },
        });
        setHistoryArr(response.data.chats);
      } catch (error) {
        console.log("Error fetching data: ", error);
      }
    }

    getHistory();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleNewMessage(userMessage) {
    console.log(`Chat ID : ${chat}`);
    console.log(`User Message: ${userMessage}`);

    setLoading(true);
    const newMessages = [
      ...messages,
      { message_content: userMessage, sender: "user", sources: null },
    ];
    setMessages(newMessages);

    const data = {
      term: "F24-25",
      course: course,
      question: userMessage,
      user_id: "674b169e502419ebf6cfb296",
      chat_id: chat,
    };



    await axios.post("http://localhost:5000/", data).then((response) => {
      console.log(response);

      const data = response.data;
      const fieldValue = data.model_response;
      setMessages([
        ...newMessages,
        {
          message_content:
            fieldValue,
          sender: "chatbot",
          sources: data.sources,
        },
      ]);
      const oldChatStatus = chat;
      setChat(data.chat_id);
      setCourse(data.course)

      setLoading(false);

        if (!oldChatStatus && data.server_response) {
            const newHistory = [
                {_id: data.chat_id, course: data.course, last_message_time: data.last_message_time, title: data.title},
                ...historyArr
            ];
            setSelectedHistory(data.chat_id)
            setHistoryArr(newHistory)
        }
    });

  }

  return (
    <div className="h-screen w-full flex">
      <SideBar setSelectedHistory = {setSelectedHistory} selectedHistory={selectedHistory} chat={chat} course={course} setCourse={setCourse} setChat={setChat} setMessages={setMessages} historyArr={historyArr} />

      <div className="flex flex-col w-3/4 justify-end">
        <div className="flex flex-col flex-grow w-full overflow-y-auto items-center">
          <div className="flex flex-col w-3/4 overflow-y-visible">
            {
              messages.map((msg, index) => (

              <Message
                key={index}
                text={msg.message_content}
                sender={msg.sender}
                sources={msg.sources}
              />
            ))}
            <div ref={messagesEndRef} />
          </div>
        </div>
        {course ? (
          <InputBox onSendMessage={handleNewMessage} loading={loading} />
        ) : (
          <div className="flex items-center justify-center">
            <h1 className="text-gray-500 text-2xl italic mb-6">
              Please Select a Course
            </h1>
          </div>
        )}
      </div>
    </div>
  );
}

export default MainScreen;
