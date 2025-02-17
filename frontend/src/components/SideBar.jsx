import { useNavigate } from "react-router-dom";
import Select from "react-select";
import { FaCommentAlt } from "react-icons/fa";

function SideBar({ setCourse, course, setChat, chat, setMessages, historyArr, selectedHistory, setSelectedHistory }) {

  const navigate = useNavigate();



  const courseOptions = [
    { value: "CS305", label: "CS305" },
    { value: "CS404", label: "CS404" },
    { value: "CS449", label: "CS449" },
    { value: "CS302", label: "CS302" },
    { value: "CS401", label: "CS401" },
    { value: "CS307", label: "CS307" },
    { value: "PSY203", label: "PSY203" },
    { value: "NS206", label: "NS206" },
    { value: "SPS303", label: "SPS303" },
  ];


  function handleCourseChange(option) {
    setCourse(option.value);
  }

  function chatChange(id) {
    setChat(id);
    setSelectedHistory(id);
  }

  function handleNewChat() {
    setChat(null);
    setMessages([]);
    setSelectedHistory(null);
    setCourse(null);
  }

  //  #c10b25
  //  #002776
  //  #004890
  return (
    <div className="h-screen w-1/4 flex flex-col overflow-hidden" style={{ backgroundColor: '#004890' }}>
      <div className={'flex flex-row justify-center items-center'}>
        <h1 className="text-white text-4xl font-semibold drop-shadow-lg mb-4 ml-2 mr-auto">
          SuGPT
        </h1>
        <FaCommentAlt
            className="text-white text-3xl cursor-pointer hover:text-gray-200 mr-3"
            onClick={() => handleNewChat()}
        />
      </div>
      <h1 onClick={() => navigate("/profscreen")}  className="text-white text-xl font-semibold drop-shadow-lg mb-4 ml-2 cursor-pointer w-max">
        Admin
      </h1>

      <div className="flex flex-col items-center min-h-screen overflow-y-auto">
        <Select
          className="w-full max-w-xs mb-4"
          options={courseOptions}
          value={courseOptions.find(option => option.value === course) || null}
          onChange={handleCourseChange}
          placeholder="Select a course"
          isSearchable
          isDisabled={chat}
        />
        <p className="text-white text-2xl font-semibold mb-2">History</p>
        {historyArr.length > 0 ? (
          historyArr.map(({ _id, title, course }) => (
            <div
              key={_id}
              className={`${
                _id === selectedHistory ? "bg-gray-300" : "bg-white"
              } cursor-pointer w-11/12 text-black p-2 rounded-lg shadow-md mb-2`}
              onClick={() => chatChange(_id)}
            >
              {`${course}: ${title}`}
            </div>
          ))
        ) : (
          <p className="text-white italic">No history available</p>
        )}
      </div>
    </div>
  );
}

export default SideBar;
