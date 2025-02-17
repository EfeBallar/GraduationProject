import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import MainScreen from "./components/MainScreen";
import ProfScreen from "./components/ProfScreen";

function App() {
  return (
      <Router>
        <Routes>
          <Route path="/" element={<MainScreen />} />
          <Route path="/profScreen" element={<ProfScreen />} />
        </Routes>
      </Router>
  );
}

export default App;
