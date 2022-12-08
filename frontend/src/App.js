import { BrowserRouter, Route, Routes, Navigate } from "react-router-dom";
import Matting from "./user/Matting";
import Login from "./login/Login";
import Navbar from "./components/Navbar";
import Register from "./register/Register";
import AdminControl from "./admin/AdminControl"
import { useEffect, useState } from "react";
import "./App.css";

function App() {
  const [logged, setLogged] = useState(true);

  let routes;

  routes = (
      <Routes>
        <Route exact path="/" element={!logged ? <Navigate replace to="/login" /> : <Matting />} />
        <Route exact path="/register" element={<Register/>} />
        <Route path="/login" element={<Login setLogged={setLogged} />} />
        <Route path="/admin-control" element={<AdminControl/>} />
      </Routes>
    )

  return (
    <BrowserRouter>
      {logged ? <Navbar setLogged={setLogged}/> : <></>} 
      { routes }
    </BrowserRouter>
  );
}
 
export default App