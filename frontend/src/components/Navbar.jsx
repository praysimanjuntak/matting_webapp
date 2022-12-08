import React, { useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import styled from "styled-components";
import People from "../svg/people.svg";
import ManyPeople from "../svg/manypeople.svg";
import { useEffect } from "react";
import axios from "axios";

const Container = styled.div`
  height: 70px;
  position: sticky;
  top: 0;
  width: 100%;
  z-index: 2001;
  background: #4472c4;
`;

const Wrapper = styled.div`
  padding: 10px 20px;
  display: flex;
  flex-wrap: wrap;
  backgroud: white;
  justify-content: space-between;
  align-items: center;
`;

const Left = styled.div`
  display: flex;
  padding: 0px 10px;
  align-items: flex-start;
`;

const ItemsWrap = styled.div`
  display: flex;
  align-items: center;
  justify-content: flex-end;
`

const MenuItem = styled.div`
  font-size: 14px;
  cursor: pointer;
  margin-left: 25px;
  height: 50px;
  &:hover {
    background-color: #3864b2;
  }
`;

const Menu = styled.div`
  width: 200px;
  height: 50px;
  position: absolute;
  right: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  border: 2px solid #4472c4;
  font-size: 20px;
  cursor: pointer;
`

const Navbar = ({ setLogged }) => {
  const [showMenu, setShowMenu] = useState(false);
  const navigate = useNavigate();
  const menuRef = useRef();

  useEffect(() => {
    if (showMenu) menuRef.current.style.display = "flex";
    else menuRef.current.style.display = "none";
  }, [showMenu])

  const Logout = async () => {
    try {
        await axios.delete('http://localhost:5000/logout');
        setLogged(false)
        navigate("/");
    } catch (error) {
        console.log(error);
    }
  }

  return (
    <Container>
      <Wrapper>
        <Left>
        </Left>
        <ItemsWrap>
            <MenuItem><Link style={{textDecoration: 'none', color: 'inherit'}} to="/admin-control">
              <img style={{width: '50px', height: '50px'}} src={ManyPeople} alt="manypeople" />
            </Link></MenuItem>
            <MenuItem onClick={() => setShowMenu(!showMenu)}>
              <img style={{width: '50px', height: '50px'}} src={People} alt="people" />  
            </MenuItem>
        </ItemsWrap>
      </Wrapper>
      <Menu onClick={Logout} ref={menuRef}>
        Sign Out
      </Menu>
    </Container>
  );
};

export default Navbar;