import React, { useState, useEffect, useCallback, useRef } from 'react'
import axios from 'axios';
import jwt_decode from "jwt-decode";
import styled from "styled-components";
import { useNavigate } from 'react-router-dom';
import { useDropzone } from 'react-dropzone';
import "./CircularLoader.css"

const downloadIcon = require('../svg/download.png');

// import openSocket from "socket.io-client";

const Container = styled.div`
    border: 2px solid black;
`

const Wrapper = styled.div`
    display: grid;
    grid-template-rows: 1fr 4fr;
    align-items: center;
    padding: 80px;
`

const Input = styled.input`
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap; /* 1 */
    clip-path: inset(50%);
    border: 0;
`

const Title = styled.label`
    text-align: center;
    font-size: 24px;
    border-radius: 5px;
    &: focus-within {
        outline: 5px solid;
    }
    color:#444;
    border:1px solid #CCC;
    box-shadow: 0 0 5px -1px rgba(0,0,0,0.2);
    cursor:pointer;
    vertical-align:middle;
    padding: 5px;
    text-align: center;
    &: hover {
        background:#DDD;
    }
`

const Text = styled.div`
    font-size: 32px;
`

const TitleMatting = styled.div`
    font-size: 50px;
    justify-self: center;
`

const Box = styled.div`
    display: grid;
    grid-template: 1fr 3fr / 1fr 1fr 1fr;
    column-gap: 50px;
    width: 100%;
`

const BoxContent = styled.div`
    grid-row-start: 2;
    display: grid;
    grid-template-rows: 2fr 1fr;
`

const UploadContainer = styled.div`
    border: 2px dashed black;
    display: flex;
    justify-content: center;
    align-items: center;
    aspect-ratio: 16 / 9;
    word-wrap: break-word;
`

const FileDesc = styled.div`
    font-size: 24px;
`

const Matting = () => {
    const [name, setName] = useState('');
    const [role, setRole] = useState('');
    const [token, setToken] = useState('');
    const [expire, setExpire] = useState('');
    const [users, setUsers] = useState([]);
    const [processing, setProcessing] = useState(false);
    const [videoFile, setVideoFile] = useState(null);
    const [imageFile, setImageFile] = useState(null);
    const [responseData, setResponseData] = useState(null);
    const [videoError, setVideoError] = useState(false);
    const [responseError, setResponseError] = useState(false); // if composition video is playable
    const playerRef = useRef(null);
    // const [progress, setProgress] = useState(0);
    const history = useNavigate();

    // useEffect(() => {
    //     const socket = openSocket("http://localhost:8000");
    //     socket.on("progress_update", data => {
    //     setProgress(data);
    //     });
    // }, [])

    // useEffect(() => {
    //     if (videoFile) handleVideo();
    // }, [videoFile])

    const handleVideo = async () => {
        await setProcessing(true);
        await setResponseData(null);
        console.log(videoFile)

        const formData = new FormData();
        formData.append("video", videoFile, videoFile.name)
        if (imageFile) {
            formData.append("image", imageFile, imageFile.name)
        }
        fetch('http://localhost:8000/video', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            return response.blob()
        })
        .then(blob => {
            let url = window.URL.createObjectURL(blob);
            setResponseData(url)
            // let a = document.createElement('a');
            // a.href = url;
            // a.download = 'composition.mp4';
            // a.click();
        })
        .then(() => setProcessing(false))
    }

    const onDownload = () => {
        if (responseData) {
            let a = document.createElement('a');
            a.href = responseData
            a.download = "composition.mp4";
            a.click()
        }
    }

    // This block of statements will redirect users to the normal matting page or the admin control page based on their role
    useEffect(() => {
        refreshToken();
        getUsers();
        if ({role}.role === "admin") {
            history("/admincontrol")
        }

    }, [history, role]);


    const refreshToken = async () => {
        try {
            const response = await axios.get('http://localhost:5000/token');
            setToken(response.data.accessToken);
            const decoded = jwt_decode(response.data.accessToken);
            setName(decoded.name);
            setRole(decoded.role);
            setExpire(decoded.exp);      
        } catch (error) {
            if (error.response) {
                history("/");
            }
        }
    }

    const axiosJWT = axios.create();
 
    axiosJWT.interceptors.request.use(async (config) => {
        const currentDate = new Date();
        if (expire * 1000 < currentDate.getTime()) {
            const response = await axios.get('http://localhost:5000/token');
            config.headers.Authorization = `Bearer ${response.data.accessToken}`;
            setToken(response.data.accessToken);
            const decoded = jwt_decode(response.data.accessToken);
            setName(decoded.name);
            setRole(decoded.role);
            setExpire(decoded.exp);
        }
        return config;
    }, (error) => {
        return Promise.reject(error);
    });
 
    const getUsers = async () => {
        const response = await axiosJWT.get('http://localhost:5000/users', {
            headers: {
                Authorization: `Bearer ${token}`    
            }
        });
        setUsers(response.data);
    }


    return (
        <Container>
            <Wrapper>
                <TitleMatting>{`Background Matting`}</TitleMatting>
                <Box>
                    <BoxContent>
                        {!videoFile ? <MyVideoDropzone alt="video" setVideoFile={setVideoFile} /> : (!videoError ? <VideoPlayer setVideoError={setVideoError} response={URL.createObjectURL(videoFile)} /> : 
                        <UploadContainer style={{fontSize: '20px', textAlign: 'center'}}>{`Error playing video due to browser limitation, but you can still proceed`}</UploadContainer>)}
                        {videoFile ? <FileDesc>{`Uploaded: ${videoFile.name}`}</FileDesc> : <></>}
                    </BoxContent>
                    <BoxContent>
                        {!imageFile ? <MyDropzone alt="background" setImageFile={setImageFile} /> : <img style={{aspectRatio: '16 / 9'}} src={URL.createObjectURL(imageFile)} alt="background" />}
                        {imageFile ? <FileDesc>{`Uploaded: ${imageFile.name}`}</FileDesc> : <></>}
                    </BoxContent>
                    <BoxContent>
                        {responseData ? 
                        (responseError ? <>
                            <video onError={() => setResponseError(true)} controls>
                                <source src={responseData} type="video/mp4" />
                            </video>
                            <img onClick={() => onDownload()} style={{width: '45px', height: '45px', cursor: 'pointer', marginTop: '10px'}} src={downloadIcon} alt="download" />
                        </> : 
                        <UploadContainer style={{fontSize: '20px', textAlign: 'center', display: 'flex', flexDirection: 'column'}}>
                            {`Error playing video due to browser limitation, but you can download it.`}
                            <img onClick={() => onDownload()} style={{width: '45px', height: '45px', cursor: 'pointer', marginTop: '10px'}} src={downloadIcon} alt="download" />
                        </UploadContainer>
                        )
                        :
                        <UploadContainer>
                            {processing ? <CircularLoader /> : 
                            <div style={{display: 'flex', flexDirection: 'column', alignItems: 'center'}}>
                                <p>Please upload a video to start the matting.</p>
                                <button style={{fontSize: '20px'}} onClick={handleVideo}>{`Start Matting`}</button>
                            </div>}
                        </UploadContainer>
                        }
                    </BoxContent>
                </Box>
            </Wrapper>
        </Container>
    )
}

const VideoPlayer = ({ response, setVideoError }) => {
    return (
      <video onError={() => setVideoError(true)} controls duration={5}>
        <source src={response} type='video/mp4' />
      </video>
    );
}

function MyVideoDropzone({ setVideoFile }) {
    const onDrop = useCallback((acceptedFiles, fileRejections) => {
        console.log("files rejected: ", fileRejections)
        console.log("files accepted: ", acceptedFiles)
        if (fileRejections.length !== 0) alert(fileRejections[0].errors[0].message)
        else {
            setVideoFile(acceptedFiles[0])
            console.log(acceptedFiles[0])
        }
    }, [])
    const {getRootProps, getInputProps, isDragActive} = useDropzone({
        onDrop,
        maxFiles: 1,
        accept: {
            'video/mp4': ['.mp4']
        }
    })
  
    return (
      <UploadContainer {...getRootProps()}>
        <input {...getInputProps()} />
        {
          isDragActive ?
            <p>Drop the video here ...</p> :
            <p>Drag 'n' drop the video here, or click to select video</p>
        }
      </UploadContainer>
    )
}

function MyDropzone({ setImageFile }) {
    const onDrop = useCallback((acceptedFiles, fileRejections) => {
      if (fileRejections.length !== 0) return alert(fileRejections[0].errors[0].message)
      else {
        setImageFile(acceptedFiles[0])
      }
    }, [])
    const {getRootProps, getInputProps, isDragActive} = useDropzone({
        onDrop,
        maxFiles: 1,
        accept: {
            'image/*': ['.png', '.jpg', '.jpeg']
        }
    })
  
    return (
      <UploadContainer {...getRootProps()}>
        <input {...getInputProps()} />
        {
          isDragActive ?
            <p>Drop the image here ...</p> :
            <p>Drag 'n' drop the image here, or click to select image</p>
        }
      </UploadContainer>
    )
}

function CircularLoader() {
    return (
      <div className="circular-loader">
        <div className="circular-loader-inner" />
      </div>
    );
}

const Playable = () => {
    let canPlay = false;
    let v = document.createElement('video');
    if (v.canPlayType && v.canPlayType('video/mp4').replace(/no/, '')) {
        canPlay = true;
    }
    return canPlay;
}
 
export default Matting;