import React, { useRef, useState, useEffect } from 'react';
import $, { event } from 'jquery';
import "./dragNdrop.scss";

export default function DragNdrop({ token, username, updateFilesList }) 
{
    const inputRef = useRef();
    const [files, setFiles] = useState(null);
    const [animation, setAnimation] = useState(false);
    const [dragCounter, setDragCounter] = useState(0);
    const [showForm, setShowForm] = useState(false);
    const [showFormCounter, setshowFormCounter] = useState(0);

    const handleDragOver = (event) => 
    {
        event.preventDefault();
    };


    //animation when drag file is over
    const handleDragEnter = (event) => 
    {
        event.preventDefault();
        setDragCounter(prev => prev + 1);
        if (!animation) setAnimation(true);
    };
    const handleDragLeave = (event) => 
    {
        event.preventDefault();
        setDragCounter(prev => prev - 1);
    };
    useEffect(() => 
    {
        if (dragCounter === 0) {
            setAnimation(false);
        }
    }, [dragCounter]);

    useEffect(() => 
    {
        if(showFormCounter === 0)
        {
            setShowForm(false);
        }
    }, [showFormCounter]);

    const handleDrop = (event) => {
        event.preventDefault();
        setFiles(event.dataTransfer.files);
        setDragCounter(0);
        setAnimation(false);
    };
    //


    const handleUpload = () => {
        const formData = new FormData();
        Array.from(files).map(file => {
            formData.append('file[]', file);
        });
        formData.append('token', token);
        formData.append('username', username);

        $.ajax({
            type: "POST",
            url: 'http://localhost/webologyTaskPHP/backend/uploadFile.php',
            data: formData,
            contentType: false,
            processData: false,

            success(data) {
                if (data.success) {
                    updateFilesList(Array.from(files).map(file => ({
                        file_name: file.name,
                        file_path: `storage/${file.name}`
                    })));
                    setFiles(null);
                } else {
                    console.error('File upload error:', data.message);
                }
            },
            error(xhr, status, error) {
                console.error('AJAX error:', status, error);
            }
        });
    };



    if (files) {
        return (
            <div className="uploads">
                <ul>
                    {Array.from(files).map((file, idx) => <li key={idx}>{file.name}</li>)}
                </ul>
                <div className="actions">
                    <button className='dragNdropBtn cancelButton' onClick={() => setFiles(null)}>Cancel</button>
                    <button className='dragNdropBtn' onClick={handleUpload}>Upload</button>
                </div>
            </div>
        );
    }

    return (
        <>
        <div 
            className={`dropzone ${animation ? "animate" : ""}`}
            onDragOver={handleDragOver}
            onDragEnter={handleDragEnter}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
        >
          { animation ? 
          (
            <h1 className={`${animation ? "animateFont" : ""}`}>Drop Files to Upload</h1>
          ) : (<h1 className={`${animation ? "animateFont" : ""}`}>Drag, hold and Drop Files to Upload</h1>

          )

          }
          <h1 className={`${animation ? "fontGone" : ""}`}>Or</h1>
          <input 
            type="file"
            multiple
            onChange={(event) => setFiles(event.target.files)}
            hidden
            ref={inputRef}
            className={`${animation ? "fontGone" : ""}`}
          />
          <button className={`dragNdropBtn ${animation ? "fontGone" : ""}`} onClick={() => inputRef.current.click()}>Select Files</button>
        </div>


        
     

        </>
    );
}