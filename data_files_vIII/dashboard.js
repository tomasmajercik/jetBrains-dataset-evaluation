// imports
import React, { useEffect, useState, useRef } from 'react';
import { Navigate } from 'react-router-dom';
import $ from 'jquery';
import './dashboard.scss';

// components
import DragNdrop from "../components/DragNdrop";
import Modal from '../components/Modal';
import YesNoModal from '../components/YesNoModal';
import Filter from '../components/filter';


//images
import downloadIcon from '../imgs/downloadIcon.png';
import renameIcon from '../imgs/renameIco.png';
import deleteIcon from '../imgs/deleteIco.png';
import tagIcon from '../imgs/tagIco.png';
import filerIcon from '../imgs/filteIco.png';


export default function Dashboard() 
{
  const isAuthenticated = sessionStorage.getItem('isAuthenticated') === 'true';
  const token = sessionStorage.getItem('token');

  const [username, setUsername] = useState("");
  const [result, setResult] = useState("");

  const [filter, setFilter] = useState(false);
  const [filtredFiles, setFiltredFiles] = useState([]);
  const [selectedTags, setSelectedTags] = useState([]);

  const [files, setFiles] = useState([]);
  const [file, setFile] = useState(null);
  const inputRef = useRef();

  const [renamingFile, setRenamingFile] = useState(null);
  const [newFileName, setNewFileName] = useState("");
  const [showModal, setShowModal] = useState(false);
  
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [deletingFile, setDeletingFile] = useState(null);

  const [showTagModal, setShowTagModal] = useState(false);
  const [toBeTaggedFile, setToBeTaggedFile] = useState(false);
  const [userTags, setUserTags] = useState([]);
  const [newTag, setNewTag] = useState("");

  const [showFilterModal, setShowFilterModal] = useState(false);
  const [refreshTags, setrefreshTags] = useState(false);
  const [currentSelectTags, setCurrentSelectTags] = useState([]);

  useEffect(() => {
    if (isAuthenticated && token) 
    {
      $.ajax({
        type: "POST",
        url: 'http://localhost/webologyTaskPHP/backend/getUserInfo.php',
        data: {token},
        success(data) 
        {
          setResult(data);
          if (data.success)
          {
            setUsername(data.username);
            setFiles(data.files || []);
          }
          else
          {
            console.error('Failed to fetch user info:', data.message);
          }
        },
        error(xhr, status, error) {
          console.error('AJAX error:', status, error);
        }
      });
    }
  }, [isAuthenticated, token]);

  const handleFileChange = (event) => 
  {
    setFile(event.target.files[0]);
  };

  const handleFileUpload = () =>
  {
    if(file)
    {
      const formData = new FormData;
      formData.append('file', file);
      formData.append('token', token);
      formData.append('username', username);

      $.ajax({

          type: "POST",
          url: 'http://localhost/webologyTaskPHP/backend/uploadFile.php',
          data: formData,
          contentType: false,
          processData: false,

          success(data) {
            if (data.success) 
            {
              setFiles([...files, { file_name: file.name, file_path: `storage/${file.name}` }]);
              setFile(null);
            } 
            else 
            {
              console.error('File upload error:', data.message);
            }
          },
          error(xhr, status, error) {
            console.error('AJAX error:', status, error);
          }

      });

    }
  }

  const updateFilesList = (newFiles) => {
    setFiles(prevFiles => [...prevFiles, ...newFiles]);
  };

  //preventing files from downloading and not downloading
  const downloadFile = (path) =>
  {
    const fileName = path;
    console.log(fileName);
    const aTag = document.createElement("a");
    aTag.href = path;
    aTag.setAttribute("download", fileName);
    document.body.appendChild(aTag);
    aTag.click();
    aTag.remove();
  }
  //

  // rename file
  const renameFile = (fileName) =>
  {
    setRenamingFile(fileName);
    setNewFileName(fileName);
    setShowModal(true);
  }
  const handleRenameSave = (newName) =>
  {
    console.log("Renaming file:", renamingFile, "to", newName);
    setShowModal(false);

    $.ajax({
      type: "POST",
      url: 'http://localhost/webologyTaskPHP/backend/renameFile.php',
      data: { oldName: renamingFile, newName: newName, username },
      success: function(data) {
          if (data.success) {
              setFiles(files.map(file =>
                  file.file_name === renamingFile ? { ...file, file_name: newName } : file
              ));
              setFiltredFiles(filtredFiles.map(file =>
                file.file_name === renamingFile ? { ...file, file_name: newName } : file
              ));
          } else {
              console.error('Failed to rename file:', data.message);
          }
      },
      error: function(xhr, status, error) {
          console.error('AJAX error:', status, error);
          console.log("Response Text:", xhr.responseText);
      }
  });
  $.ajax({
    type: "POST",
    url: 'http://localhost/webologyTaskPHP/backend/renameFileTags.php',
    data: { oldName: renamingFile, newName: newName, username },
    success: function(data) {
        if (data.success) {
          setFiltredFiles(filtredFiles.map(file =>
            file.file_name === renamingFile ? { ...file, file_name: newName } : file
          ));
        } else {
            console.error('Failed to rename file:', data.message);
        }
    },
});
  }
  /**/
  //deleteFile
  const deleteFile = (fileName) => 
  {
    setDeletingFile(fileName);
    setShowDeleteModal(true);
  }
  const handleDelete = () => 
  {
    console.log("Deleting file:", deletingFile);
    setShowDeleteModal(false);

    $.ajax({
      type: "POST",
      url: 'http://localhost/webologyTaskPHP/backend/deleteFile.php',
      data: { fileName: deletingFile, username },
      success(data) {
        if (data.success) {
          setFiles(files.filter(file => file.file_name !== deletingFile));
        } else {
          console.error('Failed to delete file:', data.message);
        }
      },
      error(xhr, status, error) {
        console.error('AJAX error:', status, error);
      }
    });
    $.ajax({
      type: "POST",
      url: 'http://localhost/webologyTaskPHP/backend/deleteFileTags.php',
      data: { fileName: deletingFile, username },
      success(data) {
        if (data.success) {
          setFiles(filtredFiles.filter(file => file.file_name !== deletingFile));
        } else {
          console.error('Failed to delete file:', data.message);
        }
      },
      error(xhr, status, error) {
        console.error('AJAX error:', status, error);
      }
    });
  }
  /**/
  // tagging 
  const updateTags = (fileName) =>
  {
    pullFreshTags(fileName);
    setShowTagModal(true);
    setToBeTaggedFile(fileName);
  }  
  const handleTagToggle = (tag) => 
  {
    if (userTags.includes(tag)) {
        setUserTags(userTags.filter(t => t !== tag));
    } else {
        setUserTags([...userTags, tag]);
    }
  };
  const handleTagInputChange = (event) => 
  {
    setNewTag(event.target.value);
  };
  const handleAddTag = (tagName) =>
  {
    if (newTag && !userTags.includes(tagName)) 
    {
      setUserTags([...userTags, newTag]);
      setNewTag("");
    }
  } 
  const onClose = () =>
  {
    setShowTagModal(false);
  }
  const handleSave = () => {
    setShowTagModal(false);
  
    $.ajax({
      type: "POST",
      url: 'http://localhost/webologyTaskPHP/backend/updateTags.php',
      data: { username: username, fileName: toBeTaggedFile, tags: JSON.stringify(userTags) },
      success: function(data) {
        console.log("Server Response:", data);
        if (data.success) {
          setFiles(files.map(file =>
            file.file_name === toBeTaggedFile ? { ...file, tags: userTags } : file
          ));

          if(!refreshTags)
            setrefreshTags(true);
          else
            setrefreshTags(false);

        } else {
          console.error('Failed to update tags:', data.message);
        }
      },
      error: function(xhr, status, error) {
        console.error('AJAX error:', status, error);
        console.log("Response Text:", xhr.responseText);
      }
    });
  }
  const pullFreshTags = (fileName) =>
  {
    $.ajax({
      type: "POST",
      url: 'http://localhost/webologyTaskPHP/backend/getFileTags.php',
      data: { username: username, fileName: fileName },
      success(data) 
      {
        if (data.success) 
        {
          setUserTags(data.tags || []);
        } 
        else 
        {
          setUserTags([]);
        }
      },
      error(xhr, status, error) {
        console.error('AJAX error:', status, error);
      }
    });
  }

  // filter
  const startFilter = () => 
  {
    setShowFilterModal(true);
    if(!refreshTags)
      setrefreshTags(true);
    else
      setrefreshTags(false);
  }
  const showFiltredData = (tags) => {
    if (tags.length > 0) 
    {
      setCurrentSelectTags(tags);
      setFilter(true);
      setFiltredFiles([]);
      tags.forEach(tag => {
          $.ajax({
              type: "POST",
              url: 'http://localhost/webologyTaskPHP/backend/getFilteredFiles.php',
              data: { username, tag },
              success: function(data) 
              {
                  if (data.success) 
                  {
                    setFiltredFiles(prevFiles => [...prevFiles, ...data.files.map(file => ({ file_name: file }))]);
                  } 
                  else 
                  {
                      console.error('Error retrieving filtered files:', data.message);
                  }
              },
              error: function(xhr, status, error) {
                  console.error('AJAX error:', status, error);
              }
          });
      });
      setShowFilterModal(false);
    } 
    else 
    {
      setCurrentSelectTags([]);
      setShowFilterModal(false);
      setFilter(false);
      setFiltredFiles([]);
    }
};
  /**/

  if (!isAuthenticated) 
  {
    return <Navigate to="/login"/>;
  }

  // layout
  return(
    <>

      <div className='header'>
        <h1>Hello <strong> {username} </strong>, welcome to your dashboard!</h1>
      </div>
      
      <div className='filesBrowser'>
        <h2>Your files: </h2>

        <DragNdrop token={token} username={username} updateFilesList={updateFilesList} />

        <div className='documentList'>
          <div className='filterSpace'>

            {
            currentSelectTags.length > 0 ? (
              <h3 className='caption'>showing files inluding tags: {
                currentSelectTags.map((currentSelectTag, index) => (<span className='currentSelectTag' key={index}> {currentSelectTag}</span>) )}</h3>)
                :
                <h3>(no filter)</h3>
              
            }
            <button className='filter' onClick={startFilter}> <img src={filerIcon} alt="Delete" /> </button>
          </div>
          <table>
            <thead>
              <tr className='tableHeader'>
                <th>File Name</th>
                <th className='actionLine'>Actions</th>
              </tr>
            </thead>
            <tbody>
              {files.length === 0 ? (
                <tr>
                  <td colSpan="2">No files</td>
                </tr>
              ) : (

                filter ?
                filtredFiles.map((file, index) => (
                  <tr className='documentLine' key={index}>

                    <td className='nameRow'>{file.file_name}</td>
                    <td className='actionRow'>
                      <button className='downloadBTN' onClick={() => downloadFile(file.file_name)}> <img src={downloadIcon} alt='download'/> </button>
                      <button className='renameBTN' onClick={() => renameFile(file.file_name)}> <img src={renameIcon} alt="Rename"/> </button>
                      <button className='deleteBTN' onClick={() => deleteFile(file.file_name)}> <img src={deleteIcon} alt="Delete" /> </button>
                      <button className='addTagBTN' onClick={() => updateTags(file.file_name, file.tags)}> <img src={tagIcon} alt="Delete" /> </button>
                    </td>

                  </tr>

                ))
                :
                files.map((file, index) => (
                  <tr className='documentLine' key={index}>
                    <td className='nameRow'>{file.file_name}</td>
                    <td className='actionRow'>
                      <button className='downloadBTN' onClick={() => downloadFile(file.file_name)}> <img src={downloadIcon} alt='download'/> </button>
                      <button className='renameBTN' onClick={() => renameFile(file.file_name)}> <img src={renameIcon} alt="Rename"/> </button>
                      <button className='deleteBTN' onClick={() => deleteFile(file.file_name)}> <img src={deleteIcon} alt="Delete" /> </button>
                      <button className='addTagBTN' onClick={() => updateTags(file.file_name, file.tags)}> <img src={tagIcon} alt="Delete" /> </button>
                    </td>
                  </tr>
                ))


              )}
            </tbody>
          </table>
        </div>

      </div>

      <Modal
        show={showModal}
        onClose={() => setShowModal(false)}
        onSave={handleRenameSave}
        newFileName={newFileName}
        setNewFileName={setNewFileName}
        oldFileName={renamingFile}
      />

      <YesNoModal
        show={showDeleteModal}
        onClose={() => setShowDeleteModal(false)}
        onSave={handleDelete}
        fileName={deletingFile}
      />
      <Filter
        show={showFilterModal}
        onClose={() => setShowFilterModal(false)}
        onSave={showFiltredData}
        username = {username}
        refresh = {refreshTags}
      />

      {showTagModal && (

        // pullFreshTags(toBeTaggedFile),

        <div className='overlay'>
          <div className="modal">
            <h4>Update Tags for file "{toBeTaggedFile}"</h4>
            {/*checkboxes*/}
            <div className="tags-list">
                {userTags.map((tag, index) => (
                  <div key={index} className="tag-item checkBoxes">
                    <input
                      className="checkBox"
                      type="checkbox"
                      checked={userTags.includes(tag)}
                      onChange={() => handleTagToggle(tag)}
                    />
                    <span>{tag}</span>
                  </div>
                ))}
            </div>
            {/*add tag*/}
            <div className="add-tag">
              <input
                className='input'
                type="text"
                value={newTag}
                onChange={handleTagInputChange}
                placeholder="Enter new tag"
                onKeyUp={(e) => e.key === 'Enter' && handleAddTag()}
              />
              <button className='addButton' onClick={handleAddTag}>Add Tag</button>
            </div>
            {/*functional buttons*/}
            <div className="modal-actions">
              <button className='cancelButton' onClick={onClose}>Cancel</button>
              <button onClick={handleSave}>Save</button>
            </div>
          </div>
        </div>

      )}

    </>
  

  );
}