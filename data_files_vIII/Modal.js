import React, { useState, useEffect } from 'react';
import './modal.scss';

export default function Modal({ show, onClose, onSave, newFileName, setNewFileName, oldFileName })
{
  const [newName, setNewName] = useState("");
  const [extension, setExtension] = useState("");

  useEffect(() => {
    if (oldFileName) {
      const filenameSplit = oldFileName.split('.');
      const newName_ = filenameSplit[0];
      const extension_ = filenameSplit.slice(1).join('.') || ''; 

      setNewName(newName_);
      setExtension(extension_);
      setNewFileName(newName_); 
    }
  }, [oldFileName, setNewFileName]);

  const handleInputChange = (event) => {
    setNewName(event.target.value);
    setNewFileName(event.target.value);
  };

  const handleSave = () => {
    const fullNewFileName = `${newName}.${extension}`;
    onSave(fullNewFileName);
  };

  if (!show) {
    return null;
  }

  return (
    <div className="modal-overlay">
      <div className="modal">
        <h2>Rename File</h2>
        <input
          className='nameInput'
          type="text"
          value={newName}
          onChange={handleInputChange}
          placeholder="Enter new name (without extension)"
        />
        <div className="modal-actions">
          <button className='cancelRenameDelButton' onClick={onClose}>Cancel</button>
          <button className='renameDelButton' onClick={handleSave} disabled={!newName}>
            Save
          </button>
        </div>
      </div>
    </div>
  );
}
