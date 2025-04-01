import React, { useState, useEffect } from 'react';
import $ from 'jquery';
import './filter.scss';

export default function Modal({ show, onClose, onSave, username, refresh })
{

    const [allUserTags, setAllUserTags] = useState([]);
    const [userTags, setUserTags] = useState([]);
    const [selectedTags, setSelectedTags] = useState([]);
  
    const clearDuplicates = (arr) =>
    {
        return( arr.filter((item, index) => arr.indexOf(item) === index) );
    }

    useEffect(() =>{
        setSelectedTags([]);

        $.ajax({
            type: "POST",
            url: 'http://localhost/webologyTaskPHP/backend/getAllTags.php',
            data: { username },
            success(data) 
            {
              if (data.success) 
              {
                setAllUserTags(data.tags || []);
              } 
              else 
              {
                setAllUserTags([]);
              }
            },
            error(xhr, status, error) {
              console.error('AJAX error:', status, error);
            }
          });

    }, [refresh, username]);
    useEffect(() => {
        setUserTags(clearDuplicates(allUserTags));
      }, [allUserTags]);

    const handleTagToggle = (tag) =>
    {
        if (userTags.includes(tag))
        {
            setUserTags(userTags.filter(t => t !== tag)); // pop
            setSelectedTags([...selectedTags, tag]); // push
        }
        else if(selectedTags.includes(tag))
        {
            setSelectedTags(selectedTags.filter(t => t !== tag)); // pop
            setUserTags([...userTags, tag]); // push
        }
    }

    const handleSave = () =>
    {
        onSave(selectedTags);
    }

  if (!show) {
    return null;
  }
    return (
        <div className="modal-overlay">
        <div className="modal">
            <h2>Show files including this tags:</h2>
            <div className="Checkedlist">
                <h4>selected:</h4>
                {selectedTags.map((tag, index) => (
                    <div key={index} className="tag-item">
                        <button className='tagButtonSel' onClick={() => handleTagToggle(tag)}><span>{tag}</span></button> 
                    </div>
                ))}
            </div>
            <div className="UnCheckedlist">
                <h4>not selected:</h4>
                {userTags.map((tag, index) => (
                    <div key={index} className="tag-item">
                        <button className='tagButton' onClick={() => handleTagToggle(tag)}><span>{tag}</span></button> 
                    </div>
                ))}
            </div>
            <div className="modal-actions">
            <button className='cancelButton' onClick={onClose}>Cancel</button>
            <button className='filterButton' onClick={handleSave}>
                Filter!
            </button>
            </div>
        </div>
        </div>
    );
}
