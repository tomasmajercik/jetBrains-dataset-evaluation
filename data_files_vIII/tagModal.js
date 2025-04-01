import React, { useState, useEffect } from 'react';
import './modal.scss';

export default function TagModal({ show, onClose, onSave, fileName, initialTags = [] }) {
    const [tags, setTags] = useState([]);
    const [newTag, setNewTag] = useState("");

    // // Use useEffect to set tags whenever initialTags changes
    // useEffect(() => {
    //     setTags(initialTags);
    // }, [initialTags]);

    const handleTagInputChange = (event) => {
        setNewTag(event.target.value);
    };

    const handleAddTag = () => {
        if (newTag && !tags.includes(newTag)) {
            setTags([...tags, newTag]);
            setNewTag("");
        }
    };

    const handleTagToggle = (tag) => {
        if (tags.includes(tag)) {
            setTags(tags.filter(t => t !== tag));
        } else {
            setTags([...tags, tag]);
        }
    };

    const handleSave = () => {
        onSave(tags);
    };

    if (!show) {
        return null;
    }

    return (
        <div className="modal-overlay">
            <div className="modal">
                <h2>Update Tags for file "{fileName}"</h2>

                <div className="tags-section">
                    <h3>Tags</h3>
                    <div className="tags-list">
                        {tags.map((tag, index) => (
                            <div key={index} className="tag-item">
                                <input
                                    type="checkbox"
                                    checked={tags.includes(tag)}
                                    onChange={() => handleTagToggle(tag)}
                                />
                                <span>{tag}</span>
                            </div>
                        ))}
                    </div>
                </div>
                <div className="add-tag">
                    <input
                        type="text"
                        value={newTag}
                        onChange={handleTagInputChange}
                        placeholder="Enter new tag"
                    />
                    <button onClick={handleAddTag}>Add Tag</button>
                </div>

                <div className="modal-actions">
                    <button onClick={onClose}>Cancel</button>
                    <button onClick={handleSave}>Save</button>
                </div>
            </div>
        </div>
    );
}
