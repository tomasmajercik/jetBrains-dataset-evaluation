import React, { useState } from 'react';
import './addPlayer.css';
import { collection, addDoc } from 'firebase/firestore';
import { database } from '../config/firebaseConfig'; // Assuming you have a firebaseConfig.js where you initialize Firebase


function AddPlayer({ showModal, toggleModal }) 
{
  const [newPlayerName, setNewPlayerName] = useState("");

  const handleAddPlayer = async () => 
  {
    if (newPlayerName) 
    {
      try 
      {
        await addDoc(collection(database, 'score'), 
        {
          name: newPlayerName,
          points: 0,
        });
  
        setNewPlayerName("");
        toggleModal();
      } 
      catch (error) 
      {
        console.error("Error adding player: ", error);
      }
    }
  };
  

  if (!showModal) return null;

  return (
    <div className="modal">
      <div className="modal-content">
        <h2 className='addH2'>Pridať hráča:</h2>
        <input
          className='addInput'
          type="text"
          value={newPlayerName}
          onChange={(e) => setNewPlayerName(e.target.value)}
          placeholder="Zadajte meno nového hráča"
        />
        <br/>
        <button className='addbtn' onClick={handleAddPlayer}>Pridať</button>
        <br/>
        <button className='addbtn2' onClick={toggleModal}>Zrušiť</button>
      </div>
    </div>
  );
}

export default AddPlayer;
