import { useEffect, useState, useRef } from 'react';
import './background.css';
import { database } from '../config/firebaseConfig';
import { useNavigate } from 'react-router-dom';
import { collection, getDocs, doc, updateDoc, onSnapshot } from 'firebase/firestore';

import AddPlayer from '../components/addPlayer';

//imgs
import loadedGif from '../imgs/loaded.gif';
import childrenBackground from '../imgs/childrenBackground.jpg';


function Background() 
{
  // Variables
  localStorage.setItem('preventFromReturn', 'true');
  const [childrenList, setChildrenList] = useState([]);
  const [error, setError] = useState("");
  const [operation, setOperation] = useState('add');

  const [showModal, setShowModal] = useState(false);

  const [showGif, setShowGif] = useState(false);

  const [selectedChild, setSelectedChild] = useState(null);
  const [pointsCount, setPointsCount] = useState(0);

  const navigate = useNavigate();
  // Functions
  
  useEffect(() => {
    const autentif = localStorage.getItem('autentif');
    if (autentif !== 'true') 
    {
      navigate("/");
    }
  }, [navigate]);


  useEffect(() => {
    const unsubscribe = onSnapshot(collection(database, "score"), (snapshot) => {
      const childrenData = snapshot.docs.map((doc) => ({
        ...doc.data(),
        id: doc.id,
      }));
      const sortedData = childrenData.sort((a, b) => 
        a.name.localeCompare(b.name)
      );
      setChildrenList(sortedData);
    });
  
    return () => unsubscribe(); // Cleanup listener on component unmount
  }, []);
  

  const selectBtnClick = (child) =>
  {
    setSelectedChild(child);
  }

  
  const toggleModal = () => {
    setShowModal(!showModal);
  };

  const addPoint = async (child) => {
    if (child && child.id) {
      try {
        const childDocRef = doc(database, "score", child.id);
        const currentPoints = child.points;
        const newPoints = currentPoints + 1;
  
        await updateDoc(childDocRef, { points: newPoints });
  
        setShowGif(true);
        setTimeout(() => {
          setShowGif(false);
        }, 1800);
  
        setError("");
        setPointsCount(0);
      } catch (err) {
        setError("Failed to update points.");
        console.error(err);
      }
    } else {
      setError("Žiadne dieťa nebolo vybrané.");
    }
  };
  
  const addPoints = async (child) => {
    if (child && child.id) {
      try {
        const childDocRef = doc(database, "score", child.id);
        const currentPoints = child.points;
        const newPoints = currentPoints + 2;
  
        await updateDoc(childDocRef, { points: newPoints });
  
        setShowGif(true);
        setTimeout(() => {
          setShowGif(false);
        }, 1800);
  
        setError("");
        setPointsCount(0);
      } catch (err) {
        setError("Failed to update points.");
        console.error(err);
      }
    } else {
      setError("Žiadne dieťa nebolo vybrané.");
    }
  };
  
  

  

  // Layout
  return (
    <>

    <button className='addPlayer' onClick={toggleModal}>+</button>

    <AddPlayer 
        showModal={showModal} 
        toggleModal={toggleModal} 
    />
      
    
      <div 
        className='title' 
        style={{
          backgroundImage: `url(${childrenBackground})`,
          backgroundPosition: 'center',
          backgroundSize: 'cover',
          backgroundRepeat: 'no-repeat',
          color: 'white',
          textAlign: 'center'
        }}
      >
        <h1 className='bckH1'>Stanovište</h1>
      </div>
    
      <div className='PointsContainer'>
          


      </div>


      <h4 className='bckH4'>vyberte dieťa:</h4>

      <div className='consoleTable'>
        <table id='childrenTableAdmin'>
          <thead>
            <tr>
              <th>Meno hráča</th>
              <th>Pridať</th>
              <th>Počet bodov</th>
            </tr>
          </thead>
          <tbody>
            {
              childrenList.map((children) => (
                <tr key={children.id}>
                  <td className='limiter'>{children.name}</td>
                  <td>
                    <button 
                      className='addBtn' 
                      onClick={() => addPoint(children)}
                    >
                      +1
                    </button>
                    <button 
                      className='addMoreBtn' 
                      onClick={() => addPoints(children)}
                    >
                      +2
                    </button>
                  </td>
                  <td>{children.points}</td>
                </tr>                
              ))
            }
          </tbody>
        </table>
        
      </div>


      {/* <div className='childpool'>
          {
              childrenList.map((children) => (
                  <button key={children.id} className='selectBtn' onClick={() => selectBtnClick(children)}>{children.name} : {children.points}</button>
              ))
          }
      </div> */}


      {/* <button 
          className='changePointsButton' 
          type='submit'
          onClick={handleSubmit}
      >
          Zmeniť!
      </button> */}

      {error && <p className='error'>{error}</p>}
    </>
  );
}

export default Background;