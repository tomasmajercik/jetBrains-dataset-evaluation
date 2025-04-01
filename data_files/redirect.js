import React, { useEffect, useRef, useState } from 'react';
import './redirect.css';
import { useNavigate } from 'react-router-dom';
// imgs
import loadingGif from '../imgs/loading.gif';
import loadedGif from '../imgs/loaded.gif';
//interface-s

function Redirect()
{
  const navigate = useNavigate(); 


  useEffect(() => {
    const autentif = localStorage.getItem('autentif');
    if (autentif !== 'true') 
    {
      navigate("/");
    }
  }, [navigate]);

  useEffect(() => {
    const preventFromReturn = localStorage.getItem('preventFromReturn');
    if (preventFromReturn == 'true') 
    {
      navigate("/");
    }
  }, [navigate]);

  //variables
  const [loaded, setLoaded] = useState(false);

  //functions

    useEffect(() => 
    {
        // Set a timeout
        const timer = setTimeout(() => {
            setLoaded(true); 
            const timer2 = setTimeout(() => {
                navigate('/background');
            }, 1800)
        }, 3000);

        // Cleanup the timer when the component is unmounted
        return () => clearTimeout(timer);
    }, []);
  
  //layout
  return (
      <>
        {loaded ? (
            <div className='main'>
                <img className='biger' src={loadedGif} alt="" />
            </div>
        ) : (
            <div className='main'>
                <img src={loadingGif} alt="" />
            </div>
        )}
      </>
  );
};

export default Redirect;