import React, { useState, useEffect } from "react";
import "../ppp.css";

function Sneha() {
  const [data, setdata] = useState({
    prediction:""
    });
    
    useEffect(() => {
      fetch("http://127.0.0.1:5000/predict", {
        method: 'GET',
      })
      .then((res) => res.json()
      .then((data) => { setdata({
                    prediction:data.prediction
                  });
                })
            );
        }, []);
        return (

            <div className='Sneha'>
              <div className='header'>
                <h1> Movie Review Classifier</h1>
              </div>
              <div className="result">
              <p>{data.prediction}</p>
              </div>
  </div>
  );
}

export default Sneha;