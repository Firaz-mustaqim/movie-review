

function App() {
  return (
	  <div>
    	  <div className="comment">
      		if (prediction==1){
            	<h1>Positive Comment</h1>
			  }
     		elif (prediction==0){    
        		<h1>Negative Comment</h1>
			 }
    		endif{ 
            	<h1>Neutral Comment</h1>
			}
      
 			</div>
	  </div>
  );
}

export default App;
