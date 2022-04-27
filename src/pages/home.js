import React from 'react'

class Home extends React.Component{
    render(){
        return(
        <div>
            <div class="container">
    		<h2 class='container-heading'><span>Movies Reviews Classifier</span> </h2>
            <div class='description'>
    			<p>A Machine Learning Web App which is implemented using Flask</p>
    		</div>
    	</div>

    	<div class="ml-container">
    		<form action="http://127.0.0.1:5000/predict" method="POST">
        		<textarea class='message-box' name="message" rows="15" cols="75" placeholder="Enter Your Message Here... (ex: This is the worst movie, i have ever seen in my life  OR  This is the wonderful movie, i have ever seen in my life)"></textarea><br/>
        		<input type="submit" class="my-cta-button" value="Predict" />
        	</form>
    	</div>
        </div>
        )
    }
}

export default Home;
