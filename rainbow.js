//rainbow.js

	function rainbow()
	{	
		var hue = (performance.now()/100)%1; //variable hue -- uses performance.now method to clarify data as a time measurement, then divides the time by 20000 milliseconds to make 20 seconds and for every %1 colour changes.
		document.getElementsByClassName("headerlol").style.backgroundColor = "hsl("+(hue*360)+", 25%, 50% )"; // find document element by id specified, set background colour to use hsl colour spectrum, hue multiplied by 360 to represent colour wheel, sat and lightness is in percentages. 
		window.requestAnimationFrame(continousUpdt); //calls the function so this runs continuously.
	}
	
	window.addEventListener("load", function() //listens for page load, then continousUpdt is called. 
	{
	window.requestAnimationFrame(continousUpdt); 
	});
	