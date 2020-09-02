'use strict';
const NodeHelper = require('node_helper');
const { spawn, exec } = require('child_process');

var cAppStarted = false

module.exports = NodeHelper.create({

	cApp_start: function () {
		const self = this;
		self.objectDet = spawn('modules/' + this.name + '/object_detection_ompss/build/start.sh',['modules/' + this.name + '/object-detection/build', self.config.image_width, self.config.image_height]);
		self.objectDet.stdout.on('data', (data) => {

			var data_chunks = `${data}`.split('\n');
			data_chunks.forEach( chunk => {

				if (chunk.length > 0) {
	
				try{
					var parsed_message = JSON.parse(chunk)

					if (parsed_message.hasOwnProperty('DETECTED_OBJECTS')){
						//console.log("[" + self.name + "] Gestures detected : " + parsed_message);
						self.sendSocketNotification('DETECTED_OBJECTS', parsed_message);
					}else if (parsed_message.hasOwnProperty('OBJECT_DET_FPS')){
						//console.log("[" + self.name + "] " + JSON.stringify(parsed_message));
						self.sendSocketNotification('OBJECT_DET_FPS', parsed_message.OBJECT_DET_FPS);
					}else if (parsed_message.hasOwnProperty('STATUS')){
						console.log("[" + self.name + "] status received: " + JSON.stringify(parsed_message));
					}
				}
				catch(err) {	
					//console.log(err)
				}
  				//console.log(chunk);
				}
			});
		});	



		/*exec(`renice -n 20 -p ${self.gestureDet.pid}`,(error,stdout,stderr) => {
				if (error) {
					console.error(`exec error: ${error}`);
  				}
			}); */

		},

		/*	var data_chunks = `${data}`.split('\n');
			data_chunks.forEach( chunk => {

				if (chunk.length > 0) {

			//console.log(`${data}`)
			try{
				var parsed_message = JSON.parse(chunk)

				if (parsed_message.hasOwnProperty('DETECTED_OBJECTS')){
					console.log("[" + self.name + "] detected object: " + parsed_message.detected.name + " center in "  + parsed_message.detected.center);
					self.sendSocketNotification('DETECTED_OBJECTS', parsed_message);
				}else if (parsed_message.hasOwnProperty('OBJECT_DET_FPS')){
					console.log("[" + self.name + "] object detection fps: " + JSON.stringify(parsed_message));
					self.sendSocketNotification('OBJECT_DET_FPS', parsed_message.OBJECT_DET_FPS);
				}else if (parsed_message.hasOwnProperty('STATUS')){
					console.log("[" + self.name + "] status received: " + JSON.stringify(parsed_message));
				}
			}
			catch(err) {	
					//console.log(err)
				}
  				//console.log(chunk);
				}
			});
		});	


	
		/*exec(`renice -n 20 -p ${self.objectDet.pid}`,(error,stdout,stderr) => {
				if (error) {
					console.error(`exec error: ${error}`);
  				}
			}); */

  	//}, 

  	// Subclass socketNotificationReceived received.
  	socketNotificationReceived: function(notification, payload) {
		const self = this;	
		if(notification === 'ObjectDetection_SetFPS') {
			if(cAppStarted) {
				self.objectDet.stdin.write(payload.toString() + "\n");
				console.log("[" + self.name + "] changing to: " + payload.toString() + " FPS");

         		}
       	 	}else if(notification === 'OBJECT_DETECITON_CONFIG') {
      			self.config = payload
      			if(!cAppStarted) {
        			cAppStarted = true;
        			this.cApp_start();
      			};
    		};
  	},

	stop: function() {
		const self = this;	
		
	}
});
