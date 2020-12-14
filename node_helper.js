'use strict';
const NodeHelper = require('node_helper');
const { spawn, exec } = require('child_process');

var cAppStarted = false

module.exports = NodeHelper.create({

cApp_start: function () {
	const self = this;
	self.objectDet = spawn('modules/' + self.name + '/object_detection_ompss/build/startYoloTRT.sh',['modules/' + self.name + '/object-detection/build', self.config.image_width, self.config.image_height]);
	self.objectDet.stdout.on('data', (data) => {

	var data_chunks = `${data}`.split('\n');
	data_chunks.forEach( chunk => {

		if (chunk.length > 0) {
			try{
				var parsed_message = JSON.parse(chunk)
				if (parsed_message.hasOwnProperty('DETECTED_OBJECTS')){
					//console.log("[" + self.name + "] Objects detected : " + JSON.stringify(parsed_message));
					self.sendSocketNotification('DETECTED_OBJECTS', parsed_message);
				}else if (parsed_message.hasOwnProperty('OBJECT_DET_FPS')){
					//console.log("[" + self.name + "] " + JSON.stringify(parsed_message));
					self.sendSocketNotification('OBJECT_DET_FPS', parsed_message.OBJECT_DET_FPS);
				}else if (parsed_message.hasOwnProperty('STATUS')){
					console.log("[" + self.name + "] status received: " + JSON.stringify(parsed_message));
				}
			}
			catch(err) {	
				if (err.message.includes("Unexpected token") && err.message.includes("in JSON")){
					console.log("[" + self.name + "] json parse error");
				} else if (err.message.includes("Unexpected end of JSON input")) {
					console.log("[" + self.name + "] Unexpected end of JSON input")
					//console.log(chunk);
				} else {
					console.log(err.message)
				}
			}
			//console.log(chunk);
		}
	});
	});	

	exec(`renice -n 20 -p ${self.objectDet.pid}`,(error,stdout,stderr) => {
		if (error) { console.error(`exec error: ${error}`);}
	});

	self.objectDet.stderr.on('data', (data) => {
		console.error(`stderr: ${data}`);
	});


	self.objectDet.on("exit", (code, signal) => {
		if (code !== 0){
			setTimeout(() => {self.cApp_start();}, 5)
		}				
	console.log("object det:");
	console.log("code: " + code);
	console.log("signal: " + signal);

	});

},

// Subclass socketNotificationReceived received.
socketNotificationReceived: function(notification, payload) {
	const self = this;	
	if(notification === 'ObjectDetection_SetFPS') {
		if(cAppStarted) {
			try{
				self.objectDet.stdin.write(payload.toString() + "\n");
				console.log("[" + self.name + "] changing to: " + payload.toString() + " FPS");
			}
			catch(err){
				console.log(err)
				console.log(err.name)
				console.log(err.message)
			}
		}
	}else if(notification === 'OBJECT_DETECITON_CONFIG') {
		self.config = payload
		if(!cAppStarted) {
			cAppStarted = true;
			self.cApp_start();
		};
	};
},

stop: function() {
	const self = this;	
		
}
});
