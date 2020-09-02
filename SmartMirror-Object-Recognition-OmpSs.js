/**
 * @file SmartMirror-Object-Recognition-OmpSs.js
 *
 * @author nkucza
 * @license MIT
 *
 * @see  TODO
 */

Module.register('SmartMirror-Object-Recognition-OmpSs',{

	defaults: {
		image_height: 1080,
		image_width: 1920,
		image_stream_path: "/dev/shm/camera_image"
	},

	start: function() {
		this.time_of_last_greeting_personal = [];
		this.time_of_last_greeting = 0;
		this.last_rec_user = [];
		this.current_user = null;
		this.sendSocketNotification('OBJECT_DETECITON_CONFIG', this.config);
		Log.info('Starting module: ' + this.name);
	},

	notificationReceived: function(notification, payload, sender) {
		if(notification === 'smartmirror-object-detectionSetFPS') {
			this.sendSocketNotification('ObjectDetection_SetFPS', payload);
        }
	},


	socketNotificationReceived: function(notification, payload) {
		if(notification === 'detected') {
			this.sendNotification('OBJECT_DETECTED', payload);
			//console.log("[" + this.name + "] " + "object detected: " + payload);
        } else if(notification === 'DETECTED_OBJECTS') {
			this.sendNotification('DETECTED_OBJECTS', payload);
			//console.log("[" + this.name + "] " + "object detected: " + payload);
        }else if (notification === 'OBJECT_DET_FPS') {
			this.sendNotification('OBJECT_DET_FPS', payload);
		};
	}
});
